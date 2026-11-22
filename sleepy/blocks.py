"""Diffusion-block training, and the measurement that says whether to use it.

A transformer's residual stream is already a denoising trajectory: each layer
makes a small correction to a state it does not replace. Diffusion Blocks
(Sakana) takes that literally — slice the depth into contiguous blocks, give each
block its own noise range, and train each block against its own local
turbation-to-clean problem. Nothing flows backward across a block boundary, so
peak activation memory and the backward graph both shrink by roughly the block
count instead of covering the whole depth.

Two things the paper's headline number hides, and this module refuses to:

* The saving is `1/blocks` of the *whole model*, and only against an end-to-end
  run of the same length. Split into six blocks and train each for a sixth of
  the steps and you saved nothing. `ab()` compares at matched step count and
  reports both the memory ratio and the cost it bought.
* A block needs depth to denoise anything. In the paper, 2 blocks beat end-to-end
  and 3+ degrade; by 6 the split is worse than not splitting. `ab()` can
  therefore return `e2e` on a fine split, and `savings()` projects the ratio a
  split would actually deliver.

Every boundary here is `detach()`ed. The target is the borrowed prefix's hidden
state and the gradient is local, so the model is exactly the union of two
independently trained halves — the regime the memory claim is about, and the one
where the end-to-end baseline is genuinely a different (leakier) procedure.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .buffer import Item, render
from .sleep import SleepConfig, encode


class UnsupportedModel(RuntimeError):
    """The model has no generic layer list to slice, so blocks are not defined."""


@dataclass
class BlockConfig:
    blocks: int = 3
    lr: float = 5e-5
    batch_items: int = 4
    steps: int = 12                 # per variant; both variants get the same count
    min_block_layers: int = 2       # depth below which a block cannot denoise
    sigma_lo: float = 0.05
    sigma_hi: float = 0.60          # below the paper's 1.0: LM embeddings are not images
    gate_kl_max: float = 0.10
    gate_core_tol: float = 0.03
    min_memory_gain: float = 0.25   # a split must buy at least 25% peak activation
    max_quality_loss: float = 0.02   # ...without costing more than 2% end-to-end loss
    # Seeds the sigma draw *and* the corruption noise, so the same seed and step
    # count replay a run exactly. Both arms of an A/B are trained from clones of
    # one state with this seed, which is what makes "matched steps" honest.
    seed: int = 0


@dataclass
class BlockReport:
    blocks: int = 0
    block_layers: int = 0
    steps: int = 0
    e2e_loss: float = float("nan")
    block_loss: float = float("nan")
    e2e_peak: float = float("nan")
    block_peak: float = float("nan")
    e2e_ratio: float = float("nan")   # mean ‖ΔW‖/‖W‖ over touched matrices
    block_ratio: float = float("nan")
    gate_ok: bool = False
    reason: str = ""

    KEYS = ("blocks", "block_layers", "steps", "e2e_loss", "block_loss",
            "e2e_peak", "block_peak", "e2e_ratio", "block_ratio", "gate_ok", "reason")

    def row(self) -> dict:
        return {k: getattr(self, k) for k in self.KEYS}

    def quality_loss(self) -> float:
        """Relative loss the split pays against the end-to-end run."""
        if not (_finite(self.e2e_loss) and _finite(self.block_loss)) or self.e2e_loss <= 0:
            return float("nan")
        return (self.block_loss - self.e2e_loss) / self.e2e_loss

    def memory_gain(self) -> float:
        if not (_finite(self.e2e_peak) and _finite(self.block_peak)) or self.e2e_peak <= 0:
            return float("nan")
        return 1.0 - self.block_peak / self.e2e_peak


def _finite(x: float) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


# -- structure -------------------------------------------------------------

def base_model(pm):
    """The HF causal LM at the top of a PeftModel, adapter wrappers and all.

    A bare causal LM works too: `run_both` is meaningful on an unadapted model,
    and every call below only ever reaches for `.model.layers`.
    """
    unwrap = getattr(pm, "get_base_model", None)
    return unwrap() if unwrap else pm


def layers_of(pm) -> torch.nn.ModuleList:
    inner = getattr(base_model(pm), "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None or not len(layers):
        raise UnsupportedModel(
            "no `.model.layers` to split; blocks support Llama/DIAL-style causal LMs only")
    return layers


def split_into(n: int, blocks: int) -> list[tuple[int, int]]:
    """Contiguous `[lo, hi)` spans covering `n` layers as evenly as possible."""
    blocks = max(1, min(blocks, n))
    base, extra = divmod(n, blocks)
    spans, lo = [], 0
    for i in range(blocks):
        hi = lo + base + (1 if i < extra else 0)
        spans.append((lo, hi))
        lo = hi
    return spans


def sigma_span(i: int, n: int, lo: float, hi: float) -> tuple[float, float]:
    """The noise band block `i` owns — high (noisy) first, low (clean) last."""
    edges = [hi - (hi - lo) * k / n for k in range(n + 1)]
    return edges[i + 1], edges[i]


def noise_like(x: torch.Tensor, seed: int | None, step: int, span: int) -> torch.Tensor:
    """`randn` shaped and typed like `x`, on `x`'s device.

    Two separate defects meet here. The obvious one: `torch.randn(x.shape)` on
    its own is a CPU float32 tensor, and adding it to a CUDA or fp16 state
    raises. The quieter one: unseeded noise makes "matched step count" only
    half a comparison — the sigmas were seeded but each arm got different
    draws, so a loss difference between arms was partly luck.

    The seed is keyed by `(seed, step, span)` rather than advanced through a
    shared generator, so call order cannot shift it: a step's noise is the same
    whichever block it happens to run in, in either arm, and no state has to
    be threaded through `_run_variant`. `seed=None` leaves it unseeded, which
    is the same draw-from-the-global-RNG the callers had before.
    """
    if seed is None:
        return torch.randn(x.shape, device=x.device, dtype=x.dtype)
    gen = torch.Generator(device=x.device).manual_seed(seed * 1_000_003 + step * 1_009 + span)
    return torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)


def plan(n_layers: int, cfg: BlockConfig, steps: int) -> list[tuple[int, int, float]]:
    """Per-step `(lo, hi, sigma)`: blocks round-robin, each inside its own band.

    Round-robin rather than block-major so a short budget still visits every
    block; `steps` is per variant, so both arms of an A/B get the same count.
    """
    spans = split_into(n_layers, cfg.blocks)
    rng = random.Random(cfg.seed)
    out = []
    for step in range(steps):
        i = step % len(spans)
        s_lo, s_hi = sigma_span(i, len(spans), cfg.sigma_lo, cfg.sigma_hi)
        out.append((spans[i][0], spans[i][1], rng.uniform(s_lo, s_hi)))
    return out


# -- the forward halves ----------------------------------------------------

class _Chain:
    """Embeddings, causal mask and rotary, shared by both halves of one forward."""

    def __init__(self, pm, ids: torch.Tensor, mask: torch.Tensor | None):
        base = base_model(pm)
        inner = base.model
        dev = base.lm_head.weight.device
        self.emb = inner.embed_tokens(ids)
        self.pos = torch.arange(ids.shape[1], device=dev).unsqueeze(0)
        try:
            from transformers.masking_utils import create_causal_mask
        except ImportError as e:  # older/newer transformers moved it
            raise UnsupportedModel(f"create_causal_mask unavailable: {e}") from e
        self.mask = create_causal_mask(base.config, self.emb, mask, None, self.pos)
        self.rot = inner.rotary_emb(self.emb, position_ids=self.pos)
        self.layers = list(inner.layers)
        self.norm = inner.norm
        self.head = base.lm_head

    def run(self, layers, h: torch.Tensor, lo: int) -> torch.Tensor:
        for layer in layers:
            h = layer(h, attention_mask=self.mask, position_ids=self.pos,
                      position_embeddings=self.rot, use_cache=False)
        return h

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(h))


def _batch(pm, b: dict, chain_cls=_Chain) -> tuple[_Chain, torch.Tensor, torch.Tensor]:
    ids = b["input_ids"]
    return chain_cls(pm, ids, b.get("attention_mask")), ids, b["labels"]


def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100)


def forward_full(pm, b: dict) -> torch.Tensor:
    chain, _, labels = _batch(pm, b)
    return _loss(chain.logits(chain.run(chain.layers, chain.emb, 0)), labels)


def block_logits(pm, b: dict, lo: int, hi: int, sigma: float,
                 seed: int | None = None, step: int = 0,
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """`forward_block`, but returning `(logits, labels)` so a caller can gate on KL.

    The prefix runs under `no_grad`, so no activation above `lo` is retained and
    no gradient crosses the boundary. Noise is scaled by the clean state's own
    RMS, which is the only normalisation that works when the scale varies layer
    to layer. The noise itself comes from `noise_like`, on `x`'s own device and
    dtype and keyed by `(seed, step, lo)` where a seed is given.
    """
    chain, _, labels = _batch(pm, b)
    with torch.no_grad():
        if lo:
            x = chain.run(chain.layers[:lo], chain.emb, 0).detach()
        else:
            x = chain.emb.detach()
        scale = x.pow(2).mean().sqrt().clamp_min(1e-6)
        x = x + sigma * scale * noise_like(x, seed, step, lo)
    return chain.logits(chain.run(chain.layers[lo:hi], x, lo)), labels


def forward_block(pm, b: dict, lo: int, hi: int, sigma: float,
                  seed: int | None = None, step: int = 0) -> torch.Tensor:
    """Local denoising loss: corrupt the boundary state, one block predicts the clean one."""
    logits, labels = block_logits(pm, b, lo, hi, sigma, seed, step)
    return _loss(logits, labels)


def forward_range(layers, chain: _Chain, h: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    return chain.run(layers[lo:hi], h, lo)


# -- measurement -----------------------------------------------------------

def _retained_bytes(pm, batch: dict, lo: int, hi: int) -> int:
    """Bytes of activation inside one span forward that still carries a gradient.

    The prefix runs under `no_grad`, so its outputs are detached and a hook can
    tell them apart by `requires_grad` — exactly the tensors a split does not
    keep, rather than a count that penalises the split for them.
    """
    inner = base_model(pm).model
    chain, _, _ = _batch(pm, batch)
    sizes: list[int] = []

    def keep(_m, _i, out):
        if out[0].requires_grad:
            sizes.append(out[0].numel() * out[0].element_size())

    handles = [layer.register_forward_hook(keep) for layer in inner.layers]
    try:
        with torch.no_grad():
            x = chain.run(chain.layers[:lo], chain.emb, 0)
        if not x.requires_grad:  # empty prefix returned the detached embedding itself
            x = x.detach()
        chain.run(chain.layers[lo:hi], x, lo)
    finally:
        for hnd in handles:
            hnd.remove()
    return sum(sizes)


def peak_activation(pm, batch: dict, spans: list[tuple[int, int]] | None = None) -> int:
    """Peak retained activation of a training forward — the memory being compared.

    For the end-to-end arm that is one span over every layer: the whole residual
    stream is retained. For the split it is the largest single block, because
    only one block is ever in flight. `max`, not the sum, is the quantity the
    block count is supposed to divide.
    """
    spans = spans or [(0, len(layers_of(pm)))]
    with torch.enable_grad():
        return max(_retained_bytes(pm, batch, lo, hi) for lo, hi in spans)


def _trainable(pm) -> list[torch.nn.Parameter]:
    return [p for p in pm.parameters() if p.requires_grad]


def _delta_ratio(pm, names: set[str]) -> float:
    """Mean relative movement of the matrices a run touched (final vs start).

    Zero-initialised matrices (a LoRA `lora_B`) have no meaningful *relative*
    movement — ‖Δ‖/0 is 1e12 at the first step — so they are skipped and the
    ratio falls back to absolute movement when nothing else moved.
    """
    rel, absolute = [], []
    for i, layer in enumerate(layers_of(pm)):
        for n, p in layer.named_parameters():
            key = f"{i}.{n}"
            start = _START.get(key)
            if start is None or key not in names or p.dim() < 2:
                continue
            delta = float((p.detach().float() - start).norm())
            absolute.append(delta)
            if float(start.norm()) > 1e-6:
                rel.append(delta / float(start.norm()))
    if rel:
        return sum(rel) / len(rel)
    return sum(absolute) / len(absolute) if absolute else float("nan")


_START: dict[str, torch.Tensor] = {}


def _touched(pm, lo: int, hi: int) -> set[str]:
    """`layers`-relative names of the 2-D parameters inside `[lo, hi)`."""
    return {f"{i}.{n}" for i, layer in enumerate(layers_of(pm)) if lo <= i < hi
            for n, p in layer.named_parameters() if p.dim() >= 2}


def _clone(pm):
    clone = copy.deepcopy(pm)
    clone.config.use_cache = False
    return clone


def _run_variant(pm, batches: list[dict], cfg: BlockConfig,
                 plan: list[tuple[int, int, float]],
                 block: bool) -> tuple[float, int]:
    """Train `pm` in place one `plan` entry per step; return loss and peak bytes."""
    pm.train()
    params = _trainable(pm)
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    losses: list[float] = []
    for step, (lo, hi, sigma) in enumerate(plan):
        b = batches[step % len(batches)]
        # Same seed for both arms and the step index keeps block noise
        # replayable; the e2e arm never reads it.
        loss = (forward_block(pm, b, lo, hi, sigma, cfg.seed, step) if block
                else forward_full(pm, b))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    tail = losses[-max(1, len(losses) // 3):]
    spans = sorted({(lo, hi) for lo, hi, _ in plan})
    return sum(tail) / len(tail), peak_activation(pm, batches[0], spans)


def _items(buffer, cfg: SleepConfig, k: int) -> list[Item]:
    items = buffer.core + buffer.all()[:k]
    return items or [Item(0, 0, "user", "", "user")]


def prepare(buffer, cfg: SleepConfig, cfg_b: BlockConfig) -> list[dict]:
    """The replay material, as `batch_items`-sized encoded batches."""
    items = _items(buffer, cfg, cfg.k_important)
    k = max(1, cfg_b.batch_items)
    return [encode(cfg, render(items[i:i + k])) for i in range(0, len(items), k)]


def run_both(pm, batches: list[dict], cfg: BlockConfig) -> BlockReport:
    """Train the same start model two ways at matched step count and compare."""
    n = len(layers_of(pm))
    spans = split_into(n, cfg.blocks)
    rep = BlockReport(blocks=len(spans), steps=cfg.steps,
                      block_layers=min(hi - lo for lo, hi in spans))
    if rep.block_layers < cfg.min_block_layers:
        # The depth is already known to be illegal, so the two training runs
        # would only be bought to confirm it.
        rep.gate_ok, rep.reason = decide(rep, cfg)
        return rep
    _START.clear()
    _START.update({f"{i}.{n}": p.detach().clone()
                   for i, layer in enumerate(layers_of(pm))
                   for n, p in layer.named_parameters() if p.dim() >= 2})

    full = [(0, n, 0.0)] * cfg.steps
    e2e = _clone(pm)
    rep.e2e_loss, rep.e2e_peak = _run_variant(e2e, batches, cfg, full, block=False)
    rep.e2e_ratio = _delta_ratio(e2e, _touched(pm, 0, n))

    blk = _clone(pm)
    blk_plan = plan(n, cfg, cfg.steps)
    rep.block_loss, rep.block_peak = _run_variant(blk, batches, cfg, blk_plan, block=True)
    rep.block_ratio = _delta_ratio(blk, _touched(pm, 0, n))
    rep.gate_ok, rep.reason = decide(rep, cfg)
    return rep


def decide(rep: BlockReport, cfg: BlockConfig) -> tuple[bool, str]:
    """Split or not. Cheap and honest: no gain is a refusal, no depth is a refusal.

    Depth is checked before the losses so an unrun split (skipped as illegal by
    `run_both`) reports the real reason rather than "non-finite loss".
    """
    if rep.block_layers < cfg.min_block_layers:
        return False, (f"{rep.block_layers} layer/block < {cfg.min_block_layers}: "
                       "blocks that shallow cannot denoise")
    if not all(_finite(x) for x in (rep.e2e_loss, rep.block_loss)):
        return False, "non-finite loss"
    if not (_finite(rep.e2e_peak) and _finite(rep.block_peak)):
        return False, "peak activation unmeasured"
    gain = rep.memory_gain()
    if gain < cfg.min_memory_gain:
        return False, f"peak activation only -{gain:.0%} saved, want >= {cfg.min_memory_gain:.0%}"
    q = rep.quality_loss()
    if q > cfg.max_quality_loss:
        return False, f"loss +{q:.1%} > {cfg.max_quality_loss:.0%}"
    return True, f"{gain:.0%} peak activation saved for {q:+.1%} loss"


def savings(pm, n_layers: int | None = None) -> dict[int, float]:
    """Projected activation saving per split size, counting depth, not divisor.

    `1/blocks` is only true when every block is deep enough. Once `blocks`
    exceeds `depth / min_block_layers` the split stops being legal and the ratio
    stops falling, so a six-block claim on a 12-layer model is really four.
    """
    n = n_layers if n_layers is not None else len(layers_of(pm))
    out = {}
    for b in range(1, n + 1):
        shallow = min(hi - lo for lo, hi in split_into(n, b))
        out[b] = 1.0 / b if shallow >= _MIN_DEPTH else _MIN_DEPTH / n
    return out


_MIN_DEPTH = BlockConfig.min_block_layers


# -- replay into a fresh slot ---------------------------------------------

def distill_inner(pm, buffer, cfg: SleepConfig, cfg_b: BlockConfig, steps: int) -> float:
    """Train the open slot one block at a time against a frozen teacher base.

    Used where a whole-model forward no longer fits (the merge residual). The
    teacher is the prospective base, read without adapters, so each block learns
    to close the merge's approximation error within its own band rather than to
    re-add a delta the base already holds.
    """
    teacher = copy.deepcopy(pm)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if not hasattr(pm, "disable_adapter"):
        raise UnsupportedModel("PeftModel.disable_adapter required to read the teacher base")

    layers = layers_of(pm)
    batches = prepare(buffer, cfg, cfg_b) or [encode(cfg, "")]
    params = _trainable(pm)
    opt = torch.optim.AdamW(params, lr=cfg_b.lr)
    losses: list[float] = []
    pm.train()
    for step, (lo, hi, sigma) in enumerate(plan(len(layers), cfg_b, steps)):
        b = batches[step % len(batches)]
        with torch.no_grad(), teacher.disable_adapter():
            tchain, _, _ = _batch(teacher, b)
            th = tchain.run(tchain.layers, tchain.emb, 0)
        chain, _, _ = _batch(pm, b)
        with torch.no_grad():
            x = chain.run(chain.layers[:lo], chain.emb, 0).detach()
            x = x + sigma * x.pow(2).mean().sqrt().clamp_min(1e-6) * noise_like(
                x, cfg_b.seed, step, lo)
        h = forward_range(chain.layers, chain, x, lo, hi)
        loss = F.mse_loss(h.float(), th[:, : h.shape[1]].float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    pm.eval()
    return sum(losses[-max(1, len(losses) // 3):]) / max(1, len(losses) // 3)


__all__ = ["BlockConfig", "BlockReport", "UnsupportedModel", "block_logits", "decide",
           "distill_inner", "forward_block", "forward_full", "layers_of", "noise_like",
           "peak_activation", "plan", "prepare", "run_both", "savings", "sigma_span",
           "split_into"]
