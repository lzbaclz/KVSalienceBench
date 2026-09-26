"""Auditable, single-request physical KV eviction (not a serving engine).

The masked reference and compact backend share *irreversible* eligibility,
absolute positions and block policies. The archived SEER simulator is NOT
assumed equivalent. Pinned HF support is deliberately narrow; unsupported
models fail rather than silently changing cache semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

SUPPORTED_TRANSFORMERS = "4.51.3"


@dataclass(frozen=True)
class EvictionConfig:
    policy: Literal["full", "h2o_block", "xqp_reconstructed"] = "h2o_block"
    backend: Literal["physical", "masked"] = "physical"
    budget_fraction: float = 0.2
    block_size: int = 32
    sink_blocks: int = 1
    recent_blocks: int = 1
    ema_decay: float = 0.9
    cross_fraction: float = 0.1
    prefill_chunk: int = 256
    observation_window: int = 64

    def __post_init__(self) -> None:
        if self.policy not in {"full", "h2o_block", "xqp_reconstructed"}:
            raise ValueError(f"unsupported policy {self.policy!r}")
        if self.backend not in {"physical", "masked"}:
            raise ValueError(f"unsupported backend {self.backend!r}")
        if not 0 < self.budget_fraction <= 1:
            raise ValueError("budget_fraction must be in (0, 1]")
        if not 0 <= self.ema_decay < 1 or not 0 < self.cross_fraction <= 1:
            raise ValueError("invalid EMA/cross fraction")
        if min(self.block_size, self.prefill_chunk, self.observation_window) < 1:
            raise ValueError("block/chunk/window sizes must be positive")
        if self.observation_window > self.prefill_chunk:
            raise ValueError("observation_window must not exceed prefill_chunk")
        if self.sink_blocks < 0 or self.recent_blocks < 1:
            raise ValueError("sink_blocks >= 0 and recent_blocks >= 1 are required")
        if self.policy == "full" and self.budget_fraction != 1:
            raise ValueError("full policy requires budget_fraction=1")

    def budget(self, prompt_tokens: int) -> int:
        import math
        if prompt_tokens < 1:
            raise ValueError("empty prompt")
        blocks = (prompt_tokens + self.block_size - 1) // self.block_size
        if self.policy == "full":
            return blocks
        budget = math.floor(self.budget_fraction * blocks)
        # Do not silently inflate a too-small budget to fit mandatory blocks.
        if budget < self.sink_blocks + self.recent_blocks:
            raise ValueError(
                f"budget {budget} blocks cannot accommodate sink+recent "
                f"({self.sink_blocks}+{self.recent_blocks}); increase context or budget"
            )
        return budget


def load_two_view_weights(path: str | Path) -> tuple[float, float, float]:
    """Accept only the frozen query/recency-neutral checkpoint, without refitting."""
    import math
    obj = json.loads(Path(path).read_text())
    w = obj.get("weights")
    b = obj.get("bias")
    if obj.get("per_layer", False) or not isinstance(w, list) or len(w) != 4:
        raise ValueError("expected one shared four-slot checkpoint")
    if w[2:] != [0.0, 0.0]:
        raise ValueError("query/recency weights must be zero; no feature substitution allowed")
    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in [*w, b]):
        raise ValueError("non-finite or non-numeric checkpoint")
    return float(w[0]), float(w[1]), float(b)


@dataclass
class LayerState:
    positions: Tensor                 # absolute token positions of physical storage
    eligible: Tensor                  # irreversible alive bit (all true in compact backend)
    block_ids: Tensor                 # sorted live logical blocks, not compact offsets
    ema: Tensor
    accumulated: Tensor

    @classmethod
    def empty(cls, device: torch.device) -> "LayerState":
        return cls(torch.empty(0, device=device, dtype=torch.long),
                   torch.empty(0, device=device, dtype=torch.bool),
                   torch.empty(0, device=device, dtype=torch.long),
                   torch.empty(0, device=device, dtype=torch.float32),
                   torch.empty(0, device=device, dtype=torch.float32))


def aligned_values(old_ids: Tensor, old_values: Tensor, new_ids: Tensor) -> Tensor:
    """Look up by logical ID; never confuse compact offsets across layers."""
    if old_ids.numel() == 0:
        return torch.zeros(new_ids.shape, device=new_ids.device, dtype=torch.float32)
    idx = torch.searchsorted(old_ids, new_ids)
    safe = idx.clamp(max=old_ids.numel() - 1)
    return torch.where((idx < old_ids.numel()) & (old_ids[safe] == new_ids),
                       old_values[safe], torch.zeros_like(new_ids, dtype=torch.float32))


def compact_kv(key: Tensor, value: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Allocate independent storage; returning a smaller view would not evict KV."""
    if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
        raise ValueError("expected matching [1, KV-heads, tokens, head_dim] K and V")
    return key.index_select(-2, indices).contiguous(), value.index_select(-2, indices).contiguous()


class PhysicalKVSession:
    """Own hooks/cache for exactly one request; use as a context manager.

    Prefill is chunked and initially uncompressed. Features are initialized
    from the last observation_window prompt queries. Every decode step uses
    retained KV plus its new token, then evicts; transient storage is therefore
    larger than the post-step budget. No CPU offload, resurrection, paging,
    continuous batching, original Ada-KV or optimized H2O is implemented.
    """

    def __init__(self, model, config: EvictionConfig, prompt_tokens: int,
                 weights: tuple[float, float, float] | None = None):
        import transformers
        from transformers.cache_utils import DynamicCache
        if transformers.__version__ != SUPPORTED_TRANSFORMERS:
            raise RuntimeError(f"requires transformers=={SUPPORTED_TRANSFORMERS}; "
                               f"found {transformers.__version__}")
        cfg = model.config
        if cfg.model_type not in {"llama", "qwen2"}:
            raise ValueError("validated adapter supports only Llama and Qwen2 causal LMs")
        if cfg._attn_implementation != "eager":
            raise ValueError("eager attention is required (features must be observed)")
        if getattr(cfg, "use_sliding_window", False):
            raise ValueError("sliding-window models are not supported")
        if getattr(model, "is_quantized", False) or getattr(model, "hf_device_map", None):
            raise ValueError("quantization and device_map/sharding are not supported")
        devices = {p.device for p in model.parameters()}
        if len(devices) != 1 or next(iter(devices)).type not in {"cpu", "cuda"}:
            raise ValueError("one CPU or CUDA device is required")
        if model.training:
            raise ValueError("model.eval() is required")
        if config.policy == "xqp_reconstructed" and weights is None:
            raise ValueError("xqp_reconstructed requires explicit frozen weights")
        self.model = model
        self.cfg = config
        self.device = next(iter(devices))
        self.dtype = next(model.parameters()).dtype
        self.prompt_tokens = prompt_tokens
        self.budget_blocks = config.budget(prompt_tokens)
        self.weights = weights
        self.cache = DynamicCache()
        self.states = [LayerState.empty(self.device) for _ in model.model.layers]
        self.handles = []
        self.logical_length = 0
        self._current_positions = torch.empty(0, device=self.device, dtype=torch.long)
        self._collect = False
        self._entered = False
        self._prefilled = False
        self._finite = torch.ones((), device=self.device, dtype=torch.bool)

    def __enter__(self) -> "PhysicalKVSession":
        if self._entered or getattr(self.model, "_physical_kv_owner", None) is not None:
            raise RuntimeError("sessions may not share a model concurrently or be re-entered")
        self._entered = True
        self.model._physical_kv_owner = self
        for i, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(
                self._pre_hook(i), with_kwargs=True))
            self.handles.append(layer.self_attn.register_forward_hook(self._post_hook(i)))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if getattr(self.model, "_physical_kv_owner", None) is self:
            del self.model._physical_kv_owner
        self._entered = False
        # A caller may retain scalar results, not an abandoned full GPU cache.
        self.cache = None
        self.states.clear()

    def _pre_hook(self, layer: int):
        def hook(module, args, kwargs):
            state = self.states[layer]
            qpos = self._current_positions
            state.positions = torch.cat((state.positions, qpos))
            state.eligible = torch.cat((state.eligible, torch.ones_like(qpos, dtype=torch.bool)))
            allow = (state.positions[None, :] <= qpos[:, None]) & state.eligible[None, :]
            # Every query has at least itself available. Use true -inf to avoid
            # a finite sentinel accidentally giving removed keys nonzero mass.
            mask = torch.zeros(allow.shape, device=self.device, dtype=self.dtype)
            mask.masked_fill_(~allow, -torch.inf)
            kwargs["attention_mask"] = mask[None, None, :, :]
            return args, kwargs
        return hook

    def _post_hook(self, layer: int):
        def hook(module, args, output):
            if not self._collect:
                return
            attn = output[1]
            if attn is None:
                raise RuntimeError("attention probabilities were not produced")
            state = self.states[layer]
            # Head-mean, query-mean, then token-mean per logical block.
            per_token = attn[0].float().mean(dim=(0, 1))
            pos = state.positions[state.eligible]
            vals = per_token[state.eligible]
            ids, inv, counts = torch.unique(pos // self.cfg.block_size,
                                            sorted=True, return_inverse=True, return_counts=True)
            cur = torch.zeros(ids.shape, device=self.device, dtype=torch.float32)
            cur.scatter_add_(0, inv, vals)
            cur /= counts
            old_ema = aligned_values(state.block_ids, state.ema, ids)
            old_sum = aligned_values(state.block_ids, state.accumulated, ids)
            existed = torch.isin(ids, state.block_ids)
            state.ema = torch.where(existed, self.cfg.ema_decay * old_ema +
                                    (1 - self.cfg.ema_decay) * cur, cur)
            state.accumulated = old_sum + cur
            state.block_ids = ids
            self._finite &= torch.isfinite(cur).all()
        return hook

    @torch.inference_mode()
    def _forward(self, ids: Tensor, collect: bool) -> Tensor:
        if not self._entered or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("active session and nonempty batch-one token IDs required")
        if ids.device != self.device:
            raise ValueError("input IDs and model must be on the same device")
        n = ids.shape[1]
        self._current_positions = torch.arange(self.logical_length, self.logical_length + n,
                                                device=self.device, dtype=torch.long)
        self._collect = collect
        # HF passes this 4-D mask through unchanged. Layer hooks install the
        # correct layer-specific mask before attention. Positions NEVER come
        # from get_seq_length(), which shrinks when physical KV is compacted.
        placeholder = torch.zeros((1, 1, n, 1), device=self.device, dtype=self.dtype)
        out = self.model(input_ids=ids, past_key_values=self.cache,
                         position_ids=self._current_positions[None, :],
                         cache_position=self._current_positions,
                         attention_mask=placeholder, use_cache=True,
                         output_attentions=False, output_hidden_states=False,
                         logits_to_keep=1, return_dict=True)
        self.cache = out.past_key_values
        self.logical_length += n
        logits = out.logits[:, -1, :].detach()
        self._finite &= torch.isfinite(logits).all()
        return logits

    @torch.inference_mode()
    def prefill(self, ids: Tensor) -> Tensor:
        if self._prefilled or self.logical_length != 0 or ids.shape != (1, self.prompt_tokens):
            raise ValueError("prefill must be called exactly once with the declared prompt")
        tail = min(self.cfg.observation_window, self.prompt_tokens)
        prefix = self.prompt_tokens - tail
        for start in range(0, prefix, self.cfg.prefill_chunk):
            self._forward(ids[:, start:min(prefix, start + self.cfg.prefill_chunk)], False)
        logits = self._forward(ids[:, prefix:], True)
        self._prefilled = True
        return logits

    @torch.inference_mode()
    def decode(self, token_id: Tensor) -> Tensor:
        if not self._prefilled or token_id.shape != (1, 1):
            raise ValueError("decode expects one token after prefill")
        return self._forward(token_id, True)

    def _select(self, layer: int) -> Tensor:
        import math
        state = self.states[layer]
        ids = state.block_ids
        if self.cfg.policy == "full" or ids.numel() <= self.budget_blocks:
            return ids
        mandatory = (ids < self.cfg.sink_blocks) | (ids >= ids[-1] - self.cfg.recent_blocks + 1)
        score = state.accumulated
        if self.cfg.policy == "xqp_reconstructed":
            within = state.ema / state.ema.max().clamp_min(1e-12)
            cross = torch.zeros_like(within)
            if layer > 0:
                prev = self.states[layer - 1]
                count = max(1, math.ceil(prev.block_ids.numel() * self.cfg.cross_fraction))
                order = torch.argsort(prev.ema, descending=True, stable=True)
                hot = prev.block_ids[order[:count]]
                cross = torch.isin(ids, hot).float()
            a, b, bias = self.weights
            # The monotone logit avoids sigmoid saturation changing top-k ties.
            score = a * within + b * cross + bias
        mandatory_idx = torch.nonzero(mandatory, as_tuple=False).flatten()
        room = self.budget_blocks - mandatory_idx.numel()
        if room < 0:
            raise RuntimeError("mandatory blocks exceed frozen budget")
        rest_idx = torch.nonzero(~mandatory, as_tuple=False).flatten()
        order = torch.argsort(score[rest_idx], descending=True, stable=True)
        return ids[torch.cat((mandatory_idx, rest_idx[order[:room]]))].sort().values

    @torch.inference_mode()
    def evict(self, forced_blocks: list[list[int]] | None = None) -> None:
        """Evict after prefill/each decode; forced schedules are correctness-only."""
        if not self._prefilled:
            raise RuntimeError("cannot evict before prefill")
        if forced_blocks is not None and len(forced_blocks) != len(self.states):
            raise ValueError("forced schedule must cover every layer")
        # Snapshot all choices before modifying any layer's cross-layer state.
        choices = [self._select(i) for i in range(len(self.states))] if forced_blocks is None else [
            torch.tensor(x, device=self.device, dtype=torch.long) for x in forced_blocks]
        for i, (state, keep) in enumerate(zip(self.states, choices)):
            if forced_blocks is not None:
                if keep.numel() == 0 or not torch.equal(keep, keep.unique(sorted=True)):
                    raise ValueError("forced blocks must be nonempty, sorted and unique")
                if not bool(torch.isin(keep, state.block_ids).all()):
                    raise ValueError("forced schedule attempts resurrection or unknown block")
                if self.cfg.policy != "full" and keep.numel() > self.budget_blocks:
                    raise ValueError("forced schedule exceeds budget")
                latest = (self.logical_length - 1) // self.cfg.block_size
                mandatory = state.block_ids[(state.block_ids < self.cfg.sink_blocks) |
                                            (state.block_ids >= latest - self.cfg.recent_blocks + 1)]
                if not bool(torch.isin(mandatory, keep).all()):
                    raise ValueError("forced schedule drops mandatory sink/recent blocks")
            retain = state.eligible & torch.isin(state.positions // self.cfg.block_size, keep)
            if self.cfg.backend == "physical":
                indices = torch.nonzero(retain, as_tuple=False).flatten()
                # Avoid an unnecessary full-cache copy in the reference policy.
                if indices.numel() != state.positions.numel():
                    key, value = compact_kv(self.cache.key_cache[i], self.cache.value_cache[i], indices)
                    self.cache.key_cache[i] = key
                    self.cache.value_cache[i] = value
                    state.positions = state.positions.index_select(0, indices)
                state.eligible = torch.ones_like(state.positions, dtype=torch.bool)
            else:
                state.eligible = retain
            active = torch.isin(state.block_ids, keep)
            state.block_ids = state.block_ids[active]
            state.ema = state.ema[active]
            state.accumulated = state.accumulated[active]

    def assert_valid(self) -> None:
        """Synchronous diagnostic outside primary latency intervals."""
        if not bool(self._finite):
            raise FloatingPointError("non-finite attention/logits; run is invalid")
        for i, state in enumerate(self.states):
            key, value = self.cache.key_cache[i], self.cache.value_cache[i]
            if key.shape != value.shape or key.shape[-2] != state.positions.numel():
                raise AssertionError("cache/position shape mismatch")
            if not bool((state.positions[1:] > state.positions[:-1]).all()):
                raise AssertionError("positions are not strictly increasing")
            actual = (state.positions[state.eligible] // self.cfg.block_size).unique(sorted=True)
            if not torch.equal(actual, state.block_ids):
                raise AssertionError("metadata not aligned with live KV")
            if self.cfg.policy != "full" and actual.numel() > self.budget_blocks:
                raise AssertionError("post-step block budget exceeded")
            if self.cfg.backend == "physical" and not bool(state.eligible.all()):
                raise AssertionError("compact backend retained dead KV")

    def schedule(self) -> list[list[int]]:
        return [s.block_ids.detach().cpu().tolist() for s in self.states]

    def memory(self) -> dict[str, int]:
        kv = [x for pair in zip(self.cache.key_cache, self.cache.value_cache) for x in pair]
        meta = [x for s in self.states for x in (s.positions, s.eligible, s.block_ids, s.ema, s.accumulated)]
        def unique_bytes(tensors):
            stores = {(t.device, t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                      for t in tensors if t.numel()}
            return sum(stores.values())
        live = sum(int(s.eligible.sum()) * self.cache.key_cache[i].shape[1] *
                   self.cache.key_cache[i].shape[-1] * self.cache.key_cache[i].element_size() * 2
                   for i, s in enumerate(self.states))
        return {"logical_live_kv_bytes": live,
                "kv_tensor_bytes": sum(t.numel() * t.element_size() for t in kv),
                "kv_storage_bytes": unique_bytes(kv),
                "metadata_storage_bytes": unique_bytes(meta),
                "max_live_blocks_per_layer": max(s.block_ids.numel() for s in self.states)}

    def selection_digest(self) -> str:
        payload = json.dumps(self.schedule(), separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()
