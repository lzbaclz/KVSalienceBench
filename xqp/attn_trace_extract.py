"""Extract real attention-dynamics traces on a GPU (the ICDM data source).

This turns the A100 box into a data generator for the ICDM paper. For each
prompt it runs an incremental decode loop on a HuggingFace causal LM, captures
per-layer attention over the KV cache at every step, aggregates keys into
blocks, and writes the JSONL schema that `xqp.trace.load_trace`, `xqp-train`,
and the ICDM drivers consume.

The pure-NumPy aggregation core (`blockify`, `update_ema`, `labels_for_horizons`)
is unit-tested on CPU. The full driver is **also CPU-validatable**: pass a
preloaded tiny model (built with `attn_implementation="eager"`) plus `input_ids`
and `device="cpu"` to exercise the entire tensor path offline (see tests), so
the A100 run is de-risked. Production: pass `model_id` + `prompts`, `device="cuda"`.

Assumes a Llama/Qwen/Mistral-family module layout
(`model.model.layers[i].self_attn.q_proj`).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


# ----------------------------- testable core -------------------------------

def blockify(x: np.ndarray, block_size: int) -> np.ndarray:
    """Segment-mean over the key axis.

    x: (kv,) per-key attention weights -> (n_blocks,);
       (kv, d) per-key vectors          -> (n_blocks, d).
    The last (ragged) block is averaged over its actual members.
    """
    x = np.asarray(x, dtype=np.float32)
    kv = x.shape[0]
    if kv == 0:
        return x.reshape((0,) + x.shape[1:])
    n_blocks = int(np.ceil(kv / block_size))
    return np.stack([x[b * block_size:(b + 1) * block_size].mean(axis=0)
                     for b in range(n_blocks)], axis=0).astype(np.float32)


def blockify_max(x: np.ndarray, block_size: int) -> np.ndarray:
    """Like ``blockify`` but segment-MAX over the key axis.
    This is a query-key probe, not the Quest page-bound algorithm."""
    x = np.asarray(x, dtype=np.float32)
    kv = x.shape[0]
    if kv == 0:
        return x.reshape((0,) + x.shape[1:])
    n_blocks = int(np.ceil(kv / block_size))
    return np.stack([x[b * block_size:(b + 1) * block_size].max(axis=0)
                     for b in range(n_blocks)], axis=0).astype(np.float32)


def update_ema(prev, cur, decay: float = 0.9) -> np.ndarray:
    """EMA tolerant of a growing block count (new blocks seed at their value)."""
    cur = np.asarray(cur, dtype=np.float32)
    if prev is None:
        return cur.copy()
    if prev.shape[0] == cur.shape[0]:
        return (decay * prev + (1 - decay) * cur).astype(np.float32)
    out = cur.copy()
    m = min(prev.shape[0], cur.shape[0])
    out[:m] = decay * prev[:m] + (1 - decay) * cur[:m]
    return out.astype(np.float32)


def labels_for_horizons(block_attn_seq: list, t: int, n_blocks_t: int,
                        r_label: float, horizons=(1, 4, 16, 64)) -> dict:
    """Top-r labels for the blocks present at step t, read off future steps.

    block_attn_seq[s] is the (n_blocks_s,) per-block attention at step s; the
    blocks present at t are the first n_blocks_t entries of any later step.
    """
    from .features import topk_indicator
    out = {}
    T = len(block_attn_seq)
    for h in horizons:
        s = t + h
        if h < 1 or s >= T:
            raise ValueError("requested future horizon is unavailable; collect additional steps")
        fut = np.asarray(block_attn_seq[s], dtype=np.float32)[:n_blocks_t]
        out[h] = topk_indicator(fut, r_label).astype(np.int64)
    return out


# ----------------------------- GPU/driver ----------------------------------

def _gpu_available() -> bool:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _layer_key(past, l):
    """Key tensor (batch, n_kv_heads, kv, head_dim) for layer l, across the
    transformers cache API versions (5.x DynamicCache.layers[l].keys; 4.x
    .key_cache[l]; legacy tuple-of-(k,v))."""
    layers = getattr(past, "layers", None)
    if layers is not None:
        return layers[l].keys
    kc = getattr(past, "key_cache", None)
    if kc is not None:
        return kc[l]
    return past[l][0]


def extract_attention_traces(model_id=None, prompts=None, out_path="traces.jsonl", *,
                             model=None, tokenizer=None, input_ids=None,
                             device="cuda", block_size: int = 32, r_label: float = 0.10,
                             horizons=(1, 4, 16, 64), max_new_tokens: int = 64,
                             dtype: str = "float16", ema_decay: float = 0.9,
                             request_id_start: int = 0,
                             query_variants: bool = False) -> int:
    """Collect version-2 traces with phase-aligned queries and genuine horizons.

    max_new_tokens is the number of FEATURE steps; an additional max(horizons)
    greedy steps supply future labels (EOS is intentionally not a stopping rule).
    Version-1 traces are not overwritten, regenerated or numerically reinterpreted.
    The corrected tensor adapter is validated only for pinned Llama/Qwen2 models.
    """
    import contextlib
    import hashlib
    import importlib
    import os
    import torch
    import transformers
    from .features import extract_features
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    if tuple(horizons) != (1, 4, 16, 64):
        raise ValueError("version-2 schema requires horizons (1,4,16,64)")
    if block_size < 1 or max_new_tokens < 1 or not 0 < r_label <= 1 or not 0 <= ema_decay < 1:
        raise ValueError("invalid extraction parameters")
    path = Path(out_path)
    meta_path = Path(str(path) + ".meta.json")
    if path.exists() or meta_path.exists():
        raise FileExistsError("refusing to overwrite traces; choose a fresh version-2 path")
    if transformers.__version__ != "4.51.3":
        raise RuntimeError("version-2 collector requires transformers==4.51.3")
    if model is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), local_files_only=True,
            trust_remote_code=False, attn_implementation="eager")
    model = model.to(device).eval()
    if model.config.model_type not in {"llama", "qwen2"} or model.config._attn_implementation != "eager":
        raise ValueError("version-2 collector supports eager Llama/Qwen2 only")
    if getattr(model.config, "use_sliding_window", False):
        raise ValueError("sliding-window attention requires a separately validated collector")
    if input_ids is not None:
        id_list = [t.to(device) for t in input_ids]
    else:
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True,
                                                       trust_remote_code=False)
        if prompts is None:
            raise ValueError("provide prompts or input_ids")
        id_list = [tokenizer(p, return_tensors="pt").input_ids.to(device) for p in prompts]
    if not id_list:
        raise ValueError("no input requests")
    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // n_heads)
    n_layers = len(model.model.layers)
    total_steps = max_new_tokens + max(horizons)
    metadata = {"trace_version": 2, "query_space": "post-normalization, post-RoPE",
                "query_probe": "per-token/head dot-max, NOT Quest page bounds",
                "label_boundary": "additional future steps; no terminal clamping",
                "stopping_rule": "fixed greedy steps, including horizon lookahead; EOS ignored",
                "feature_steps": max_new_tokens, "lookahead_steps": max(horizons),
                "block_size": block_size, "r_label": r_label, "ema_decay": ema_decay,
                "transformers": transformers.__version__, "torch": torch.__version__,
                "model_config": model.config.to_dict(),
                "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "requests": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + f".tmp.{os.getpid()}")
    n_rows = 0
    try:
        with temp.open("x") as fh:
            for pi, ids in enumerate(id_list):
                if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
                    raise ValueError("nonempty batch-one prompts required")
                if ids.shape[1] + total_steps > model.config.max_position_embeddings:
                    raise ValueError("prompt plus feature steps and horizon lookahead exceeds model context")
                q_capture, rope = {}, {}
                with contextlib.ExitStack() as hooks:
                    for li, layer in enumerate(model.model.layers):
                        attn_module = layer.self_attn
                        apply_rope = importlib.import_module(type(attn_module).__module__).apply_rotary_pos_emb
                        def pre_hook(module, args, kwargs, idx=li):
                            if "position_embeddings" not in kwargs:
                                raise RuntimeError("RoPE tensors unavailable; cannot align queries")
                            rope[idx] = kwargs["position_embeddings"]
                        def q_hook(module, args, out, idx=li, attn=attn_module, rotate=apply_rope):
                            # q_proj output is pre-RoPE, whereas cached K is post-RoPE.
                            # Reconstruct the actual query coordinate system explicitly.
                            q = out.view(out.shape[0], out.shape[1], n_heads, head_dim).transpose(1, 2)
                            if hasattr(attn, "q_norm"):
                                q = attn.q_norm(q)
                            cos, sin = rope[idx]
                            q_rot, _ = rotate(q, q, cos, sin)
                            q_capture[idx] = q_rot[0, :, -1, :].detach().float().cpu().numpy()
                        handle = attn_module.register_forward_pre_hook(pre_hook, with_kwargs=True)
                        hooks.callback(handle.remove)
                        handle = attn_module.q_proj.register_forward_hook(q_hook)
                        hooks.callback(handle.remove)
                    # Eager attention can still materialize scores internally, even
                    # output_attentions=False. Bound the prefill query dimension.
                    past = None
                    with torch.inference_mode():
                        for start in range(0, ids.shape[1], 256):
                            out = model(ids[:, start:start+256], past_key_values=past,
                                        use_cache=True, output_attentions=False, logits_to_keep=1)
                            past = out.past_key_values
                        # Consume the actual first generated token, NOT the final
                        # prompt token a second time (legacy collector bug).
                        next_id = out.logits[:, -1:].argmax(-1)
                    forwarded = []
                    ema = [None] * n_layers
                    history = [[] for _ in range(n_layers)]
                    feat_rows, last_used_cache = [], {}
                    for t in range(total_steps):
                        forwarded.append(int(next_id.item()))
                        with torch.inference_mode():
                            out = model(next_id, past_key_values=past, use_cache=True,
                                        output_attentions=True, logits_to_keep=1)
                        past = out.past_key_values
                        for l in range(n_layers):
                            attn_l = out.attentions[l][0].float().mean(0).squeeze(0).cpu().numpy()
                            if not np.isfinite(attn_l).all():
                                raise FloatingPointError(f"non-finite attention: request {pi}, layer {l}, step {t}")
                            block_attn = blockify(attn_l, block_size)
                            history[l].append(block_attn)
                            ema[l] = update_ema(ema[l], block_attn, ema_decay)
                            if t >= max_new_tokens:
                                continue
                            key = _layer_key(past, l)[0].float().cpu().numpy()
                            K_block = blockify(key.mean(0), block_size)
                            q_heads = q_capture[l]
                            nb = len(block_attn)
                            last_used = last_used_cache.get(l, np.zeros(0, dtype=np.float32))
                            if len(last_used) < nb:
                                last_used = np.concatenate([last_used, np.full(nb-len(last_used), t, np.float32)])
                            last_used_cache[l] = last_used
                            F = extract_features(ema_within=ema[l],
                                ema_prev_layer=ema[l-1] if l else None,
                                K_layer=K_block, q_prev=q_heads.mean(0), step=t,
                                last_used=last_used, r_cross=r_label)
                            qv = None
                            if query_variants:
                                if n_heads % key.shape[0]:
                                    raise ValueError("invalid grouped-query head mapping")
                                kv_of_head = np.arange(n_heads) // (n_heads // key.shape[0])
                                dots = np.einsum("hkd,hd->hk", key[kv_of_head], q_heads) / np.sqrt(head_dim)
                                qv = blockify_max(dots.max(0), block_size), blockify(dots.mean(0), block_size)
                            if not np.isfinite(F).all():
                                raise FloatingPointError("non-finite features")
                            feat_rows.append((l, t, F, nb, qv))
                        next_id = out.logits[:, -1:].argmax(-1)
                        if not bool(torch.isfinite(out.logits).all()):
                            raise FloatingPointError("non-finite logits")
                    for l, t, F, nb, qv in feat_rows:
                        labels = labels_for_horizons(history[l], t, nb, r_label, horizons)
                        for b in range(nb):
                            row = dict(trace_version=2, request_id=f"p{request_id_start+pi}",
                                layer=l, step=t, block_idx=b, f_within=float(F[b, 0]),
                                f_cross=float(F[b, 1]), f_query=float(F[b, 2]), f_pos=float(F[b, 3]))
                            for h in horizons:
                                row[f"y_h{h}"] = int(labels[h][b])
                            if qv is not None:
                                row["f_query_dotmax"] = float(qv[0][b])
                                row["f_query_dotmean"] = float(qv[1][b])
                            fh.write(json.dumps(row, allow_nan=False) + "\n")
                            n_rows += 1
                    metadata["requests"].append({"id": f"p{request_id_start+pi}",
                        "input_tokens": ids.shape[1],
                        "input_ids_sha256": hashlib.sha256(ids.cpu().numpy().astype("<i8").tobytes()).hexdigest(),
                        "forwarded_generated_token_ids": forwarded})
                del past, out, history, feat_rows
        metadata["rows"] = n_rows
        h = hashlib.sha256()
        with temp.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                h.update(chunk)
        metadata["trace_sha256"] = h.hexdigest()
        # Publish by hard-link (atomic no-clobber). Metadata is a separate
        # completion prerequisite checked by the new query-control driver.
        os.link(temp, path)
        with meta_path.open("x") as f:
            json.dump(metadata, f, indent=2, allow_nan=False)
            f.write("\n")
    finally:
        temp.unlink(missing_ok=True)
    return n_rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="xqp-attn-trace")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=None, help="text file, one prompt per line")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.prompts:
        prompts = [ln for ln in Path(a.prompts).read_text().splitlines() if ln.strip()]
    else:
        prompts = ["Summarize the history of long-context language models."]
    n = extract_attention_traces(a.model, prompts, a.out, device=a.device,
                                 block_size=a.block_size, max_new_tokens=a.max_new_tokens)
    print(json.dumps({"out": a.out, "rows": n}))
