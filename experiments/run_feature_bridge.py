#!/usr/bin/env python3
"""Offline-to-runtime feature bridge (PC item P1-1): separate feature realization
from trajectory with ONE scorer, ONE label definition and the SAME prompts.

Three measurements, all with the frozen two-view checkpoint (the object the
manuscript evaluates in the loop), all with the label "block is in the top
ceil(r*n) of the block-mean attention h steps later, among blocks present now":

  A. offline-style features on the FULL-cache trajectory. The block attention
     stream recorded by the reference backend is re-aggregated with the
     version-2 collector's conventions: EMA seeded at the first decode step
     (no prefill observation window), within = EMA / (max + 1e-9), cross =
     top-ceil(r*n) indicator of the previous layer's EMA, layer 0 falls back to
     its own EMA.
  B. runtime-reconstructed features on the SAME full-cache trajectory, i.e.
     exactly what ``PhysicalKVSession._select`` would score at that step:
     EMA seeded from the last observation-window prompt queries, within =
     EMA / max(EMA), cross = membership in the previous layer's top
     ceil(cross_fraction*n) blocks by EMA, layer 0 cross = 0.
  C. the same runtime features along the ON-POLICY trajectory (xqp_reconstructed
     at the given budget, irreversible eviction), where the candidate set at a
     step is the retained block set and the future label is read from the
     attention the retained blocks actually receive.

A-B isolates feature realization (same rows, same labels, same tokens);
B-C isolates the trajectory (same realization, policy-induced states). This
is a controlled diagnostic of the offline-to-loop gap, not a serving result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def topk_indicator(x: np.ndarray, r: float) -> np.ndarray:
    n = x.shape[0]
    k = max(1, int(math.ceil(r * n)))
    idx = np.argsort(-x, kind="stable")[:k]
    out = np.zeros(n, np.float32)
    out[idx] = 1.0
    return out


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata
    n_pos = float(y.sum()); n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class Recorder:
    """Collect (block ids, block attention, EMA after update) per layer per step."""

    def __init__(self, n_layers):
        self.steps = [[] for _ in range(n_layers)]

    def snapshot(self, layer, ids, cur, ema):
        self.steps[layer].append((ids.detach().cpu().numpy().astype(np.int64),
                                  cur.detach().float().cpu().numpy(), ema.detach().float().cpu().numpy()))


def run_trajectory(model, ids, cfg, weights, steps, recorder_cls, eos_ids):
    """Run prefill + `steps` decode steps (EOS ignored) and record block statistics."""
    import torch
    from xqp import physical_kv as pk
    rec = recorder_cls(len(model.model.layers))
    session = pk.PhysicalKVSession(model, cfg, ids.shape[1], weights)
    original_post = session._post_hook

    def patched_post(layer):
        inner = original_post(layer)

        def hook(module, args, output):
            inner(module, args, output)
            if session._collect:
                st = session.states[layer]
                rec.snapshot(layer, st.block_ids, session._last_cur[layer], st.ema)
        return hook

    # Expose `cur` (block attention of this step) by wrapping the state update.
    session._last_cur = {}
    orig_post_hook = pk.PhysicalKVSession._post_hook

    def post_hook_capture(self, layer):
        base = orig_post_hook(self, layer)

        def hook(module, args, output):
            if not self._collect:
                return
            attn = output[1]
            state = self.states[layer]
            per_token = attn[0].float().mean(dim=(0, 1))
            pos = state.positions[state.eligible]
            vals = per_token[state.eligible]
            ids_, inv, counts = torch.unique(pos // self.cfg.block_size, sorted=True,
                                             return_inverse=True, return_counts=True)
            cur = torch.zeros(ids_.shape, device=self.device, dtype=torch.float32)
            cur.scatter_add_(0, inv, vals)
            cur /= counts
            self._last_cur[layer] = cur
            base(module, args, output)
        return hook

    session._post_hook = lambda layer: patched_post_factory(session, layer, post_hook_capture, rec)
    generated = []
    with session:
        logits = session.prefill(ids)
        session.evict()
        tok = int(logits.argmax(-1).item())
        generated.append(tok)
        for _ in range(steps):
            logits = session.decode(torch.tensor([[tok]], device=ids.device))
            session.evict()
            tok = int(logits.argmax(-1).item())
            generated.append(tok)
        session.assert_valid()
    return rec, generated


def patched_post_factory(session, layer, post_hook_capture, rec):
    capture = post_hook_capture(session, layer)

    def hook(module, args, output):
        capture(module, args, output)
        if session._collect:
            st = session.states[layer]
            rec.snapshot(layer, st.block_ids, session._last_cur[layer], st.ema)
    return hook


def features_and_labels(rec: Recorder, cfg, weights, horizon, r_label, ema_decay):
    """Return dict of per-realization (scores, labels) arrays for one request."""
    a, b, bias = weights
    n_layers = len(rec.steps)
    # Decode-step snapshots only (prefill snapshot is index 0 for the observation window).
    out = {"offline_style": ([], []), "runtime": ([], [])}
    # Offline-style EMA seeded at first decode step, per layer, keyed by block id.
    off_ema = [dict() for _ in range(n_layers)]
    n_steps = len(rec.steps[0])
    for t in range(1, n_steps - horizon):      # index 0 = prefill window; decode steps start at 1
        prev_off_norm = None
        prev_rt_hot = None
        for l in range(n_layers):
            ids_t, cur_t, ema_rt = rec.steps[l][t]
            ids_f, cur_f, _ = rec.steps[l][t + horizon]
            # label: future top-r among blocks present now AND still present later
            present = np.isin(ids_t, ids_f)
            if present.sum() < 2:
                continue
            fut = cur_f[np.searchsorted(ids_f, ids_t[present])]
            y = topk_indicator(fut, r_label)
            # offline-style EMA update (v2 collector: new blocks seeded at their value)
            e = off_ema[l]
            for bid, v in zip(ids_t.tolist(), cur_t.tolist()):
                e[bid] = v if bid not in e else ema_decay * e[bid] + (1 - ema_decay) * v
            for bid in list(e):
                if bid not in set(ids_t.tolist()):
                    del e[bid]
            ema_off = np.array([e[bid] for bid in ids_t.tolist()], np.float32)
            within_off = ema_off / (ema_off.max() + 1e-9)
            if l == 0:
                cross_off = topk_indicator(ema_off, r_label)
            else:
                cross_off = np.zeros_like(within_off)
                if prev_off_norm is not None:
                    hot_ids, hot_flag = prev_off_norm
                    hot = set(hot_ids[hot_flag > 0.5].tolist())
                    cross_off = np.array([1.0 if bid in hot else 0.0 for bid in ids_t.tolist()], np.float32)
            prev_off_norm = (ids_t, topk_indicator(ema_off, r_label))
            # runtime realization (exactly PhysicalKVSession._select)
            within_rt = ema_rt / max(float(ema_rt.max()), 1e-12)
            cross_rt = np.zeros_like(within_rt)
            if l > 0 and prev_rt_hot is not None:
                cross_rt = np.isin(ids_t, prev_rt_hot).astype(np.float32)
            count = max(1, math.ceil(ids_t.shape[0] * cfg.cross_fraction))
            order = np.argsort(-ema_rt, kind="stable")
            prev_rt_hot = ids_t[order[:count]]
            s_off = a * within_off[present] + b * cross_off[present] + bias
            s_rt = a * within_rt[present] + b * cross_rt[present] + bias
            out["offline_style"][0].append(s_off); out["offline_style"][1].append(y)
            out["runtime"][0].append(s_rt); out["runtime"][1].append(y)
    return {k: (np.concatenate(v[0]) if v[0] else np.zeros(0), np.concatenate(v[1]) if v[1] else np.zeros(0))
            for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=str(ROOT / "experiments/predictors/xqp_closed_2view_h4.json"))
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--prefill-chunk", type=int, default=256)
    ap.add_argument("--observation-window", type=int, default=64)
    ap.add_argument("--steps", type=int, default=68, help="decode steps recorded (EOS ignored)")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--r-label", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N requests (0 = all)")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--max-input-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from xqp.physical_kv import EvictionConfig, load_two_view_weights
    from xqp.physical_validation import read_requests
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    weights = load_two_view_weights(args.weights)
    requests = read_requests(args.data)
    if args.limit:
        requests = requests[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=getattr(torch, args.dtype),
                                                 attn_implementation="eager").to(args.device).eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
    common = dict(block_size=args.block_size, prefill_chunk=args.prefill_chunk,
                  observation_window=args.observation_window, backend="physical")
    full_cfg = EvictionConfig(policy="full", budget_fraction=1.0, **common)
    pol_cfg = EvictionConfig(policy="xqp_reconstructed", budget_fraction=args.budget, **common)
    per_request, pooled = [], {k: ([], []) for k in ("A_offline_full", "B_runtime_full", "C_runtime_policy")}
    t0 = time.time()
    for i, row in enumerate(requests):
        if args.chat:
            ids = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=True,
                                          add_generation_prompt=True, return_tensors="pt")
        else:
            ids = tok(row["prompt"], return_tensors="pt").input_ids
        if ids.shape[1] > args.max_input_tokens:
            raise ValueError("input exceeds cap; freeze explicit truncation first")
        ids = ids.to(args.device)
        rec_full, gen_full = run_trajectory(model, ids, full_cfg, None, args.steps, Recorder, eos_ids)
        fl = features_and_labels(rec_full, full_cfg, weights, args.horizon, args.r_label, full_cfg.ema_decay)
        rec_pol, gen_pol = run_trajectory(model, ids, pol_cfg, weights, args.steps, Recorder, eos_ids)
        pl = features_and_labels(rec_pol, pol_cfg, weights, args.horizon, args.r_label, pol_cfg.ema_decay)
        item = dict(dataset=row["dataset"], id=row["id"], input_tokens=int(ids.shape[1]),
                    A_offline_full=auc(fl["offline_style"][1], fl["offline_style"][0]),
                    B_runtime_full=auc(fl["runtime"][1], fl["runtime"][0]),
                    C_runtime_policy=auc(pl["runtime"][1], pl["runtime"][0]),
                    rows_full=int(fl["runtime"][1].size), rows_policy=int(pl["runtime"][1].size),
                    pos_rate_full=float(fl["runtime"][1].mean()), pos_rate_policy=float(pl["runtime"][1].mean()),
                    same_first_tokens=int(sum(1 for a_, b_ in zip(gen_full, gen_pol) if a_ == b_)))
        per_request.append(item)
        for key, (s, y) in (("A_offline_full", fl["offline_style"]), ("B_runtime_full", fl["runtime"]),
                            ("C_runtime_policy", pl["runtime"])):
            pooled[key][0].append(s.astype(np.float32)); pooled[key][1].append(y.astype(np.int8))
        print(f"[{i+1}/{len(requests)}] {row['dataset']}/{row['id']} A={item['A_offline_full']:.3f} "
              f"B={item['B_runtime_full']:.3f} C={item['C_runtime_policy']:.3f}", flush=True)
        del rec_full, rec_pol
    rng = np.random.default_rng(args.seed)
    keys = ("A_offline_full", "B_runtime_full", "C_runtime_policy")
    mat = np.array([[r[k] for k in keys] for r in per_request])
    boot = rng.choice(len(per_request), size=(2000, len(per_request)), replace=True)

    def ci(v):
        return [float(np.quantile(v[boot].mean(1), q)) for q in (0.025, 0.975)]
    summary = {k: dict(mean_request_auc=float(mat[:, j].mean()), ci95=ci(mat[:, j]),
                       pooled_auc=auc(np.concatenate(pooled[k][1]).astype(np.float32),
                                      np.concatenate(pooled[k][0]).astype(np.float64)))
               for j, k in enumerate(keys)}
    for name, (x, y) in (("B_minus_A_feature_realization", (1, 0)), ("C_minus_B_trajectory", (2, 1)),
                         ("C_minus_A_total", (2, 0))):
        d = mat[:, x] - mat[:, y]
        summary[name] = dict(mean=float(d.mean()), ci95=ci(d), n_negative=int((d < 0).sum()),
                             n_positive=int((d > 0).sum()))
    payload = dict(schema="feature-bridge-v1", model=args.model, args=vars(args), n_requests=len(per_request),
                   label="top ceil(r*n) of block-mean attention at t+h among blocks present at t (and still present at t+h)",
                   scorer=dict(weights=list(weights), sha256=sha256_file(args.weights)),
                   data_sha256=sha256_file(args.data),
                   runtime=dict(python=platform.python_version(), torch=torch.__version__,
                                transformers=importlib.metadata.version("transformers"),
                                gpu=torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else None,
                                co_tenancy_note="quality/AUC only; no timing is reported"),
                   summary=summary, per_request=per_request, wall_s=time.time() - t0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
