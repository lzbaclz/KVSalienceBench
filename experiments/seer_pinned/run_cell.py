#!/usr/bin/env python3
"""Run ONE masked-loop cell with the pinned external SEER revision, fail-closed.

The archived expansion sweep (experiments/results/expand) was produced by
``seer.eval.runner`` at an unpinned revision, in its float16 default. This
wrapper makes a rerun auditable:

* ``--seer-root`` must contain ``PINNED_GIT_SHA.txt`` (a read-only ``git
  archive`` export of one SEER commit); the wrapper hashes every ``seer/*.py``
  file and records the SHA, the hashes, the runtime versions and the exact
  runner arguments in ``<out>.provenance.json``.
* ``--dtype`` is mandatory; nothing defaults to float16 silently.
* One documented, numerically neutral patch is applied in-process: after the
  simulator's attention post-hook has ingested a layer's attention weights, the
  weights are replaced by a 1x1x1x1 placeholder in the tuple returned to the
  HuggingFace decoder layer. Without this the prefill forward with
  ``output_attentions=True`` retains every layer's [H, 4096, 4096] attention
  matrix (26-34 GB) until the forward returns. The ingested statistics, kept
  sets, masks, tokens and scores are unchanged, which the CPU test in
  ``tests/test_seer_pinned_patch.py`` verifies on a tiny model.
* The output must not exist; the runner's JSON is left exactly as written.
  A record of the per-item references and both scorers is produced later by
  ``experiments/analyze_expand_sensitivity.py`` from the local LongBench rows.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install_attention_memory_patch(sim_module) -> str:
    """Wrap MaskingSimulator._make_post_hook so the HF layer keeps a placeholder."""
    original = sim_module.MaskingSimulator._make_post_hook

    def _make_post_hook(self, layer_id: int):
        import torch
        inner = original(self, layer_id)

        def hook(module, inputs, outputs):
            inner(module, inputs, outputs)          # ingestion exactly as before
            if isinstance(outputs, tuple) and len(outputs) >= 2 and torch.is_tensor(outputs[1]) \
                    and outputs[1].ndim == 4 and outputs[1].shape[-2] > 1:
                # Prefill only (Q > 1): drop the O(L^2) matrix from what the
                # decoder layer collects into out.attentions. Decode steps are O(L).
                return (outputs[0], outputs[1][:, :1, :1, :1].detach().clone()) + tuple(outputs[2:])
            return None
        return hook

    sim_module.MaskingSimulator._make_post_hook = _make_post_hook
    return "MaskingSimulator._make_post_hook: prefill attention weights replaced by a 1x1x1x1 placeholder after ingestion"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seer-root", required=True, help="read-only export with PINNED_GIT_SHA.txt")
    p.add_argument("--model", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--dataset", required=True, help="LongBench task name; LONGBENCH_PATH is derived from --longbench-dir")
    p.add_argument("--longbench-dir", default="/public/data_zoo/longbench/data")
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", required=True, choices=["float16", "bfloat16", "float32"])
    p.add_argument("--context-length", type=int, default=4096)
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--hbm-budget", type=float, default=0.20)
    p.add_argument("--slo", default="P99=200ms")
    p.add_argument("--io-mode", default="measured-dma")
    p.add_argument("--decision-period", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--xqp-ckpt", default=str(ROOT / "experiments/predictors/xqp_closed_2view_h4.json"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-chat", action="store_true", help="the archived sweep used --chat; keep it unless deliberately changed")
    p.add_argument("--no-memory-patch", action="store_true", help="run the pinned code unpatched (needs an idle 80GB GPU)")
    args = p.parse_args(argv)

    seer_root = Path(args.seer_root).resolve()
    sha_file = seer_root / "PINNED_GIT_SHA.txt"
    if not sha_file.exists():
        raise SystemExit(f"{seer_root} is not a pinned export (missing PINNED_GIT_SHA.txt)")
    sys.path.insert(0, str(ROOT))
    from scripts.prepare_seer import REVISION, expected_sources
    seer_files = {str(f.relative_to(seer_root)): sha256_file(f) for f in sorted(seer_root.rglob("*.py"))}
    if sha_file.read_text().strip() != REVISION or seer_files != expected_sources():
        raise SystemExit("simulator source does not match all 70 pinned rerun records")
    out = Path(args.out).resolve()
    prov_path = Path(str(out) + ".provenance.json")
    if out.exists() or prov_path.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    lb_path = Path(args.longbench_dir) / f"{args.dataset}.jsonl"
    if not lb_path.exists():
        raise SystemExit(f"missing LongBench rows: {lb_path}")
    if args.policy == "xqp" and not Path(args.xqp_ckpt).exists():
        raise SystemExit(f"missing scorer checkpoint {args.xqp_ckpt}")
    out.parent.mkdir(parents=True, exist_ok=True)

    os.environ["LONGBENCH_PATH"] = str(lb_path)
    os.environ["SEER_STRICT_WORKLOAD"] = "1"
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for var in ("LONGBENCH_SEED", "LONGBENCH_OFFSET"):
        if os.environ.get(var):
            raise SystemExit(f"{var} is set; the archived sweep used canonical first-N rows")
    sys.path.insert(0, str(seer_root))
    import torch
    import transformers
    sim = importlib.import_module("seer.eval.sim")
    runner = importlib.import_module("seer.eval.runner")
    patch = None if args.no_memory_patch else install_attention_memory_patch(sim)

    runner_argv = ["--model", args.model, "--policy", args.policy, "--workload", "longbench",
                   "--context_length", str(args.context_length), "--num_requests", str(args.num_requests),
                   "--max_new_tokens", str(args.max_new_tokens), "--hbm_budget", str(args.hbm_budget),
                   "--slo", args.slo, "--io_mode", args.io_mode, "--decision_period", str(args.decision_period),
                   "--seed", str(args.seed), "--dtype", args.dtype, "--device", args.device, "--out", str(out)]
    if not args.no_chat:
        runner_argv.append("--chat")
    if args.policy == "xqp":
        runner_argv += ["--xqp-ckpt", args.xqp_ckpt]
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                       "--format=csv,noheader"], text=True).strip()
    except Exception:  # noqa: BLE001
        gpu = None
    provenance = dict(schema="seer-pinned-cell-v1", status="running",
                      seer_git_sha=sha_file.read_text().strip(), seer_root=str(seer_root),
                      seer_source_sha256=seer_files, memory_patch=patch,
                      runner_argv=runner_argv, wrapper_args=vars(args),
                      longbench_path=str(lb_path), longbench_sha256=sha256_file(lb_path),
                      row_selection="canonical first-N rows (LONGBENCH_SEED/OFFSET unset): request id i is source row i",
                      xqp_ckpt_sha256=sha256_file(Path(args.xqp_ckpt)) if args.policy == "xqp" else None,
                      runtime=dict(python=platform.python_version(), torch=torch.__version__,
                                   transformers=importlib.metadata.version("transformers"),
                                   cuda=torch.version.cuda, gpu_state_at_launch=gpu,
                                   co_tenancy_note="latency fields are not reportable if the GPU was shared"),
                      wrapper_sha256=sha256_file(Path(__file__)), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
    sys.argv = ["seer.eval.runner"] + runner_argv
    t0 = time.time()
    try:
        runner.main()
    except BaseException as exc:  # noqa: BLE001
        provenance.update(status="failed", error=repr(exc), wall_s=time.time() - t0)
        prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
        raise
    if not out.exists():
        provenance.update(status="failed", error="runner produced no output", wall_s=time.time() - t0)
        prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
        return 1
    obj = json.loads(out.read_text())
    n = len(obj.get("results", []))
    ids = {r.get("id") for r in obj.get("results", [])}
    complete = n == args.num_requests and len(ids) == n
    provenance.update(status="passed" if complete else "incomplete", n_results=n, n_unique_ids=len(ids),
                      output_sha256=sha256_file(out), wall_s=time.time() - t0,
                      finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[run_cell] {args.policy}/{args.dataset}: {n} results, status={provenance['status']}, {provenance['wall_s']:.0f}s")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
