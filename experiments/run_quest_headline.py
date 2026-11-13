"""Retired version-1 query-control driver. No historical JSON is modified.

The old driver used unpinned external SEER prompts, reused .done sentinels,
accepted missing features, and described pre-/post-RoPE products as faithful
Quest evidence. See docs/CAMERA_READY_AUDIT.md. Original source is preserved
in experiments/legacy/run_quest_headline_v1.py for forensic reference only.

Use experiments/run_query_control_v2.py with a new frozen prompt cohort.
"""
if __name__ == "__main__":
    raise SystemExit("Retired: use experiments/run_query_control_v2.py; see docs/PHYSICAL_KV_VALIDATION.md")
