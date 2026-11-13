"""Regression checks for the runtime and LongBench scoring conventions."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import analyze_expand_sensitivity as aes  # noqa: E402








def test_normalization_order_is_the_documented_difference():
    # articles-before-punctuation (runner) vs punctuation-before-articles (LongBench)
    assert aes.runner_f1("![](The!Rebirth)", "The Rebirth") == 1.0
    assert aes.longbench_f1("![](The!Rebirth)", ["The Rebirth"], "hotpotqa") == 0.0
    assert aes.longbench_f1("first line\nsecond", ["first line"], "triviaqa") == 1.0
