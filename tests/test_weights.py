"""Tests for the weight-derivation logic in scripts/compute_weights.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import compute_weights as cw


class TestClusterAdjustment:
    def test_single_member_unchanged(self):
        raw = {"a": 0.5}
        clusters = {"a": "x"}
        out = cw._adjust_for_clusters(raw, clusters)
        assert out["a"] == pytest.approx(0.5)

    def test_two_member_cluster_max_weight_split_proportionally(self):
        # cluster_max = 0.8, cluster_sum = 1.4, shares: 0.8 * (raw/sum)
        raw = {"a": 0.8, "b": 0.6}
        clusters = {"a": "x", "b": "x"}
        out = cw._adjust_for_clusters(raw, clusters)
        assert out["a"] == pytest.approx(0.8 * 0.8 / 1.4)
        assert out["b"] == pytest.approx(0.8 * 0.6 / 1.4)
        # cluster total contribution equals max member
        assert (out["a"] + out["b"]) == pytest.approx(0.8)

    def test_zero_cluster_yields_zero(self):
        raw = {"a": 0.0, "b": 0.0}
        clusters = {"a": "x", "b": "x"}
        out = cw._adjust_for_clusters(raw, clusters)
        assert out == {"a": 0.0, "b": 0.0}

    def test_higher_auc_member_gets_larger_share(self):
        raw = {"strong": 0.9, "weak": 0.3}
        clusters = {"strong": "c", "weak": "c"}
        out = cw._adjust_for_clusters(raw, clusters)
        assert out["strong"] > out["weak"]


class TestSeededWeightsConsistency:
    def test_weights_sum_to_one(self):
        weights = json.loads((ROOT / "indicators" / "weights.json").read_text())
        weights = {k: v for k, v in weights.items() if not k.startswith("_")}
        s = sum(v["weight"] for v in weights.values())
        assert s == pytest.approx(1.0, abs=0.005)

    def test_all_aucs_in_valid_range(self):
        weights = json.loads((ROOT / "indicators" / "weights.json").read_text())
        for k, v in weights.items():
            if k.startswith("_"):
                continue
            assert 0.5 <= v["auc"] <= 1.0, f"{k} AUC {v['auc']} out of range"

    def test_high_tier_has_top_aucs(self):
        weights = json.loads((ROOT / "indicators" / "weights.json").read_text())
        # The literature says yield curve, Sahm, EBP/HY spread should be the top signals
        top_aucs = sorted(
            [(k, v["auc"]) for k, v in weights.items() if not k.startswith("_")],
            key=lambda x: -x[1],
        )[:5]
        top_keys = {k for k, _ in top_aucs}
        assert "sahm_rule" in top_keys
        assert "yield_curve_10y3m" in top_keys
        assert "hy_credit_spread" in top_keys

    def test_unconventional_weights_small(self):
        weights = json.loads((ROOT / "indicators" / "weights.json").read_text())
        unconventional = ["lipstick_proxy", "rword_index", "google_trends_recession", "copper_gold_ratio_yoy"]
        total = sum(weights[k]["weight"] for k in unconventional)
        assert total < 0.10, f"Unconventional indicators total weight {total} too high"
