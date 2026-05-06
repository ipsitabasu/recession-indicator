"""Registry consistency tests."""
import pytest

from indicators.registry import INDICATORS


def test_unique_keys():
    keys = [i.key for i in INDICATORS]
    assert len(keys) == len(set(keys)), "duplicate indicator keys"


def test_count_is_seventeen():
    assert len(INDICATORS) == 17


def test_each_indicator_has_required_metadata():
    for ind in INDICATORS:
        assert ind.key
        assert ind.name
        assert ind.tier in ("high", "medium", "low")
        assert ind.cluster
        assert ind.description
        assert ind.rationale
        assert callable(ind.fetch)
        assert callable(ind.score_fn)
        assert 0.0 <= ind.weight <= 1.0
        assert 0.0 <= ind.auc <= 1.0


def test_weights_sum_to_one():
    s = sum(i.weight for i in INDICATORS)
    assert s == pytest.approx(1.0, abs=0.005)


def test_clusters_present():
    expected = {"yield-curve", "labor", "credit", "activity", "housing", "sentiment", "markets"}
    actual = {i.cluster for i in INDICATORS}
    assert expected.issubset(actual)


def test_high_tier_is_top_weighted():
    # All high-tier indicators should outweigh all low-tier ones
    high_min = min(i.weight for i in INDICATORS if i.tier == "high")
    low_max = max(i.weight for i in INDICATORS if i.tier == "low")
    assert high_min > low_max
