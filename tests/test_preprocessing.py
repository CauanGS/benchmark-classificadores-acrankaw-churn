"""Testes da discretização Fayyad–Irani (MDLP) e utilidades associadas."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    MDLPDiscretizer,
    assign_bins,
    cuts_to_interval_labels,
    fayyad_irani_cuts,
)


def test_single_class_has_no_cuts():
    x = np.arange(20, dtype=float)
    y = np.array(["a"] * 20)
    assert fayyad_irani_cuts(x, y) == []


def test_two_separable_clusters_single_cut():
    x = np.array([1, 2, 3, 4, 10, 11, 12, 13], dtype=float)
    y = np.array(["n", "n", "n", "n", "p", "p", "p", "p"])
    cuts = fayyad_irani_cuts(x, y)
    assert len(cuts) == 1
    assert 4 < cuts[0] < 10


def test_cut_is_midpoint():
    x = np.array([1, 2, 10, 11], dtype=float)
    y = np.array(["n", "n", "p", "p"])
    (cut,) = fayyad_irani_cuts(x, y)
    assert cut == pytest.approx((2 + 10) / 2)


def test_value_equal_to_cut_goes_to_left_bin():
    x = np.array([1, 2, 10, 11], dtype=float)
    y = np.array(["n", "n", "p", "p"])
    (cut,) = fayyad_irani_cuts(x, y)
    bins = assign_bins(np.array([2, 10, cut, 20]), [cut])
    assert list(bins) == [0, 1, 0, 1]


def test_pure_noise_produces_no_cuts():
    rng = np.random.default_rng(42)
    x = rng.normal(size=500)
    y = rng.choice(["n", "p"], size=500)
    assert fayyad_irani_cuts(x, y) == []


def test_three_clusters_two_cuts():
    x = np.concatenate([[0] * 20, [10] * 20, [20] * 20]).astype(float)
    y = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
    cuts = fayyad_irani_cuts(x, y)
    assert len(cuts) == 2
    lo, hi = sorted(cuts)
    assert 0 < lo < 10 and 10 < hi < 20


def test_order_independent_with_duplicates():
    x = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    y = np.array(["n", "p", "n", "p", "p"])
    a = fayyad_irani_cuts(x, y)
    b = fayyad_irani_cuts(x[::-1], y[::-1])
    assert a == b


def test_interval_labels_format():
    labels = cuts_to_interval_labels([5.5])
    assert labels == ["(-∞, 5.5]", "(5.5, +∞]"]
    labels2 = cuts_to_interval_labels([])
    assert labels2 == ["(-∞, +∞]"]


def test_discretizer_drops_column_without_gain():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"num": rng.normal(size=200)})
    y = rng.choice(["n", "p"], size=200)
    disc = MDLPDiscretizer()
    disc.fit(df, y)
    assert disc.dropped_columns_ == ["num"]
    assert disc.passthrough_columns_ == []


def test_discretizer_passthrough_low_cardinality():
    df = pd.DataFrame({"cat": ["a", "b"] * 50, "num": [1, 2] * 50})
    y = np.array(["n", "p"] * 50)
    disc = MDLPDiscretizer(min_numeric_cardinality=10)
    t = disc.fit_transform(df, y)
    assert disc.passthrough_columns_ == ["cat", "num"]
    assert (t["cat"] == t["cat"].astype(str)).all()


def test_discretizer_binning_is_deterministic():
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.normal(-3, 0.5, 200), rng.normal(3, 0.5, 200)])
    y = np.array(["n"] * 200 + ["p"] * 200)
    df = pd.DataFrame({"x": x})
    d1 = MDLPDiscretizer().fit_transform(df, y)
    d2 = MDLPDiscretizer().fit_transform(df, y)
    assert d1["x"].equals(d2["x"])
