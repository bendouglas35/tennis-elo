import numpy as np
import pytest

from tennis import metrics


def test_vectors_match_scalar_metrics():
    y = np.array([1, 0, 1, 1, 0.0]); p = np.array([0.8, 0.3, 0.6, 0.5, 0.1])
    assert metrics.log_loss_vec(y, p).mean() == pytest.approx(metrics.log_loss(y, p))
    assert metrics.brier_vec(y, p).mean() == pytest.approx(metrics.brier(y, p))
    assert metrics.accuracy_vec(y, p).tolist() == [1, 1, 1, 0.5, 1]


def test_cluster_bootstrap_mean_and_coverage():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 10)
    x = rng.normal(1.0, 1.0, 600) + np.repeat(rng.normal(0, 0.5, 60), 10)
    res = metrics.cluster_bootstrap(groups, {"x": x, "z": x * 0}, diffs=[("x", "z")], n_boot=500, seed=1)
    mean, lo, hi = res["x"]
    assert mean == pytest.approx(x.mean()) and lo < mean < hi
    assert res["x - z"][0] == pytest.approx(x.mean())
    # clustered intervals are wider than naive ones when groups share an offset
    naive = 1.96 * x.std() / np.sqrt(len(x))
    assert (hi - lo) / 2 > naive
