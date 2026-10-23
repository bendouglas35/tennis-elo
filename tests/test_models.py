import numpy as np
import pandas as pd
import pytest

from tennis import dataset, models


def _synthetic(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in dataset.ANTI})
    for c in dataset.SYM:
        df[c] = rng.normal(size=n)
    df.loc[rng.random(n) < 0.1, "d_f_ace_rate"] = np.nan
    df["d_x_pace_s_ace"] = df["pace_static"] * df["d_f_ace_rate"]      # the real interaction column
    logit = 1.2 * df["d_elo_blend"] + 1.5 * df["d_x_pace_s_ace"].fillna(0)
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    df["yr"] = np.array([2019, 2020, 2021])[np.arange(n) * 3 // n]
    df["match_id"] = np.arange(n).astype(str)
    df["tourney_id"] = (np.arange(n) // 10).astype(str)
    return df


def test_mirror_is_an_involution_and_flips_anti_columns_only():
    df = _synthetic(50)
    X = df[dataset.ALL_COLUMNS].copy()
    M = dataset.mirror(X)
    pd.testing.assert_frame_equal(dataset.mirror(M), X)
    assert (M["pace_static"] == X["pace_static"]).all()
    assert np.allclose(M["d_elo_blend"], -X["d_elo_blend"])


def test_models_are_exactly_antisymmetric():
    df = _synthetic()
    cols = dataset.columns_for(["elo", "form", "pace_static", "context"])
    mirrored = df.copy()
    mirrored[dataset.ALL_COLUMNS] = dataset.mirror(df[dataset.ALL_COLUMNS])
    lr = models.LRModel(cols, C=0.3).fit(df)
    assert np.allclose(lr.predict(df) + lr.predict(mirrored), 1.0, atol=1e-9)
    gbm = models.GBMModel(cols, n_estimators=40, num_leaves=4, min_child_samples=20).fit(df)
    assert np.allclose(gbm.predict(df) + gbm.predict(mirrored), 1.0, atol=1e-9)


def test_models_learn_the_planted_signal_including_the_interaction():
    df = _synthetic(3000)
    tr, ev = df[df["yr"] < 2021], df[df["yr"] == 2021]
    base = dataset.columns_for(["elo", "form", "context"])
    with_pace = base + dataset.GROUPS["pace_static"]
    ll = lambda m: models.log_loss(ev["y"], m.fit(tr).predict(ev))
    assert ll(models.LRModel(with_pace, 0.3)) < ll(models.LRModel(base, 0.3)) - 0.01
    assert ll(models.GBMModel(with_pace, 150, num_leaves=4, min_child_samples=20)) < 0.66


def test_lr_ignores_symmetric_columns_and_handles_missing():
    df = _synthetic(600)
    lr = models.LRModel(dataset.columns_for(["elo", "context", "form"])).fit(df)
    assert set(lr.columns) <= set(dataset.ANTI)
    assert np.isfinite(lr.predict(df)).all()


def test_platt_slope_recovers_overconfidence():
    rng = np.random.default_rng(1)
    z = rng.normal(size=20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-z))).astype(float)
    over = 1 / (1 + np.exp(-2.0 * z))                  # twice too confident
    pl = models.Platt().fit(over, y)
    assert pl.slope_ == pytest.approx(0.5, abs=0.06)


def test_rolling_origin_never_trains_on_the_evaluated_year():
    df = _synthetic(900)
    seen = []

    class Spy(models.LRModel):
        def fit(self, tr, y=None):
            seen.append(int(tr["yr"].max()))
            return super().fit(tr, y)

    models.rolling_origin(df, lambda: Spy(dataset.columns_for(["elo"])), [2020, 2021])
    assert seen == [2019, 2020]


def test_development_frame_refuses_test_years():
    df = pd.DataFrame({"yr": [2021, 2023]})
    with pytest.raises(RuntimeError):
        dataset.assert_no_test_years(df)
