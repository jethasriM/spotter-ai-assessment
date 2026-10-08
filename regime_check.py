"""Leave-one-WEEK-out check inside a regime. Run: python regime_check.py [--months 2025-08 2025-10]

Lomo removes a whole month, so a month with a unique regime (August: ~46% good quote_signal, like
Nov/Dec) has no analogue in training. Here we hold out one week and keep the other weeks of that month,
excluding only the week itself and its neighbours (to limit leakage across adjacent days).
Needs TwoStage, clean_train, build_city_coords, make_features, make_model, to_dollars, metrics from train_predict.py.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from train_predict import (TARGET, TwoStage, build_city_coords, clean_train, find_data_dir,
                           make_features, make_model, metrics, to_dollars)

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=None)
ap.add_argument("--months", nargs="+", default=["2025-08"])
args = ap.parse_args()

train, _ = clean_train(pd.read_csv(find_data_dir(args.data_dir) / "train_test.csv"))
week = train["date"].dt.to_period("W")
month = train["date"].dt.to_period("M").astype(str)
y_log = np.log(train[TARGET] / train["distance"])
rows = []
for m in args.months:
    for w in sorted(week[month == m].unique()):
        te = np.where((week == w) & (month == m))[0]
        if len(te) < 200:  # skip tiny edge weeks
            continue
        tr = np.where(~week.isin([w - 1, w, w + 1]))[0]
        dtr, dte = train.iloc[tr], train.iloc[te]
        coords = build_city_coords(dtr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = TwoStage(coords).fit(dtr, y_log.iloc[tr])
            Xb = make_features(dtr, coords, "base")
            rg = make_model("ridge", Xb).fit(Xb, y_log.iloc[tr])
        for name, lp in (("twostage", ts.predict(dte)), ("base_ridge", rg.predict(make_features(dte, coords, "base")))):
            rows.append({"month": m, "week": str(w.start_time.date()), "model": name, "n": len(te),
                         **metrics(dte[TARGET].values, to_dollars(lp, dte["distance"]))})
res = pd.DataFrame(rows)
print(res.pivot_table(index=["month", "week"], columns="model", values="MAE").round(1))
print("\nMean MAE by month:\n", res.groupby(["month", "model"])["MAE"].mean().unstack().round(1))