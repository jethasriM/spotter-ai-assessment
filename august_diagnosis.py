"""Where does the two-stage model lose in a mixed regime? Run: python august_diagnostic.py [--months 2025-08 2025-10]

Same leave-one-WEEK-out setup as regime_check.py (hold out a week, drop its neighbours, keep the rest),
but each held-out row is split by the true good/bad flag of quote_signal (train only; uses labels).

Prints, per month:
  1. MAE of base Ridge, twostage, and an ORACLE gate (twostage on truly-good rows, base Ridge on bad rows).
     oracle vs twostage = headroom from better row sorting. oracle vs base = the ceiling the data allows.
  2. MAE split by good / bad rows, and how big a correction stage 2 applies to each (mean |log| shift).
     If twostage applies similar corrections to good and bad rows, it is not telling them apart.
  3. Separability: share of good vs bad rows whose quote_signal is within 10% of the base estimate.
     If many bad rows look "agreeing", no feature built from agreement can separate them.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from train_predict import (TARGET, TwoStage, build_city_coords, clean_train, find_data_dir,
                           make_features, to_dollars)

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=None)
ap.add_argument("--months", nargs="+", default=["2025-08", "2025-10"])
args = ap.parse_args()

train, _ = clean_train(pd.read_csv(find_data_dir(args.data_dir) / "train_test.csv"))
week = train["date"].dt.to_period("W")
month = train["date"].dt.to_period("M").astype(str)
y_log = np.log(train[TARGET] / train["distance"])

parts = []
for m in args.months:
    for w in sorted(week[month == m].unique()):
        te = np.where((week == w) & (month == m))[0]
        if len(te) < 200:
            continue
        tr = np.where(~week.isin([w - 1, w, w + 1]))[0]
        dtr, dte = train.iloc[tr], train.iloc[te]
        coords = build_city_coords(dtr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = TwoStage(coords).fit(dtr, y_log.iloc[tr])
            s1 = ts.base_.predict(make_features(dte, coords, "base"))
            lp = ts.predict(dte)
        y = dte[TARGET].values
        p_base = to_dollars(s1, dte["distance"])
        p_ts = to_dollars(lp, dte["distance"])
        good = ((dte["quote_signal"] / (dte[TARGET] / dte["distance"]) - 1).abs() <= 0.10).values
        parts.append(pd.DataFrame({
            "month": m, "week": str(w.start_time.date()), "good": good,
            "err_base": np.abs(p_base - y), "err_ts": np.abs(p_ts - y),
            "err_oracle": np.abs(np.where(good, p_ts, p_base) - y),
            "shift": np.abs(lp - s1),
            "err_quote": np.abs(dte["quote_signal"].values * dte["distance"].values - y),
            "quote_relerr": np.abs(dte["quote_signal"].values / (y / dte["distance"].values) - 1),
            "looks_agree": (np.log(dte["quote_signal"].values) - s1 < TwoStage.AGREE_TOL)
                           & (np.log(dte["quote_signal"].values) - s1 > -TwoStage.AGREE_TOL),
        }))
d = pd.concat(parts, ignore_index=True)

print("\n1) MAE by month (oracle = perfect good/bad gate):")
print(d.groupby("month")[["err_base", "err_ts", "err_oracle"]].mean().round(1).rename(columns=lambda c: c[4:]))

print("\n2) By true quote_signal quality:")
g = d.groupby(["month", "good"]).agg(n=("good", "size"), base=("err_base", "mean"), twostage=("err_ts", "mean"),
                                     mean_abs_log_shift=("shift", "mean")).round(3)
g["share"] = (g["n"] / g.groupby("month")["n"].transform("sum")).round(2)
print(g)

print("\n3) Share of rows whose quote_signal is within 10% of the base estimate:")
print(d.groupby(["month", "good"])["looks_agree"].mean().unstack().round(2).rename(columns={False: "bad rows", True: "good rows"}))

print("\n3b) How precise is quote_signal itself? (quote_signal * distance used directly as the prediction)")
q = d.groupby(["month", "good"]).agg(quote_MAE=("err_quote", "mean"), twostage_MAE=("err_ts", "mean"),
                                     rel_err_mean=("quote_relerr", "mean"),
                                     rel_err_p90=("quote_relerr", lambda s: s.quantile(0.9))).round(3)
print(q)

print("\n4) Twostage minus base MAE by week (negative = twostage better), with good-row share:")
wk = d.groupby(["month", "week"]).agg(good_share=("good", "mean"), diff=("err_ts", "mean"), base=("err_base", "mean"))
wk["diff"] = wk["diff"] - wk["base"]
print(wk[["good_share", "diff"]].round(2))