"""How trustworthy is quote_signal? Run: python diagnose_quote.py [--data-dir data]"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from train_predict import basic_clean, find_data_dir

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=None)
args = ap.parse_args()
tr = basic_clean(pd.read_csv(find_data_dir(args.data_dir) / "train_test.csv"))
tr["rpm"] = tr["posted_rate"] / tr["distance"]
tr["ratio"] = tr["quote_signal"] / tr["rpm"]

print("quote_signal / rpm quantiles:")
print(tr["ratio"].quantile([.01, .05, .10, .25, .5, .75, .90, .95, .99]).round(3).to_string())

print("\nShare of rows where quote_signal is within X of rpm:")
for tol in (0.03, 0.05, 0.10, 0.25):
    print(f"  +/-{tol:.0%}: {(tr['ratio'] - 1).abs().le(tol).mean():.1%}")

tr["good"] = (tr["ratio"] - 1).abs() <= 0.10
print("\nShare 'good' (within 10%) by group -> are the bad rows concentrated somewhere?")
for name, key in [("month", tr["date"].dt.to_period("M")), ("equipment", tr["equipment"]),
                  ("weekday", tr["date"].dt.dayofweek),
                  ("distance decile", pd.qcut(tr["distance"], 10, labels=False)),
                  ("market_index missing", tr["market_index"].isna()),
                  ("weight missing", tr["weight"].isna())]:
    print(f"\n{name}:\n", tr.groupby(key)["good"].mean().round(3).to_string())

bad = tr[~tr["good"]]
print(f"\nBad rows: {len(bad)} ({len(bad)/len(tr):.1%})")
print("  rpm outside [1,4]          :", f"{(~bad['rpm'].between(1, 4)).mean():.1%}", "(label looks corrupted)")
print("  quote_signal outside [1,4] :", f"{(~bad['quote_signal'].between(1, 4)).mean():.1%}", "(feature looks corrupted)")
print("  both plausible but differ  :", f"{(bad['rpm'].between(1, 4) & bad['quote_signal'].between(1, 4)).mean():.1%}")
print("\nQuote_signal quantiles, good vs bad rows:")
print(pd.DataFrame({"good": tr.loc[tr.good, "quote_signal"].describe(), "bad": bad["quote_signal"].describe()}).round(3))

pred = tr["quote_signal"] * tr["distance"]
mae = lambda m: (pred[m] - tr.loc[m, "posted_rate"]).abs().mean()
print(f"\nMAE of the trivial rule rate = quote_signal x distance: all rows {mae(tr.index == tr.index):.1f}"
      f" | good rows {mae(tr.good):.1f} | bad rows {mae(~tr.good):.1f}")