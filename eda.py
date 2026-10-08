"""Exploratory data analysis for the freight rate assessment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_predict import find_data_dir


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default="eda_outputs")
    args = ap.parse_args()
    data_dir = find_data_dir(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(data_dir / "train_test.csv")
    va = pd.read_csv(data_dir / "validation.csv")
    dec = pd.read_csv(data_dir / "december_chart_inputs.csv")

    section("1. Shapes and columns")
    print("train_test:", tr.shape, "| validation:", va.shape, "| december:", dec.shape)
    print("columns only in train:", sorted(set(tr.columns) - set(va.columns)))
    print("columns only in validation:", sorted(set(va.columns) - set(tr.columns)))
    print("december columns:", list(dec.columns))

    section("2. Dtypes and missing values (train)")
    print(pd.DataFrame({"dtype": tr.dtypes.astype(str), "n_missing": tr.isna().sum()}))
    print("\nMissing share in validation:\n", va.isna().mean().round(4).to_string())

    section("3. Duplicates")
    print("duplicate load_id:", tr["load_id"].duplicated().sum())
    print("fully duplicated rows (ignoring load_id):", tr.drop(columns="load_id").duplicated().sum())

    for df in (tr, va):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    section("4. Date coverage (decides the split strategy)")
    print("train :", tr["date"].min(), "->", tr["date"].max(), "| unparseable:", tr["date"].isna().sum())
    print("valid :", va["date"].min(), "->", va["date"].max(), "| unparseable:", va["date"].isna().sum())
    print("train rows per month:\n", tr.groupby(tr["date"].dt.to_period("M")).size().to_string())
    overlap = va["date"].min() <= tr["date"].max()
    print("\nValidation overlaps train dates?", overlap,
          "-> random/group split is OK" if overlap else "-> use a forward-in-time split")

    section("5. Numeric summary (train)")
    num = ["distance", "weight", "market_index", "quote_signal", "posted_rate"]
    print(tr[[c for c in num if c in tr]].describe(percentiles=[.01, .5, .99]).T.round(3))

    tr["rpm"] = tr["posted_rate"] / tr["distance"]
    section("6. Rate per mile")
    print(tr["rpm"].describe(percentiles=[.001, .01, .5, .99, .999]).round(3).to_string())
    print("non-positive rates:", (tr["posted_rate"] <= 0).sum(), "| non-positive distance:", (tr["distance"] <= 0).sum())
    print("\nmedian rpm by equipment:\n", tr.groupby("equipment")["rpm"].agg(["count", "median", "std"]).round(3))

    if "quote_signal" in tr:
        section("7. quote_signal vs rate per mile")
        clean = tr[tr["rpm"].between(1, 5)]
        print("Pearson, all rows (outlier-driven):", round(tr["quote_signal"].corr(tr["rpm"]), 4))
        print("Spearman, all rows:", round(tr["quote_signal"].corr(tr["rpm"], method="spearman"), 4))
        print("Pearson, rpm in [1,5]:", round(clean["quote_signal"].corr(clean["rpm"]), 4))
        print("Median |quote_signal/rpm - 1|, rpm in [1,5]:",
              round(((clean["quote_signal"] / clean["rpm"]) - 1).abs().median(), 4))
        print("quote_signal nulls: train", tr["quote_signal"].isna().mean().round(4),
              "| validation", va["quote_signal"].isna().mean().round(4) if "quote_signal" in va else "column absent")

    section("8. Signals over time: train vs validation)")
    both = pd.concat([tr.assign(src="train"), va.assign(src="valid")])
    mon = both.groupby(both["date"].dt.to_period("M")).agg(
        src=("src", "first"), mean_quote_signal=("quote_signal", "mean"),
        mean_market_index=("market_index", "mean"), median_rpm=("rpm", "median"))
    print(mon.round(3).to_string())
    if "market_index" in tr:
        print("\nmarket_index within-date std (0 => date-level signal):",
              round(tr.groupby("date")["market_index"].std().median(), 4))

    section("9. Weight")
    print("negative weights:", (tr["weight"] < 0).sum(), "| at +/-47,500 cap:", (tr["weight"].abs() == 47500).sum(),
          "| missing:", tr["weight"].isna().sum())

    section("10. Coordinate sanity")
    for side in ("pickup", "delivery"):
        g = tr.groupby(side)[[f"{side}_lat", f"{side}_lon"]].nunique()
        print(f"{side}: cities={len(g)}, cities with >1 distinct lat/lon = {(g.max(axis=1) > 1).sum()}")
    hv = haversine_miles(tr.pickup_lat, tr.pickup_lon, tr.delivery_lat, tr.delivery_lon)
    ratio = (tr["distance"] / hv.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    print("distance / haversine ratio quantiles:\n", ratio.quantile([.05, .5, .95]).round(2).to_string())
    print("# if the ratio is far from ~1.1-1.3, the lat/lon are not real-world coordinates; rely on `distance`.")

    lane = tr.groupby(["pickup", "delivery"])["distance"].agg(["count", "std"])
    print("\nlanes:", len(lane), "| median distance std within lane:", round(lane["std"].median(), 3))

    section("11. Category drift (validation vs train)")
    for c in ("pickup", "delivery", "equipment"):
        unseen = set(va[c].dropna().unique()) - set(tr[c].dropna().unique())
        print(f"{c}: unseen in train = {sorted(unseen)[:10]} ({len(unseen)})")
    unseen_rows = (va["pickup"].isin(set(va["pickup"]) - set(tr["pickup"])) |
                   va["delivery"].isin(set(va["delivery"]) - set(tr["delivery"]))).mean()
    print(f"share of validation rows touching a city unseen in train: {unseen_rows:.1%}")
    print("weight range of the train:", tr["weight"].min(), tr["weight"].max(), "| valid:", va["weight"].min(), va["weight"].max())
    print("distance range of the train:", tr["distance"].min(), tr["distance"].max(), "| valid:", va["distance"].min(), va["distance"].max())

    section("12. December inputs")
    print(dec.head(3).to_string(index=False))
    print("Does Training data covers December?", (tr["date"].dt.month == 12).any())

    # Plots(matplotlib)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    tr["rpm"].clip(upper=tr["rpm"].quantile(0.999)).hist(bins=80, ax=ax[0])
    ax[0].set_title("Rate per mile")
    tr.groupby(tr["date"].dt.to_period("M"))["rpm"].median().plot(ax=ax[1], marker="o")
    ax[1].set_title("Median rpm by month")
    if "market_index" in tr:
        tr.groupby("date")["market_index"].mean().plot(ax=ax[2])
        ax[2].set_title("market_index over time")
    fig.tight_layout()
    fig.savefig(out / "eda_overview.png", dpi=130)
    print(f"\nSaved {out / 'eda_overview.png'}")


if __name__ == "__main__":
    main()