"""Freight rate pipeline: clean -> features -> time-based CV -> fit -> predict.

Usage:
    python train_predict.py [--data-dir data] [--out-dir .] [--cv-folds 3]

Outputs (in --out-dir):
    validation_predictions.csv     load_id,predicted_rate   (12,000 rows)
    december_predictions.csv       december_chart_inputs.csv with predicted_rate filled
    cv_results.csv                 per-fold metrics for every model / feature set
    run_summary.json               choices made (feature set, model, cleaning counts)
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SEED = 42
CATS = ["equipment", "pickup", "delivery"]
MARKET = ["market_index", "quote_signal"]
TARGET = "posted_rate"
# base = what the December input provides; quote/full add validation-only signals
# "weekly": only weekday/day-of-month calendar features (safe to extrapolate to unseen months);
# "full" adds month/week/dayofyear, which trees cannot extrapolate to months never seen in training.
CALENDAR = "weekly"
VARIANTS = {"base": [], "quote": ["quote_signal"], "full": MARKET}


# --------------------------------------------------------------------------- io
def find_data_dir(arg: str | None) -> Path:
    """The README says `data/` but the repo folder is `Data/`; Linux is case-sensitive."""
    for cand in ([arg] if arg else []) + ["data", "Data"]:
        if cand and (Path(cand) / "train_test.csv").is_file():
            return Path(cand)
    raise SystemExit("Could not find train_test.csv. Pass --data-dir.")


# ----------------------------------------------------------------------- cleaning
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type fixes that are safe for train AND validation/december (no row dropping)."""
    df = df.copy()
    for c in CATS:
        df[c] = df[c].astype("string").str.strip().str.title()
        if c == "equipment":  # "dry van", "DRY VAN " -> "Dry Van"
            df[c] = df[c].str.replace(r"\s+", " ", regex=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["distance", "weight", *MARKET, TARGET]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["weight"] = df["weight"].abs()  # negative weights are sign flips (min was -47,500)
    return df


def clean_train(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Row-dropping cleaning, applied to training data only."""
    log = {"rows_in": len(raw), "negative_weights_flipped": int((pd.to_numeric(raw["weight"], errors="coerce") < 0).sum())}
    df = basic_clean(raw)
    df = df.drop_duplicates(subset="load_id")
    log["after_dedupe_load_id"] = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "load_id"])
    log["after_dedupe_content"] = len(df)

    bad = df["date"].isna() | ~(df[TARGET] > 0) | ~(df["distance"] > 0)
    log["dropped_invalid_date_rate_distance"] = int(bad.sum())
    df = df[~bad].copy()

    # weight: impute by equipment median (impossible values -> NaN first)
    df.loc[~df["weight"].between(1_000, 60_000), "weight"] = np.nan
    log["weight_imputed"] = int(df["weight"].isna().sum())
    df["weight"] = df["weight"].fillna(df.groupby("equipment")["weight"].transform("median"))
    df["weight"] = df["weight"].fillna(df["weight"].median())

    # rate-per-mile outliers: robust z-score of log(rpm) within equipment x distance decile
    # (rpm falls with distance, so comparing within a distance band avoids flagging short hauls)
    log_rpm = np.log(df[TARGET] / df["distance"])
    band = pd.qcut(df["distance"], 10, labels=False, duplicates="drop")
    grp = [df["equipment"], band]
    med = log_rpm.groupby(grp).transform("median")
    mad = (log_rpm - med).abs().groupby(grp).transform("median") * 1.4826
    z = (log_rpm - med) / mad.replace(0, np.nan)
    out = z.abs() > 6
    log["dropped_rpm_outliers"] = int(out.sum())
    df = df[~out].copy()
    log["rows_out"] = len(df)
    return df.reset_index(drop=True), log


# ---------------------------------------------------------------------- features
def build_city_coords(*frames: pd.DataFrame) -> dict:
    """City -> median coordinates (absent from the December input). Coordinates are features, not labels,
    so the final model may borrow them from validation rows (covers cities unseen in training)."""
    df = pd.concat([f[["pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]] for f in frames])
    p = df.groupby("pickup")[["pickup_lat", "pickup_lon"]].median()
    d = df.groupby("delivery")[["delivery_lat", "delivery_lon"]].median()
    return {"pickup": p, "delivery": d}


def make_features(df: pd.DataFrame, coords: dict, variant: str) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in CATS:
        X[c] = df[c].astype(object)
    X["distance"] = df["distance"]
    X["log_distance"] = np.log(df["distance"])
    X["weight"] = df["weight"]
    d = df["date"]
    if CALENDAR == "full":
        X["month"] = d.dt.month
        X["week"] = d.dt.isocalendar().week.astype(float)
        X["dayofyear"] = d.dt.dayofyear
    X["dow"] = d.dt.dayofweek
    X["dom"] = d.dt.day
    X["pickup_lat"] = df["pickup"].map(coords["pickup"]["pickup_lat"])
    X["pickup_lon"] = df["pickup"].map(coords["pickup"]["pickup_lon"])
    X["delivery_lat"] = df["delivery"].map(coords["delivery"]["delivery_lat"])
    X["delivery_lon"] = df["delivery"].map(coords["delivery"]["delivery_lon"])
    cols = VARIANTS[variant]
    if "quote_signal" in cols:
        X["quote_signal"] = df["quote_signal"]
        X["implied_rate"] = df["quote_signal"] * df["distance"]  # quote_signal ~ rate/mile
    if "market_index" in cols:
        X["market_index"] = df["market_index"]
    return X


# ------------------------------------------------------------------------ models
class NaiveRpm:
    """Baseline: median log(rate/mile) by equipment x distance decile."""

    def fit(self, X, y):
        self.edges_ = np.unique(np.quantile(X["distance"], np.linspace(0, 1, 11)))
        key = self._key(X)
        self.table_ = pd.Series(y.values).groupby(key.values).median()
        self.default_ = float(np.median(y))
        return self

    def _key(self, X):
        b = np.clip(np.digitize(X["distance"], self.edges_[1:-1]), 0, None)
        return X["equipment"].astype(str) + "|" + pd.Series(b, index=X.index).astype(str)

    def predict(self, X):
        return self._key(X).map(self.table_).fillna(self.default_).values


def make_model(name: str, X: pd.DataFrame):
    num_cols = [c for c in X.columns if c not in CATS]
    if name == "naive_rpm":
        return NaiveRpm()
    if name == "hybrid":
        return Hybrid()
    if name == "ridge":
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CATS),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ])
        return Pipeline([("pre", pre), ("reg", Ridge(alpha=1.0))])
    cfgs = {
        "hgb_small": dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40),
        "hgb_medium": dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0),
        "hgb_large": dict(learning_rate=0.03, max_iter=800, max_leaf_nodes=63, min_samples_leaf=30, l2_regularization=1.0),
    }
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                         encoded_missing_value=np.nan, min_frequency=10, max_categories=200)
    pre = ColumnTransformer([("cat", enc, CATS)], remainder="passthrough", verbose_feature_names_out=False)
    reg = HistGradientBoostingRegressor(categorical_features=[0, 1, 2], random_state=SEED, **cfgs[name])
    return Pipeline([("pre", pre), ("reg", reg)])


class Hybrid:
    """Ridge captures the linear signal level (extrapolates under drift); a gradient-boosted model then
    learns the residual structure (lane, equipment, weekday, ...) with the ridge output as an extra feature."""

    def fit(self, X, y):
        self.ridge_ = make_model("ridge", X).fit(X, y)
        r = self.ridge_.predict(X)
        X2 = X.assign(ridge_pred=r)
        self.hgb_ = make_model("hgb_medium", X2).fit(X2, y - r)
        return self

    def predict(self, X):
        r = self.ridge_.predict(X)
        return r + self.hgb_.predict(X.assign(ridge_pred=r))


class TwoStage:
    """Regime-aware use of quote_signal.

    quote_signal tracks the true rate per mile in some periods and is degraded in others. Stage 1 is a
    base model that never sees quote_signal. Stage 2 (gradient boosting on the stage-1 residual) sees
    quote_signal plus how well it agrees with stage 1 for that row and for that week's loads. The weekly
    agreement is computed from features only, so it also works on validation weeks that have no labels."""

    AGREE_TOL = 0.10  # |log(quote_signal) - log(stage-1 rpm)| below this counts as agreement

    def __init__(self, coords: dict, alpha: float = 1.0):
        self.coords = coords
        self.alpha = alpha  # 1 = full stage-2 correction; <1 shrinks toward the safe base model

    def _stage2_X(self, df: pd.DataFrame, s1: np.ndarray) -> pd.DataFrame:
        X = make_features(df, self.coords, "quote")
        s1 = pd.Series(s1, index=df.index)
        lr = np.log(df["quote_signal"]) - s1
        week = df["date"].dt.to_period("W")
        X["s1"], X["lr"], X["abs_lr"] = s1, lr, lr.abs()
        X["week_agree"] = (lr.abs() < self.AGREE_TOL).astype(float).groupby(week).transform("mean")
        X["week_med_lr"] = lr.groupby(week).transform("median")
        return X

    def fit(self, df: pd.DataFrame, y_log: pd.Series):
        Xb = make_features(df, self.coords, "base")
        self.base_ = make_model("ridge", Xb).fit(Xb, y_log)
        s1 = self.base_.predict(Xb)
        X2 = self._stage2_X(df, s1)
        self.hgb_ = make_model("hgb_small", X2).fit(X2, y_log - s1)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        s1 = self.base_.predict(make_features(df, self.coords, "base"))
        return s1 + self.alpha * self.hgb_.predict(self._stage2_X(df, s1))


MODELS = ["naive_rpm", "ridge", "hgb_small", "hgb_medium", "hgb_large", "hybrid"]


def to_dollars(log_rpm_pred: np.ndarray, distance: pd.Series) -> np.ndarray:
    """Model predicts log(rate per mile); convert back to a dollar rate."""
    return np.exp(log_rpm_pred) * distance.values


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {"MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(np.mean(err**2)),
            "MAPE%": 100 * np.mean(np.abs(err) / y_true)}


# --------------------------------------------------------------------- validation
def make_folds(df: pd.DataFrame, n_folds: int, gap: int, mode: str = "forward"):
    """Expanding-window time folds. Test on each of the last `n_folds` months; train only on months
    that end at least `gap` months before the test month (gap=1 mimics predicting December from Oct data)."""
    months = df["date"].dt.to_period("M")
    uniq = sorted(months.unique())
    if mode == "lomo":  # leave-one-month-out: regime-matched check, NOT forward-looking
        for m in uniq:
            yield str(m), np.where(months != m)[0], np.where(months == m)[0]
    elif len(uniq) >= n_folds + gap + 2:
        for m in uniq[-n_folds:]:
            tr_idx = np.where(months < m - gap)[0]
            if len(tr_idx):
                yield str(m), tr_idx, np.where(months == m)[0]
    else:  # not enough months: fall back to random K-fold (and say so)
        from sklearn.model_selection import KFold
        print("WARNING: too few months of data; using random KFold.")
        for i, (a, b) in enumerate(KFold(n_folds, shuffle=True, random_state=SEED).split(df)):
            yield f"kfold{i}", a, b


TWO_STAGE = ["twostage", "twostage_shrunk"]


def two_stage(name: str, coords: dict) -> TwoStage:
    return TwoStage(coords, alpha=1.0 if name == "twostage" else 0.5)


def cross_validate(df: pd.DataFrame, variant: str, n_folds: int, gap: int,
                   mode: str = "forward", models: list[str] | None = None) -> pd.DataFrame:
    rows = []
    y_log = np.log(df[TARGET] / df["distance"])
    for fold, tr_idx, te_idx in make_folds(df, n_folds, gap, mode):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        fc = build_city_coords(tr)  # coordinates from the fold's training rows only
        Xtr, Xte = make_features(tr, fc, variant), make_features(te, fc, variant)
        names = ["ridge", "hgb_small", "hybrid"] if mode == "lomo" else MODELS
        ts = None
        for name in names + (TWO_STAGE if variant == "quote" else []):
            if models and name not in models:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if name in TWO_STAGE:
                    if ts is None:  # stage 1 + 2 fitted once; the shrunk variant only changes alpha at predict time
                        ts = TwoStage(fc).fit(tr, y_log.iloc[tr_idx])
                    ts.alpha = two_stage(name, fc).alpha
                    pred = to_dollars(ts.predict(te), te["distance"])
                else:
                    m = make_model(name, Xtr).fit(Xtr, y_log.iloc[tr_idx])
                    pred = to_dollars(m.predict(Xte), te["distance"])
            rows.append({"features": variant, "model": name, "fold": fold,
                         "n_train": len(tr), "n_test": len(te), **metrics(te[TARGET].values, pred)})
    return pd.DataFrame(rows)


def best_model(cv: pd.DataFrame, features: list[str]) -> tuple[str, str]:
    sub = cv[cv["features"].isin(features) & (cv["model"] != "naive_rpm")]
    return sub.groupby(["features", "model"])["MAE"].mean().idxmin()


def fit_final(df: pd.DataFrame, coords: dict, variant: str, name: str):
    y = np.log(df[TARGET] / df["distance"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name in TWO_STAGE:
            return two_stage(name, coords).fit(df, y)
        X = make_features(df, coords, variant)
        return make_model(name, X).fit(X, y)


def predict_rate(model, df: pd.DataFrame, coords: dict, variant: str, name: str) -> np.ndarray:
    log_rpm = model.predict(df) if name in TWO_STAGE else model.predict(make_features(df, coords, variant))
    return np.maximum(to_dollars(log_rpm, df["distance"]), 1.0)


def regime_table(train: pd.DataFrame, val: pd.DataFrame, coords: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How often quote_signal agrees with a base-model estimate, per month and per week. In train we can
    compare with the truth (share within 10% of the real rate); for validation only agreement exists."""
    Xb = make_features(train, coords, "base")
    base = make_model("ridge", Xb).fit(Xb, np.log(train[TARGET] / train["distance"]))
    parts = []
    for src, df in (("train", train), ("valid", val)):
        lr = np.log(df["quote_signal"]) - base.predict(make_features(df, coords, "base"))
        t = pd.DataFrame({"src": src, "agree": (lr.abs() < TwoStage.AGREE_TOL).astype(float).values,
                          "month": df["date"].dt.to_period("M").astype(str).values,
                          "week": df["date"].dt.to_period("W").dt.start_time.values})
        t["truly_good"] = (((df["quote_signal"] / (df[TARGET] / df["distance"]) - 1).abs() <= 0.10).astype(float).values
                           if src == "train" else np.nan)
        parts.append(t)
    t = pd.concat(parts)
    monthly = t.groupby("month").agg(src=("src", "first"), agree=("agree", "mean"), truly_good=("truly_good", "mean")).round(3)
    weekly = t[t["week"] >= "2025-06-30"].groupby("week").agg(src=("src", "first"), agree=("agree", "mean")).round(2)
    return monthly, weekly


# -------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--cv-folds", type=int, default=3)
    ap.add_argument("--gap-months", type=int, default=1, help="months skipped between train and CV test month")
    ap.add_argument("--val-features", default="auto", choices=["auto", "base", "quote", "full"],
                    help="force the feature set for validation predictions (auto = best CV MAE)")
    ap.add_argument("--cv-mode", default="lomo", choices=["lomo", "forward"],
                    help="lomo = leave-one-month-out (default; lets the half-degraded August regime be tested); "
                         "forward = expanding-window time CV with --gap-months")
    ap.add_argument("--models", default=None, help="comma-separated subset of models to evaluate")
    ap.add_argument("--variants", default=None, help="comma-separated subset of feature sets, e.g. base,quote")
    ap.add_argument("--regime-tol", type=float, default=0.15,
                    help="select models on CV months whose quote_signal agreement is within this of validation's")
    ap.add_argument("--calendar", default="weekly", choices=["weekly", "full"],
                    help="weekly = weekday/day-of-month only (default); full = add month/week/dayofyear")
    args = ap.parse_args()
    global CALENDAR
    CALENDAR = args.calendar
    data_dir, out = find_data_dir(args.data_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train, clean_log = clean_train(pd.read_csv(data_dir / "train_test.csv"))
    val = basic_clean(pd.read_csv(data_dir / "validation.csv"))
    template = pd.read_csv(data_dir / "validation_predictions_template.csv")
    dec_raw = pd.read_csv(data_dir / "december_chart_inputs.csv")
    dec = basic_clean(dec_raw)
    print("Cleaning:", clean_log)

    # Validation rows are never dropped: repair instead.
    val["weight"] = val["weight"].fillna(val["equipment"].map(train.groupby("equipment")["weight"].median()))
    val["weight"] = val["weight"].fillna(train["weight"].median())
    dec["weight"] = dec["weight"].fillna(train["weight"].median())

    usable = [v for v, cols in VARIANTS.items()
              if all(c in val and c in train and val[c].notna().mean() > 0.9 for c in cols)]
    if args.variants:
        usable = [v for v in usable if v in args.variants.split(",")]
    elif args.cv_mode == "lomo":
        usable = [v for v in usable if v != "full"]  # market_index did not help; keep lomo fast
    model_filter = args.models.split(",") if args.models else None

    coords = build_city_coords(train, val)  # borrow coordinates for cities unseen in training
    monthly, weekly = regime_table(train, val, coords)
    val_agree = float(monthly.loc[monthly["src"] == "valid", "agree"].mean())
    print("\nquote_signal regime check (agree = matches a base-model estimate within 10%):")
    print(monthly.to_string())
    print("\nWeekly agreement since July:")
    print(weekly.T.to_string())
    print(f"\nMean validation agreement: {val_agree:.2f}")
    print("\nFeature sets used:", usable, "| CV mode:", args.cv_mode)

    cv = pd.concat([cross_validate(train, v, args.cv_folds, args.gap_months, args.cv_mode, model_filter)
                    for v in usable], ignore_index=True)
    cv.to_csv(out / "cv_results.csv", index=False)
    pd.set_option("display.width", 120)
    summary = cv.groupby(["features", "model"])[["MAE", "RMSE", "MAPE%"]].mean().round(3)
    print(f"\nCV mean over {cv['fold'].nunique()} folds ({args.cv_mode}, gap={args.gap_months}):\n", summary)
    best_fold = cv.pivot_table(index=["features", "model"], columns="fold", values="MAE").round(1)
    print("\nMAE by test month:\n", best_fold)

    # Select on the CV months whose quote_signal regime resembles validation's (fallback: all months)
    cv["agree"] = cv["fold"].map(monthly["agree"])
    matched = cv[(cv["agree"] - val_agree).abs() <= args.regime_tol]
    sel = matched if len(matched) else cv
    print(f"\nSelection folds (agreement within {args.regime_tol} of {val_agree:.2f}): {sorted(sel['fold'].unique())}")
    print("Mean MAE on selection folds:\n", sel.groupby(["features", "model"])["MAE"].mean().round(2).to_string())
    cand = usable if args.val_features == "auto" else [args.val_features]
    feat_val, name_val = best_model(sel, cand)
    base_cv = sel[(sel["features"] == "base") & (sel["model"] != "naive_rpm")]
    name_dec = base_cv.groupby("model")["MAE"].mean().idxmin() if len(base_cv) else "ridge"
    print(f"\nValidation predictions: {name_val} on '{feat_val}' features")
    print(f"December predictions:   {name_dec} on 'base' features (the December input has only 7 columns)")

    m_val = fit_final(train, coords, feat_val, name_val)
    pred_val = predict_rate(m_val, val, coords, feat_val, name_val)
    pv = pd.Series(pred_val, index=val["load_id"].values)
    sub = template[["load_id"]].copy()
    sub["predicted_rate"] = sub["load_id"].map(pv).round(2)
    assert sub["predicted_rate"].notna().all() and len(sub) == len(template), "ID mismatch with template"
    sub.to_csv(out / "validation_predictions.csv", index=False)

    m_dec = m_val if (feat_val == "base" and name_val == name_dec) else fit_final(train, coords, "base", name_dec)
    pred_dec = predict_rate(m_dec, dec, coords, "base", name_dec)
    dec_out = dec_raw.copy()
    dec_out["predicted_rate"] = np.round(pred_dec, 2)
    dec_out.to_csv(out / "december_predictions.csv", index=False)

    (out / "run_summary.json").write_text(json.dumps({
        "cleaning": clean_log, "calendar": CALENDAR, "cv_mode": args.cv_mode, "validation_agreement": val_agree,
        "selection_folds": sorted(sel["fold"].unique()), "usable_feature_sets": usable, "cv_gap_months": args.gap_months,
        "validation_features": feat_val, "validation_model": name_val, "december_model": name_dec,
        "december_in_training_data": bool((train["date"].dt.month == 12).any()),
        "cv_mean": summary.reset_index().to_dict(orient="records"),
    }, indent=2, default=float))
    print(f"\nWrote {out/'validation_predictions.csv'} ({len(sub):,} rows) and {out/'december_predictions.csv'}")
    print("Next: python score.py --predictions validation_predictions.csv "
          "--december-predictions december_predictions.csv")


if __name__ == "__main__":
    main()