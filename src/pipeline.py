"""
Freight Rate Prediction Challenge -- end-to-end solution.

Reproduces both submission files from the raw data:
    validation_predictions.csv
    data/december_chart_inputs.csv   (predicted_rate column filled)

Usage:
    python src/pipeline.py --data-dir data --out-dir .
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

ORIGIN = pd.Timestamp("2025-01-01")
OUTLIER_THRESHOLD = 0.5      # chosen inside an empty band in the residual histogram


# --------------------------------------------------------------------------- #
# data quality
# --------------------------------------------------------------------------- #
def repair(train: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """Repair sign-flipped weights, impute missing values. Returns daily market index."""
    for df in (train, valid):
        df["weight_missing"] = df["weight"].isna().astype(int)
        df["weight"] = df["weight"].abs()          # 292 sign-flipped rows

    # market_index is a daily signal (within-day sd 0.025 vs annual range 0.68-1.47).
    # Built from feature columns only -- no target values -- and from every row that
    # will need a prediction, which is exactly the information available at predict time.
    mkt_daily = (
        pd.concat([train[["date", "market_index"]], valid[["date", "market_index"]]])
        .dropna()
        .groupby("date")["market_index"]
        .mean()
    )
    for df in (train, valid):
        df["market_missing"] = df["market_index"].isna().astype(int)
        df["market_index_daily"] = df["date"].map(mkt_daily)
        df["market_index"] = df["market_index"].fillna(df["market_index_daily"])

    wmed = train.groupby("equipment")["weight"].median()
    for df in (train, valid):
        df["weight"] = df["weight"].fillna(df["equipment"].map(wmed))

    return mkt_daily


def flag_outliers(frame: pd.DataFrame) -> np.ndarray:
    """Flag multiplicatively corrupted targets via a robust log-log fit."""
    ld = np.log(frame["distance"].values)
    lr = np.log(frame["posted_rate"].values)
    keep = np.ones(len(frame), bool)
    for _ in range(5):
        coef = np.polyfit(ld[keep], lr[keep], 1)
        res = lr - np.polyval(coef, ld)
        mad = 1.4826 * np.median(np.abs(res - np.median(res)))
        keep = np.abs(res - np.median(res)) < 5 * mad
    return np.abs(res) > OUTLIER_THRESHOLD


def city_coordinates(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """One lat/lon per city -- coordinates are a pure lookup, with zero within-city variance."""
    pick = pd.concat([train[["pickup", "pickup_lat", "pickup_lon"]],
                      valid[["pickup", "pickup_lat", "pickup_lon"]]]).rename(
        columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"})
    drop = pd.concat([train[["delivery", "delivery_lat", "delivery_lon"]],
                      valid[["delivery", "delivery_lat", "delivery_lon"]]]).rename(
        columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"})
    return pd.concat([pick, drop]).drop_duplicates().set_index("city")


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def add_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    qstart = df["date"].dt.to_period("Q").dt.start_time
    df["doq"] = (df["date"] - qstart).dt.days       # 0..91, position within quarter
    df["quarter"] = df["date"].dt.quarter
    df["t"] = (df["date"] - ORIGIN).dt.days         # global linear trend
    df["dow"] = df["date"].dt.dayofweek
    return df


def make_X_lin(df: pd.DataFrame) -> pd.DataFrame:
    """Linear stage: everything that must extrapolate past the training window."""
    x = df["doq"].values / 91.0
    return pd.DataFrame({
        "log_distance": np.log(df["distance"]),
        "log_weight": np.log(df["weight"]),
        "log_market": np.log(df["market_index_daily"]),
        "reefer": (df["equipment"] == "Reefer").astype(float),
        "flatbed": (df["equipment"] == "Flatbed").astype(float),
        "pickup_lat": df["pickup_lat"], "pickup_lon": df["pickup_lon"],
        "delivery_lat": df["delivery_lat"], "delivery_lon": df["delivery_lon"],
        "trend": df["t"] / 365.0,                   # level drift across quarters
        "doq1": x, "doq2": x ** 2, "doq3": x ** 3,  # within-quarter ramp
    }, index=df.index)


def make_X_tree(df: pd.DataFrame) -> pd.DataFrame:
    """Tree stage: local non-linearity and interactions. No trend term, by design."""
    return pd.DataFrame({
        "log_distance": np.log(df["distance"]),
        "log_weight": np.log(df["weight"]),
        "log_market": np.log(df["market_index_daily"]),
        "eq": df["equipment"].map({"Dry Van": 0, "Reefer": 1, "Flatbed": 2}).astype(float),
        "doq": df["doq"].astype(float),
        "dow": df["dow"].astype(float),
        "pickup_lat": df["pickup_lat"], "pickup_lon": df["pickup_lon"],
        "delivery_lat": df["delivery_lat"], "delivery_lon": df["delivery_lon"],
    }, index=df.index)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class HybridRateModel:
    """OLS in log space, plus gradient boosting on its residual.

    Sum is taken in log space, so the tree contributes a multiplier on the
    dollar scale and predictions are positive by construction.
    """

    def fit(self, frame: pd.DataFrame) -> "HybridRateModel":
        y = np.log(frame["posted_rate"])
        self.lin_ = LinearRegression().fit(make_X_lin(frame), y)
        residual = y - self.lin_.predict(make_X_lin(frame))
        self.gbm_ = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0,
        ).fit(make_X_tree(frame), residual)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.lin_.predict(make_X_lin(frame))
                      + self.gbm_.predict(make_X_tree(frame)))


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    data, out = Path(args.data_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data / "train_test.csv", parse_dates=["date"])
    valid = pd.read_csv(data / "validation.csv", parse_dates=["date"])
    template = pd.read_csv(data / "validation_predictions_template.csv")

    mkt_daily = repair(train, valid)
    coords = city_coordinates(train, valid)
    train["is_outlier"] = flag_outliers(train)
    train, valid = add_time(train), add_time(valid)

    fit_rows = train.loc[~train["is_outlier"]]
    print(f"training on {len(fit_rows):,} of {len(train):,} rows "
          f"({train['is_outlier'].sum()} corrupted targets excluded)")
    model = HybridRateModel().fit(fit_rows)

    # ---- 12,000 validation predictions --------------------------------------
    valid["predicted_rate"] = model.predict(valid)
    preds = (template[["load_id"]]
             .merge(valid[["load_id", "predicted_rate"]], on="load_id", how="left"))
    preds["predicted_rate"] = preds["predicted_rate"].round(2)
    assert list(preds.columns) == ["load_id", "predicted_rate"]
    assert len(preds) == 12_000
    assert preds["predicted_rate"].notna().all() and (preds["predicted_rate"] > 0).all()
    preds.to_csv(out / "validation_predictions.csv", index=False)
    print(f"wrote validation_predictions.csv  "
          f"(${preds.predicted_rate.min():.0f}-${preds.predicted_rate.max():.0f})")

    # ---- 31 fixed December rows ---------------------------------------------
    dec_path = data / "december_chart_inputs.csv"
    dec_raw = pd.read_csv(dec_path, dtype=str)      # strings preserve the scorer's checks
    dec = dec_raw.copy()
    dec["date"] = pd.to_datetime(dec_raw["date"])
    dec["distance"] = dec_raw["distance"].astype(float)
    dec["weight"] = dec_raw["weight"].astype(float)
    dec["pickup_lat"] = coords.loc[dec["pickup"], "lat"].values
    dec["pickup_lon"] = coords.loc[dec["pickup"], "lon"].values
    dec["delivery_lat"] = coords.loc[dec["delivery"], "lat"].values
    dec["delivery_lon"] = coords.loc[dec["delivery"], "lon"].values
    dec["market_index_daily"] = dec["date"].map(mkt_daily)
    assert dec["market_index_daily"].notna().all(), "no market index for some December date"
    dec = add_time(dec)

    december = model.predict(dec)
    filled = dec_raw.copy()                         # original 7 columns, original order
    filled["predicted_rate"] = np.round(december, 2)
    filled.to_csv(dec_path, index=False)
    print(f"filled {dec_path.name}  "
          f"Dec 1 ${december[0]:.2f} -> Dec 31 ${december[-1]:.2f} "
          f"({december[-1] / december[0] - 1:+.1%})")


if __name__ == "__main__":
    main()
