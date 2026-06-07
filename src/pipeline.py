import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENERGY_PATH = PROJECT_ROOT / "data/raw/energy_dataset.csv"
WEATHER_PATH = PROJECT_ROOT / "data/raw/weather_features.csv"
PROCESSED_PATH = PROJECT_ROOT / "data/processed/spain_hourly.csv"
METRICS_PATH = PROJECT_ROOT / "data/processed/pipeline_metrics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "data/processed/pipeline_predictions.csv"


def load_raw(energy_path=ENERGY_PATH, weather_path=WEATHER_PATH):
    energy = pd.read_csv(energy_path)
    energy["time"] = pd.to_datetime(energy["time"], utc=True)
    energy = energy.sort_values("time").reset_index(drop=True)

    weather = pd.read_csv(weather_path)
    mask = weather["city_name"] == "Madrid"
    weather_madrid = weather.loc[mask].copy()
    weather_madrid["time"] = pd.to_datetime(weather_madrid["dt_iso"], utc=True)
    keep = ["time", "temp", "humidity", "wind_speed", "clouds_all"]
    weather_madrid = weather_madrid[keep].drop_duplicates(subset="time")

    return energy.merge(weather_madrid, on="time", how="left")


def prepare_hourly(df):
    drop_cols = ["generation hydro pumped storage aggregated", "forecast wind offshore eday ahead"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
    df["time"] = df["time"].dt.tz_convert("Europe/Madrid").dt.tz_localize(None)

    hourly = df.rename(columns={"time": "ds", "total load actual": "y"})
    hourly["unique_id"] = "spain_load"

    keep_cols = [
        "unique_id", "ds", "y",
        "generation solar", "generation wind onshore", "generation nuclear",
        "generation fossil gas", "total load forecast", "price actual",
        "temp", "humidity", "wind_speed", "clouds_all",
    ]
    return hourly[keep_cols].reset_index(drop=True)


def add_features(df):
    out = df.copy()
    out["hour"] = out["ds"].dt.hour
    out["dayofweek"] = out["ds"].dt.dayofweek
    out["month"] = out["ds"].dt.month
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)

    for lag in [1, 24, 48, 168]:
        out[f"lag_{lag}"] = out["y"].shift(lag)

    out["rolling_24_mean"] = out["y"].shift(1).rolling(24).mean()
    out["rolling_168_mean"] = out["y"].shift(1).rolling(168).mean()
    out = out.dropna()
    return pd.DataFrame(out.reset_index(drop=True))


def train_test_split_time(df: pd.DataFrame, test_hours: int = 24 * 14):
    split_date = df["ds"].max() - pd.Timedelta(hours=test_hours - 1)
    train = pd.DataFrame(df.loc[df["ds"] < split_date])
    test = pd.DataFrame(df.loc[df["ds"] >= split_date])
    return train, test


def metric_dict(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def compare_absolute_errors(y_true, baseline_pred, candidate_pred, n_bootstrap=2000):
    baseline_error = np.abs(y_true - baseline_pred)
    candidate_error = np.abs(y_true - candidate_pred)
    diff = baseline_error - candidate_error

    np.random.seed(42)
    boot_means = [np.mean(np.random.choice(diff, size=len(diff), replace=True)) for _ in range(n_bootstrap)]

    t_stat, t_pvalue = stats.ttest_rel(baseline_error, candidate_error)
    try:
        wres = stats.wilcoxon(baseline_error, candidate_error)
        wilcoxon_stat = float(wres[0])  # type: ignore[arg-type]
        wilcoxon_pvalue = float(wres[1])  # type: ignore[arg-type]
    except ValueError:
        wilcoxon_stat, wilcoxon_pvalue = np.nan, np.nan

    perc = np.percentile(boot_means, [2.5, 97.5]).tolist()
    ci_low, ci_high = perc[0], perc[1]
    return {
        "mean_absolute_error_improvement": float(np.mean(diff)),
        "paired_t_stat": float(t_stat),
        "paired_t_pvalue": float(t_pvalue),
        "wilcoxon_stat": float(wilcoxon_stat),
        "wilcoxon_pvalue": float(wilcoxon_pvalue),
        "bootstrap_ci_95_low": float(ci_low),
        "bootstrap_ci_95_high": float(ci_high),
    }


def run_pipeline(energy_path=ENERGY_PATH, weather_path=WEATHER_PATH):
    started = time.perf_counter()
    raw = load_raw(energy_path, weather_path)
    hourly = prepare_hourly(raw)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(PROCESSED_PATH, index=False)

    supervised = add_features(hourly)
    train, test = train_test_split_time(supervised)

    feature_cols = [
        "generation solar", "generation wind onshore", "generation nuclear",
        "generation fossil gas", "total load forecast", "price actual",
        "temp", "humidity", "wind_speed", "clouds_all",
        "hour", "dayofweek", "month", "is_weekend",
        "lag_1", "lag_24", "lag_48", "lag_168",
        "rolling_24_mean", "rolling_168_mean",
    ]

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        ),
    }

    predictions = test[["unique_id", "ds", "y"]].copy()
    metrics = {
        "dataset_rows_raw": int(len(raw)),
        "dataset_rows_hourly": int(len(hourly)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "target": "Spain total load actual, hourly",
    }

    # Сезонный наивный прогноз: берём значение сутки назад
    predictions["seasonal_naive_24"] = test["lag_24"].to_numpy()
    metrics["seasonal_naive_24"] = metric_dict(test["y"].to_numpy(), predictions["seasonal_naive_24"])

    for name, model in models.items():
        fit_started = time.perf_counter()
        model.fit(train[feature_cols], train["y"])
        pred = model.predict(test[feature_cols])
        predictions[name] = pred
        metrics[name] = metric_dict(test["y"].to_numpy(), pred)
        metrics[name]["fit_predict_seconds"] = float(time.perf_counter() - fit_started)

    y_true = test["y"].to_numpy()
    metrics["statistical_tests"] = {
        "random_forest_vs_seasonal_naive_24": compare_absolute_errors(
            y_true,
            predictions["seasonal_naive_24"],
            predictions["random_forest"],
        ),
        "random_forest_vs_linear_regression": compare_absolute_errors(
            y_true,
            predictions["linear_regression"],
            predictions["random_forest"],
        ),
    }

    metrics["total_seconds"] = float(time.perf_counter() - started)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline for Spain electricity load forecasting")
    parser.add_argument("--energy-path", type=Path, default=ENERGY_PATH)
    parser.add_argument("--weather-path", type=Path, default=WEATHER_PATH)
    args = parser.parse_args()
    metrics = run_pipeline(args.energy_path, args.weather_path)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
