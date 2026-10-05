"""
Big Data Analytics — Mini Project
Food Delivery Data Analytics — predictive layer.

Two Random Forest models trained on delivered orders:

  1. delivery-time regressor   — features: distance, hour, weekend, weather,
                                 traffic, restaurant prep time, cuisine, zone
  2. rating regressor          — features: delivery time, distance, weather,
                                 order value, promo usage, cuisine

Run directly to see model metrics and feature importances for the report.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from analysis import load_data

RANDOM_STATE = 42


# ------------------------------------------------------------------ helpers
def _encode(df):
    """One-hot encode categoricals (dropping first level)."""
    return pd.get_dummies(df, drop_first=True)


def _regression_report(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "R2": round(r2_score(y_true, y_pred), 3),
    }


# ------------------------------------------------------------------ delivery-time model
def build_delivery_time_model(orders, restaurants, seed=RANDOM_STATE):
    delivered = orders[orders["status"] == "Delivered"].merge(
        restaurants[["restaurant_id", "prep_time_min", "zone"]],
        on="restaurant_id", how="left")

    feature_cols = ["distance_km", "order_hour", "is_weekend",
                    "prep_time_min", "weather", "traffic_condition",
                    "cuisine", "zone"]
    X = delivered[feature_cols]
    y = delivered["delivery_time_min"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=seed)
    X_tr_enc = _encode(X_tr)
    X_te_enc = _encode(X_te).reindex(columns=X_tr_enc.columns, fill_value=0)

    lin = LinearRegression().fit(X_tr_enc, y_tr)
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                               n_jobs=-1, random_state=seed)
    rf.fit(X_tr_enc, y_tr)

    metrics = {"linear": _regression_report(y_te, lin.predict(X_te_enc)),
               "random_forest": _regression_report(y_te, rf.predict(X_te_enc))}

    importances = (pd.Series(rf.feature_importances_, index=X_tr_enc.columns)
                   .sort_values(ascending=False).head(15))

    return {"model": rf, "feature_columns": list(X_tr_enc.columns),
            "metrics": metrics, "importances": importances,
            "baseline_mean_mae": round(float(np.mean(np.abs(
                y_te - y_tr.mean()))), 2)}


# ------------------------------------------------------------------ rating model
def build_rating_model(orders, seed=RANDOM_STATE):
    delivered = orders[orders["status"] == "Delivered"]

    feature_cols = ["delivery_time_min", "distance_km", "weather",
                    "order_value", "promo_code", "cuisine", "order_hour",
                    "is_weekend"]
    X = delivered[feature_cols]
    y = delivered["delivery_rating"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=seed)
    X_tr_enc = _encode(X_tr)
    X_te_enc = _encode(X_te).reindex(columns=X_tr_enc.columns, fill_value=0)

    lin = LinearRegression().fit(X_tr_enc, y_tr)
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=10,
                               n_jobs=-1, random_state=seed)
    rf.fit(X_tr_enc, y_tr)

    metrics = {"linear": _regression_report(y_te, lin.predict(X_te_enc)),
               "random_forest": _regression_report(y_te, rf.predict(X_te_enc))}

    importances = (pd.Series(rf.feature_importances_, index=X_tr_enc.columns)
                   .sort_values(ascending=False).head(15))

    return {"model": rf, "feature_columns": list(X_tr_enc.columns),
            "metrics": metrics, "importances": importances}


# ------------------------------------------------------------------ console report
def main():
    orders, customers, restaurants = load_data()

    print("=" * 70)
    print("FOOD DELIVERY ANALYTICS — MODELING REPORT")
    print("=" * 70)

    print("\n-- Model 1: delivery-time prediction (delivered orders) --")
    dt = build_delivery_time_model(orders, restaurants)
    for name, m in dt["metrics"].items():
        print(f"  {name:14s} MAE={m['MAE']:6.2f} min  RMSE={m['RMSE']:6.2f}  "
              f"R2={m['R2']:.3f}")
    print(f"  {'baseline':14s} MAE={dt['baseline_mean_mae']:6.2f} min "
          f"(predicting the mean)")
    print("\n  Top feature importances:")
    print(dt["importances"].to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n-- Model 2: rating prediction (delivered orders) --")
    rt = build_rating_model(orders)
    for name, m in rt["metrics"].items():
        print(f"  {name:14s} MAE={m['MAE']:6.2f} stars  RMSE={m['RMSE']:6.2f}  "
              f"R2={m['R2']:.3f}")
    print("\n  Top feature importances:")
    print(rt["importances"].to_string(float_format=lambda v: f"{v:.3f}"))

if __name__ == "__main__":
    main()
