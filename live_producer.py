"""
Big Data Analytics — Mini Project
Real-time layer: simulated live order stream.

Emulates a platform's order-event stream (in production this would be Kafka /
the platform's event bus). Every second it generates a burst of new orders
whose volume follows the real hour-of-day demand curve from the historical
data, and appends them to data/live_orders.jsonl — the file the dashboard's
Live tab tails.

Run:
    python live_producer.py            # Ctrl+C to stop

The dashboard (dashboard.py, Live tab) reads the same file — producer and
dashboard are independent processes, like a real streaming pipeline.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import load_data

LIVE_FILE = Path("data/live_orders.jsonl")
WINDOW_DAYS = 365            # historical window used to learn the demand curve
MAX_RUSH = 8                 # max orders per second at peak

# ---------------------------------------------------------------- reference data
orders, customers, restaurants = load_data()

# hour-of-day demand curve (learned from history, not hardcoded)
hist = orders[orders["order_date"] >= orders["order_date"].max() - pd.Timedelta(days=WINDOW_DAYS)]
hour_curve = hist.groupby("order_hour")["order_id"].count().astype(float)
hour_curve = (hour_curve / hour_curve.max() * MAX_RUSH).round(2)

rng = np.random.default_rng()

cust_ids = customers["customer_id"].to_numpy()
rest_ids = restaurants["restaurant_id"].to_numpy()
rest_rating = restaurants.set_index("restaurant_id")["rating"]
rest_prep = restaurants.set_index("restaurant_id")["prep_time_min"]
cusine_by_rest = restaurants.set_index("restaurant_id")["cuisines"].str.split(", ").str[0]

promo_codes = np.array(["NONE", "SAVE50", "WELCOME20", "FEST10"])
promo_p = [0.62, 0.18, 0.12, 0.08]
drivers = np.array([f"D{i:04d}" for i in range(1, 601)])


def make_order(now: datetime, hour: int) -> dict:
    """Create one live order event with attributes consistent with the batch schema."""
    rest = rng.choice(rest_ids)
    dist = round(float(np.clip(rng.gamma(2.0, 1.8) + 0.4, 0.4, 14)), 2)
    peak = 12 <= hour <= 14 or 19 <= hour <= 22
    traffic = str(rng.choice(["Low", "Moderate", "Heavy"],
                             p=[0.35, 0.35, 0.30] if peak else [0.55, 0.30, 0.15]))
    prep = int(rest_prep[rest])
    rider = dist * (2.6 + 0.9 * (1.25 * peak + rng.random() * 0.8))
    delivery_min = int(np.clip(prep * (1 + 0.18 * peak) + rider + rng.normal(2, 1.2), 12, 95))
    value = int(np.clip(rng.uniform(80, 700), 79, 2000))
    return {
        "order_id": f"L{now.strftime('%Y%m%d%H%M%S')}{rng.integers(100, 999)}",
        "customer_id": str(rng.choice(cust_ids)),
        "restaurant_id": str(rest),
        "order_ts": now.isoformat(timespec="seconds"),
        "order_hour": int(hour),
        "cuisine": str(cusine_by_rest[rest]),
        "order_value": value,
        "delivery_fee": int(np.where(dist < 3, rng.integers(15, 30),
                                     rng.integers(25, 60))),
        "promo_code": str(rng.choice(promo_codes, p=promo_p)),
        "distance_km": dist,
        "traffic_condition": traffic,
        "driver_id": str(rng.choice(drivers)),
        "delivery_time_min": delivery_min,
        "delivery_rating": int(np.clip(round(
            rest_rating[rest] * 0.5 + 1.4 - (delivery_min - 30) / 22
            + rng.normal(0, 0.5)), 1, 5)),
        "weather": "Clear",
        "status": "Delivered",
    }


def main():
    LIVE_FILE.parent.mkdir(exist_ok=True)
    print(f"Streaming live orders -> {LIVE_FILE}")
    print("Pattern follows the historical hour-of-day demand curve "
          f"(peak = {hour_curve.max():.0f} orders/sec-burst). Ctrl+C to stop.")
    seq = 0
    try:
        while True:
            now = datetime.now()
            # orders/sec for this second, from the learned curve
            rate = float(hour_curve.get(now.hour, 2.0))
            n = rng.poisson(rate)
            with open(LIVE_FILE, "a", encoding="utf-8") as f:
                for _ in range(n):
                    f.write(json.dumps(make_order(now, now.hour)) + "\n")
                    seq += 1
            if n:
                print(f"\r{now.strftime('%H:%M:%S')}  +{n:2d} orders  (total {seq:>6})   ",
                      end="", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\nStopped. {seq} live orders written in this session.")


if __name__ == "__main__":
    main()
