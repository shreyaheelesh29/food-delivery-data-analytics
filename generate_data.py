"""
Big Data Analytics — Mini Project
Food Delivery Data Analytics: synthetic data generator.

Generates a realistic relational dataset:
  customers.csv   — 5,000 customers across 12 city zones
  restaurants.csv — 800 restaurants, each with cuisines, ratings, prep time
  orders.csv      — 50,000 orders with delivery times, ratings, timestamps

All relationships are engineered so real patterns exist to discover:
  * Peak hours: lunch (12–14) and dinner (19–22) rushes
  * Weekend uplift
  * Delivery time grows with distance, and degrades during peak hours & rain
  * Rating correlates with delivery time and food quality (restaurant rating)
  * Cuisine popularity varies by zone and time of day
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
rng_f = np.random.default_rng(4242)  # separate stream for fee/promo fields — keeps
                                     # base CSVs byte-identical when regenerating

N_CUSTOMERS = 5000
N_RESTAURANTS = 800
N_ORDERS = 50_000

# ---------------------------------------------------------------- zones
ZONES = [
    ("Koramangala", 12.9352, 77.6245),
    ("Indiranagar", 12.9784, 77.6408),
    ("Whitefield", 12.9698, 77.7500),
    ("HSR Layout", 12.9116, 77.6474),
    ("Jayanagar", 12.9250, 77.5938),
    ("Electronic City", 12.8452, 77.6602),
    ("Marathahalli", 12.9569, 77.7011),
    ("Rajajinagar", 12.9916, 77.5525),
    ("Malleshwaram", 13.0035, 77.5696),
    ("Hebbal", 13.0358, 77.5970),
    ("BTM Layout", 12.9166, 77.6101),
    ("Yelahanka", 13.1007, 77.5963),
]
# population-ish weight per zone (demand)
ZONE_WEIGHTS = np.array([0.16, 0.12, 0.11, 0.10, 0.08, 0.09,
                         0.09, 0.06, 0.05, 0.06, 0.05, 0.03])

CUISINES = ["North Indian", "South Indian", "Chinese", "Italian", "Mexican",
            "Biryani", "Burgers", "Pizza", "Desserts", "Healthy",
            "Rolls & Wraps", "Korean", "Continental", "Seafood", "Cafe"]

# popularity weight of each cuisine (Biryani & North Indian heavy — India)
CUISINE_W = np.array([0.14, 0.13, 0.12, 0.07, 0.04, 0.12, 0.07, 0.07,
                      0.05, 0.04, 0.06, 0.03, 0.03, 0.02, 0.01])

VEG = {"South Indian": 0.85, "Desserts": 0.9, "Healthy": 0.5, "North Indian": 0.6}


def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between coordinate pairs."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- customers
cust_zones = rng.choice(len(ZONES), size=N_CUSTOMERS, p=ZONE_WEIGHTS)
customers = pd.DataFrame({
    "customer_id": [f"C{i:05d}" for i in range(1, N_CUSTOMERS + 1)],
    "zone": [ZONES[i][0] for i in cust_zones],
    "signup_date": pd.to_datetime("2023-01-01")
    + pd.to_timedelta(rng.integers(0, 730, N_CUSTOMERS), unit="D"),
})

# ---------------------------------------------------------------- restaurants
rest_zones = rng.choice(len(ZONES), size=N_RESTAURANTS, p=ZONE_WEIGHTS)
cuisine_lists = []
for _ in range(N_RESTAURANTS):
    k = rng.choice([1, 2, 3], p=[0.55, 0.35, 0.10])
    # popularity-weighted cuisine mix
    picks = rng.choice(CUISINES, size=k, replace=False, p=CUISINE_W / CUISINE_W.sum())
    cuisine_lists.append(", ".join(picks))

base_quality = rng.beta(5, 2, N_RESTAURANTS)          # latent food quality
rest_rating = np.clip(base_quality * 1.6 + rng.normal(2.9, 0.15, N_RESTAURANTS)
                      - (rng.random(N_RESTAURANTS) < 0.05) * 0.8, 2.0, 4.9)
prep_time = np.clip(rng.normal(18, 6, N_RESTAURANTS) + (5 - rest_rating) * 2, 8, 45)

restaurants = pd.DataFrame({
    "restaurant_id": [f"R{i:04d}" for i in range(1, N_RESTAURANTS + 1)],
    "zone": [ZONES[i][0] for i in rest_zones],
    "cuisines": cuisine_lists,
    "rating": np.round(rest_rating, 1),
    "price_for_two": np.round(rng.choice([150, 250, 350, 500, 700],
                                         p=[.15, .30, .28, .18, .09], size=N_RESTAURANTS), -1),
    "prep_time_min": np.round(prep_time).astype(int),
    "lat": [ZONES[i][1] + rng.normal(0, 0.012) for i in rest_zones],
    "lon": [ZONES[i][2] + rng.normal(0, 0.012) for i in rest_zones],
})

# ---------------------------------------------------------------- orders
days = pd.date_range("2024-01-01", "2024-12-31", freq="D")

# day-of-year weight: weekends busier, December festive bump
day_w = np.where(days.dayofweek >= 5, 1.35, 1.0)
day_w *= np.where(days.month == 12, 1.25, 1.0)
day_w /= day_w.sum()
order_days = pd.DatetimeIndex(rng.choice(days, size=N_ORDERS, p=day_w))

# hour-of-day: lunch 12–14 and dinner 19–22 peaks
hours = np.arange(24)
hour_w = np.array([0.5, 0.3, 0.2, 0.15, 0.15, 0.3, 0.8, 1.5, 2.0, 1.8,
                   1.6, 2.2, 4.5, 4.8, 3.2, 2.0, 1.6, 2.0, 4.0, 6.0, 6.2, 5.0, 2.5, 1.0])
hour_w /= hour_w.sum()
order_hours = rng.choice(hours, size=N_ORDERS, p=hour_w)

is_weekend = (order_days.dayofweek >= 5).astype(int)
is_peak = ((order_hours >= 12) & (order_hours <= 14)) | ((order_hours >= 19) & (order_hours <= 22))
is_rain = rng.random(N_ORDERS) < 0.18  # monsoon-ish probability

# traffic condition (delivery-log flavor): worse during peak hours & weekends
traffic_prob = np.full((N_ORDERS, 3), [0.50, 0.35, 0.15])          # Low, Moderate, Heavy
traffic_prob[:, 2] += 0.28 * is_peak + 0.08 * is_weekend           # shift to Heavy
traffic_prob[:, 1] -= 0.18 * is_peak
traffic_prob[:, 0] -= 0.10 * is_peak
traffic_prob = np.clip(traffic_prob, 0.02, 0.95)
traffic_prob /= traffic_prob.sum(axis=1, keepdims=True)
traffic_condition = np.array([rng_f.choice(["Low", "Moderate", "Heavy"], p=p)
                              for p in traffic_prob])

order_customers = rng.integers(0, N_CUSTOMERS, N_ORDERS)
order_rests = rng.integers(0, N_RESTAURANTS, N_ORDERS)

o_zone = customers["zone"].values[order_customers]
r_zone = restaurants["zone"].values[order_rests]
dist = haversine(
    customers["zone"].map({z[0]: z[1] for z in ZONES}).values[order_customers],
    customers["zone"].map({z[0]: z[2] for z in ZONES}).values[order_customers],
    restaurants["lat"].values[order_rests],
    restaurants["lon"].values[order_rests],
)
# 80% of orders stay local — clamp cross-zone distances realistically
dist = np.where(rng.random(N_ORDERS) < 0.8, dist * 0.25, dist)
dist = np.clip(dist + 0.4, 0.4, 14)

prep = restaurants["prep_time_min"].values[order_rests]

traffic = (1.25 * is_peak + 0.9 * is_weekend + 1.1 * is_rain)
traffic_mult = np.select([traffic_condition == "Heavy", traffic_condition == "Moderate"],
                         [1.14, 1.06], default=1.0)
rider_time = (dist * (2.6 + 0.9 * traffic) * traffic_mult
              + rng.normal(2, 1.2, N_ORDERS))
delivery_time = np.clip(prep * (1 + 0.18 * is_peak + 0.25 * is_rain)
                        + rider_time, 12, 95).astype(int)

# rating: driven by delivery time, restaurant quality, rain penalty
lat_quality = (base_quality[order_rests] - 0.55) * 2.2
score = (lat_quality
         - (delivery_time - 32) / 22
         - 0.35 * is_rain
         + rng.normal(0, 0.55, N_ORDERS))
rating = np.clip(np.round(score * 1.15 + 3.05), 1, 5)

order_values = (restaurants["price_for_two"].values[order_rests]
                * rng.uniform(0.5, 1.4, N_ORDERS) / 2).round(0)
order_values = np.clip(order_values + rng.choice([0, 25, 50, 99],
                                                 p=[.55, .2, .15, .1], size=N_ORDERS), 79, 2000)

status = rng.choice(["Delivered", "Cancelled"], size=N_ORDERS, p=[.94, .06])

# ---- delivery-log commerce fields (separate rng_f stream: base CSVs unchanged)
delivery_fee = np.where(dist < 3, rng_f.integers(15, 30, N_ORDERS),
                        np.where(dist < 7, rng_f.integers(25, 45, N_ORDERS),
                                 rng_f.integers(35, 65, N_ORDERS))).astype(int)
promo = rng_f.choice(["NONE", "SAVE50", "WELCOME20", "FEST10"], size=N_ORDERS,
                     p=[0.62, 0.18, 0.12, 0.08])
promo_discount = np.select(
    [promo == "SAVE50", promo == "WELCOME20", promo == "FEST10"],
    [np.minimum(50, order_values * 0.1),
     np.minimum(order_values * 0.20, 120),
     order_values * 0.10], default=0.0)
discount = np.round(promo_discount).astype(int)
drivers = np.array([f"D{i:04d}" for i in range(1, 601)])
driver_id = rng_f.choice(drivers, size=N_ORDERS)
loyalty_tier = rng_f.choice(["Bronze", "Silver", "Gold"], size=N_ORDERS,
                            p=[0.55, 0.30, 0.15])

orders = pd.DataFrame({
    "order_id": [f"O{i:06d}" for i in range(1, N_ORDERS + 1)],
    "customer_id": customers["customer_id"].values[order_customers],
    "restaurant_id": restaurants["restaurant_id"].values[order_rests],
    "order_date": order_days.strftime("%Y-%m-%d"),
    "order_hour": order_hours,
    "is_weekend": is_weekend,
    "cuisine": rng.choice(CUISINES, size=N_ORDERS, p=CUISINE_W / CUISINE_W.sum()),
    "order_value": order_values.astype(int),
    "delivery_fee": delivery_fee,
    "discount": discount,
    "promo_code": promo,
    "distance_km": np.round(dist, 2),
    "traffic_condition": traffic_condition,
    "driver_id": driver_id,
    "delivery_time_min": delivery_time,
    "delivery_rating": rating,
    "weather": np.where(is_rain, "Rain", "Clear"),
    "status": status,
})

# customer loyalty tier on the customer table
customers["loyalty_tier"] = rng_f.choice(["Bronze", "Silver", "Gold"],
                                         size=N_CUSTOMERS, p=[0.55, 0.30, 0.15])
# cancelled orders have no delivery time / rating / driver assignment
orders.loc[orders["status"] == "Cancelled",
           ["delivery_time_min", "delivery_rating", "driver_id"]] = np.nan

# ---------------------------------------------------------------- write
customers.to_csv("data/customers.csv", index=False)
restaurants.to_csv("data/restaurants.csv", index=False)
orders.to_csv("data/orders.csv", index=False)
print(f"customers:   {len(customers):>6,} rows")
print(f"restaurants: {len(restaurants):>6,} rows")
print(f"orders:      {len(orders):>6,} rows")
