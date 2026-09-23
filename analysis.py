"""
Big Data Analytics — Mini Project
Food Delivery Data Analytics — analysis layer.

Pure-pandas functions that compute every metric the dashboard and console
report need. Import `load_data()` and call the individual analysis functions,
or run `python analysis.py` for a full console report.
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)


# ------------------------------------------------------------------ loading
def load_data(data_dir: str = "data"):
    orders = pd.read_csv(f"{data_dir}/orders.csv", parse_dates=["order_date"])
    customers = pd.read_csv(f"{data_dir}/customers.csv", parse_dates=["signup_date"])
    restaurants = pd.read_csv(f"{data_dir}/restaurants.csv")
    # convenience flags / derived columns
    orders["month"] = orders["order_date"].dt.to_period("M").astype(str)
    orders["weekday"] = orders["order_date"].dt.day_name()
    return orders, customers, restaurants


# ------------------------------------------------------------------ headline KPIs
def headline_kpis(orders, restaurants):
    delivered = orders[orders["status"] == "Delivered"]
    return {
        "total_orders": len(orders),
        "total_revenue": int(orders["order_value"].sum()),
        "avg_order_value": round(orders["order_value"].mean(), 1),
        "avg_delivery_time": round(delivered["delivery_time_min"].mean(), 1),
        "avg_rating": round(delivered["delivery_rating"].mean(), 2),
        "cancellation_rate": round((orders["status"] == "Cancelled").mean() * 100, 2),
        "active_customers": orders["customer_id"].nunique(),
        "active_restaurants": orders["restaurant_id"].nunique(),
        "restaurants_total": len(restaurants),
    }


# ------------------------------------------------------------------ cuisine popularity
def cuisine_popularity(orders):
    """Top cuisines by order count, revenue and average rating."""
    g = (orders.groupby("cuisine")
         .agg(orders=("order_id", "count"),
              revenue=("order_value", "sum"),
              avg_rating=("delivery_rating", "mean"),
              avg_delivery_time=("delivery_time_min", "mean"))
         .round({"avg_rating": 2, "avg_delivery_time": 1})
         .sort_values("orders", ascending=False))
    g["revenue_lakh"] = (g["revenue"] / 1e5).round(1)
    return g


# ------------------------------------------------------------------ busiest restaurants
def busiest_restaurants(orders, restaurants, top_n=15):
    g = (orders.groupby("restaurant_id")
         .agg(orders=("order_id", "count"),
              revenue=("order_value", "sum"),
              avg_rating=("delivery_rating", "mean"))
         .join(restaurants.set_index("restaurant_id")[["zone", "cuisines", "rating"]])
         .round({"avg_rating": 2})
         .sort_values("orders", ascending=False)
         .head(top_n))
    return g


# ------------------------------------------------------------------ demand by location
def zone_demand(orders, restaurants):
    rest_zone = restaurants.set_index("restaurant_id")["zone"]
    cust_zone = orders["customer_id"]  # customer zone via customers table
    out = (orders.assign(rest_zone=orders["restaurant_id"].map(rest_zone))
           .groupby("rest_zone")
           .agg(orders=("order_id", "count"),
                revenue=("order_value", "sum"),
                avg_delivery_time=("delivery_time_min", "mean"),
                avg_rating=("delivery_rating", "mean"))
           .round({"avg_delivery_time": 1, "avg_rating": 2})
           .sort_values("orders", ascending=False))
    out["revenue_lakh"] = (out["revenue"] / 1e5).round(1)
    return out


# ------------------------------------------------------------------ peak hours / weekday
def hourly_demand(orders):
    return (orders.groupby("order_hour")
            .agg(orders=("order_id", "count"),
                 avg_delivery_time=("delivery_time_min", "mean"),
                 avg_rating=("delivery_rating", "mean"))
            .round(2))


def weekday_demand(orders):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    g = (orders.groupby("weekday")
         .agg(orders=("order_id", "count"), revenue=("order_value", "sum"))
         .reindex(days))
    return g


# ------------------------------------------------------------------ factors affecting delivery time
def delivery_time_factors(orders):
    delivered = orders[orders["status"] == "Delivered"].copy()
    buckets = pd.cut(delivered["distance_km"],
                     bins=[0, 2, 4, 6, 8, 10, 15],
                     labels=["0-2 km", "2-4 km", "4-6 km", "6-8 km", "8-10 km", "10+ km"])
    by_distance = (delivered.assign(dist_bucket=buckets)
                   .groupby("dist_bucket", observed=True)["delivery_time_min"]
                   .agg(["mean", "count"]).round(1))

    by_hour = delivered.groupby("order_hour")["delivery_time_min"].mean().round(1)

    by_weather = (delivered.groupby("weather")
                  .agg(avg_time=("delivery_time_min", "mean"),
                       avg_rating=("delivery_rating", "mean"),
                       orders=("order_id", "count")).round(1))

    by_traffic = (delivered.groupby("traffic_condition")
                  .agg(avg_time=("delivery_time_min", "mean"),
                       avg_rating=("delivery_rating", "mean"),
                       orders=("order_id", "count")).round(1)
                  .reindex(["Low", "Moderate", "Heavy"]))

    weekend_effect = (delivered.groupby("is_weekend")
                      .agg(avg_time=("delivery_time_min", "mean"),
                           orders=("order_id", "count")).round(1))

    corr = delivered[["delivery_time_min", "distance_km", "order_hour",
                      "order_value"]].corr()["delivery_time_min"].round(3)
    return {"by_distance": by_distance, "by_hour": by_hour,
            "by_weather": by_weather, "by_traffic": by_traffic,
            "weekend": weekend_effect, "correlations": corr}


# ------------------------------------------------------------------ factors affecting rating
def rating_factors(orders):
    delivered = orders[orders["status"] == "Delivered"].copy()
    time_buckets = pd.cut(delivered["delivery_time_min"],
                          bins=[0, 25, 35, 45, 60, 120],
                          labels=["<25 min", "25-35 min", "35-45 min", "45-60 min", ">60 min"])
    rating_by_time = (delivered.assign(time_bucket=time_buckets)
                      .groupby("time_bucket", observed=True)["delivery_rating"]
                      .agg(["mean", "count"]).round(2))

    rating_by_hour = delivered.groupby("order_hour")["delivery_rating"].mean().round(2)

    rating_by_weather = (delivered.groupby("weather")["delivery_rating"]
                         .agg(["mean", "count"]).round(2))

    corr = delivered[["delivery_rating", "delivery_time_min", "distance_km",
                      "order_hour", "order_value"]].corr()["delivery_rating"].round(3)
    return {"rating_by_time": rating_by_time, "rating_by_hour": rating_by_hour,
            "rating_by_weather": rating_by_weather, "correlations": corr}


# ------------------------------------------------------------------ monthly trend
def monthly_trend(orders):
    g = (orders.groupby("month")
         .agg(orders=("order_id", "count"),
              revenue=("order_value", "sum"),
              avg_rating=("delivery_rating", "mean"),
              avg_delivery_time=("delivery_time_min", "mean"))
         .round({"avg_rating": 2, "avg_delivery_time": 1}))
    return g


# ------------------------------------------------------------------ console report
def main():
    orders, customers, restaurants = load_data()

    print("=" * 70)
    print("FOOD DELIVERY DATA ANALYTICS — CONSOLE REPORT")
    print("=" * 70)

    kpis = headline_kpis(orders, restaurants)
    print("\n-- Headline KPIs --")
    for k, v in kpis.items():
        print(f"  {k:22s}: {v:,}" if isinstance(v, int) else f"  {k:22s}: {v}")

    print("\n-- Top 10 cuisines by orders --")
    print(cuisine_popularity(orders).head(10))

    print("\n-- 15 busiest restaurants --")
    print(busiest_restaurants(orders, restaurants))

    print("\n-- Demand by zone (restaurant side) --")
    print(zone_demand(orders, restaurants))

    print("\n-- Hourly demand --")
    print(hourly_demand(orders))

    print("\n-- Weekday demand --")
    print(weekday_demand(orders))

    f = delivery_time_factors(orders)
    print("\n-- Delivery time vs distance --")
    print(f["by_distance"])
    print("\n-- Delivery time by weather --")
    print(f["by_weather"])
    print("\n-- Delivery time by traffic condition --")
    print(f["by_traffic"])
    print("\n-- Weekend effect on delivery time --")
    print(f["weekend"])
    print("\n-- Correlations with delivery_time_min --")
    print(f["correlations"])

    r = rating_factors(orders)
    print("\n-- Rating vs delivery-time bucket --")
    print(r["rating_by_time"])
    print("\n-- Rating by weather --")
    print(r["rating_by_weather"])
    print("\n-- Correlations with delivery_rating --")
    print(r["correlations"])

    print("\n-- Monthly trend --")
    print(monthly_trend(orders))


if __name__ == "__main__":
    main()
