"""
Big Data Analytics — Mini Project
Food Delivery Analytics Dashboard (Streamlit + Plotly).

Run:  streamlit run dashboard.py
"""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import (load_data, headline_kpis, cuisine_popularity,
                      busiest_restaurants, zone_demand, hourly_demand,
                      weekday_demand, delivery_time_factors, rating_factors,
                      monthly_trend)

st.set_page_config(page_title="Food Delivery Analytics", page_icon="🍔",
                   layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ data (cached)
@st.cache_data
def load():
    orders, customers, restaurants = load_data()
    return orders, customers, restaurants

@st.cache_resource
def train_models(_orders, _restaurants):
    from modeling import build_delivery_time_model, build_rating_model
    dt = build_delivery_time_model(_orders, _restaurants)
    rt = build_rating_model(_orders)
    return dt, rt

orders, customers, restaurants = load()

# ------------------------------------------------------------------ sidebar filters
st.sidebar.header("Filters")

min_date = orders["order_date"].min().date()
max_date = orders["order_date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_date, max_date),
                                   min_value=min_date, max_value=max_date)

zone_options = ["All"] + sorted(restaurants["zone"].unique())
zone = st.sidebar.selectbox("Restaurant zone", zone_options)

cuisine_options = ["All"] + sorted(orders["cuisine"].unique())
cuisine = st.sidebar.selectbox("Cuisine", cuisine_options)

promo_options = ["All"] + sorted(orders["promo_code"].unique())
promo = st.sidebar.selectbox("Promo code", promo_options)

hours_range = st.sidebar.slider("Order hour", 0, 23, (0, 23))
status = st.sidebar.radio("Order status", ["All", "Delivered only", "Cancelled only"])

if status == "Delivered only":
    orders_f = orders[orders["status"] == "Delivered"]
elif status == "Cancelled only":
    orders_f = orders[orders["status"] == "Cancelled"]
else:
    orders_f = orders

mask = orders_f["order_date"].dt.date.between(date_range[0], date_range[-1]) \
       & orders_f["order_hour"].between(*hours_range)
if zone != "All":
    rest_zone_map = restaurants.set_index("restaurant_id")["zone"]
    mask &= orders_f["restaurant_id"].map(rest_zone_map) == zone
if cuisine != "All":
    mask &= orders_f["cuisine"] == cuisine
if promo != "All":
    mask &= orders_f["promo_code"] == promo
orders_f = orders_f[mask]

# ------------------------------------------------------------------ header
st.title("🍔 Food Delivery Analytics Dashboard")
st.caption("Big Data Analytics Mini Project — 50,000 orders · 800 restaurants · 5,000 customers (2024)")

kpis = headline_kpis(orders_f, restaurants)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Orders", f"{kpis['total_orders']:,}")
c2.metric("Revenue", f"₹{kpis['total_revenue']:,}")
c3.metric("Avg Order Value", f"₹{kpis['avg_order_value']}")
c4.metric("Avg Delivery Time", f"{kpis['avg_delivery_time']} min")
c5.metric("Avg Rating", f"⭐ {kpis['avg_rating']}")
c6.metric("Cancellation Rate", f"{kpis['cancellation_rate']}%")

# ------------------------------------------------------------------ tabs
tab_overview, tab_cuisine, tab_rest, tab_geo, tab_time, tab_factors, tab_whatif, tab_live = st.tabs(
    ["📈 Overview", "🍽️ Cuisines", "🏪 Restaurants", "📍 Locations",
     "⏰ Peak Hours", "🚚 Delivery & Rating Factors", "🤖 What-If (ML)", "🔴 Live"])

# ================================================================== LIVE tab
# Real-time layer: orders streamed by live_producer.py are tailed from
# data/live_orders.jsonl and merged with today's slice of the historical data
# for rolling KPIs. Auto-refreshes every 3 s while the tab is open.
with tab_live:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=3000, key="live_refresh")
    except ImportError:
        st.warning("pip install streamlit-autorefresh for auto-refresh; \
                   use the browser refresh button otherwise.")

    from pathlib import Path

    live_file = Path("data/live_orders.jsonl")

    @st.cache_data(ttl=2)
    def read_live(path_str: str):
        p = Path(path_str)
        if not p.exists():
            return pd.DataFrame()
        rows = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)

    live = read_live(str(live_file))

    st.markdown(
        "**🔴 Live stream** — new orders arrive every second from "
        "`live_producer.py` (rate follows the historical demand curve). "
        "This page auto-refreshes every 3 s.")

    if live.empty:
        st.info("No live orders yet. Start the producer in a second terminal: "
                "`python live_producer.py`")
    else:
        live["order_ts"] = pd.to_datetime(live["order_ts"])
        live["minute"] = live["order_ts"].dt.floor("min")

        now = pd.Timestamp.now()
        last5 = live[live["order_ts"] >= now - pd.Timedelta(minutes=5)]

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Live orders", f"{len(live):,}")
        k2.metric("Orders / min (last 5 min)", f"{len(last5) / 5:.1f}")
        k3.metric("Revenue (live)", f"₹{live['order_value'].sum():,}")
        k4.metric("Avg delivery time",
                  f"{live['delivery_time_min'].mean():.0f} min")
        k5.metric("Avg rating",
                  f"{live['delivery_rating'].mean():.2f} ★")

        left, right = st.columns(2)

        per_min = (live.groupby("minute")
                   .agg(orders=("order_id", "count"),
                        revenue=("order_value", "sum")))
        fig = px.area(per_min.reset_index(), x="minute", y="orders",
                      title="Live order rate (per minute)",
                      labels={"minute": "Time", "orders": "Orders"})
        fig.update_layout(height=340)
        left.plotly_chart(fig, use_container_width=True)

        live_cuisine = live["cuisine"].value_counts().head(6)
        fig = px.bar(x=live_cuisine.values, y=live_cuisine.index,
                     orientation="h",
                     title="Live orders by cuisine (top 6)",
                     labels={"x": "Orders", "y": ""})
        fig.update_layout(height=340, showlegend=False)
        right.plotly_chart(fig, use_container_width=True)

        # live restaurant positions on the city map
        live_rest = (live.groupby("restaurant_id").size()
                     .rename("live_orders").to_frame()
                     .join(restaurants.set_index("restaurant_id")[["lat", "lon",
                                                                   "zone"]] 
                           if all(c in restaurants.columns for c in
                                  ("lat", "lon")) else None,
                           how="inner"))
        if not live_rest.empty:
            fig = px.scatter_map(live_rest.reset_index(), lat="lat", lon="lon",
                                 size="live_orders", zoom=10, height=360,
                                 map_style="open-street-map",
                                 hover_name="zone",
                                 title="Live orders by restaurant location")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Latest orders")
        show_cols = ["order_ts", "order_id", "cuisine", "order_value",
                     "distance_km", "traffic_condition", "delivery_time_min",
                     "delivery_rating"]
        st.dataframe(live.sort_values("order_ts", ascending=False)
                     [show_cols].head(12), use_container_width=True,
                     height=340)

# ================================================================== static tabs

# ------------------------------------------------------------------ overview
with tab_overview:
    left, right = st.columns(2)

    monthly = monthly_trend(orders_f)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly["orders"],
                             name="Orders", mode="lines+markers", yaxis="y"))
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly["revenue"],
                             name="Revenue", mode="lines+markers",
                             yaxis="y2", line=dict(dash="dot")))
    fig.update_layout(
        title="Monthly Orders & Revenue",
        yaxis=dict(title="Orders"),
        yaxis2=dict(title="Revenue (₹)", overlaying="y", side="right"),
        height=380, legend=dict(orientation="h", y=1.12))
    left.plotly_chart(fig, use_container_width=True)

    wd = weekday_demand(orders_f)
    colors = ["#EF553B" if d in ("Saturday", "Sunday") else "#636EFA"
              for d in wd.index]
    fig = px.bar(x=wd.index, y=wd["orders"], title="Weekday Demand",
                 labels={"x": "", "y": "Orders"}, color=colors)
    fig.update_layout(showlegend=False, height=380)
    right.plotly_chart(fig, use_container_width=True)

    delivered = orders_f[orders_f["status"] == "Delivered"]
    fig = px.histogram(delivered, x="delivery_time_min", nbins=45,
                       title="Delivery Time Distribution",
                       labels={"delivery_time_min": "Delivery time (min)"},
                       color_discrete_sequence=["#00CC96"])
    fig.add_vline(x=delivered["delivery_time_min"].mean(),
                  line_dash="dash", annotation_text=f"mean {delivered['delivery_time_min'].mean():.0f} min")
    fig.update_layout(height=380)
    left.plotly_chart(fig, use_container_width=True)

    rating_counts = delivered["delivery_rating"].value_counts().sort_index()
    fig = px.bar(x=rating_counts.index.astype(str), y=rating_counts.values,
                 title="Rating Distribution",
                 labels={"x": "Rating", "y": "# orders"},
                 color_discrete_sequence=["#AB63FA"])
    fig.update_layout(height=380)
    right.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ cuisines
with tab_cuisine:
    cp = cuisine_popularity(orders_f)

    left, right = st.columns(2)
    fig = px.bar(cp.head(10).sort_values("orders"),
                 x="orders", y=cp.head(10).sort_values("orders").index,
                 orientation="h", title="Top 10 Cuisines by Orders",
                 labels={"y": "Cuisine", "x": "Orders"},
                 color="orders", color_continuous_scale="Sunset")
    fig.update_layout(height=420, showlegend=False)
    left.plotly_chart(fig, use_container_width=True)

    fig = px.pie(cp.head(8), names=cp.head(8).index, values="orders",
                 title="Cuisine Share (Top 8)", hole=0.45)
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=420)
    right.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    fig = px.bar(cp.sort_values("revenue_lakh", ascending=False).head(10),
                 x=cp.sort_values("revenue_lakh", ascending=False).head(10).index,
                 y="revenue_lakh", title="Top Cuisines by Revenue (₹ Lakh)",
                 labels={"x": "Cuisine", "y": "Revenue (₹ lakh)"},
                 color_discrete_sequence=["#FFA15A"])
    fig.update_layout(height=400, showlegend=False)
    left.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(cp, x="avg_delivery_time", y="avg_rating",
                     size="orders", color=cp.index, text=cp.index,
                     title="Cuisine Quality Map — rating vs delivery time (bubble = orders)",
                     labels={"avg_delivery_time": "Avg delivery time (min)",
                             "avg_rating": "Avg rating"})
    fig.update_layout(height=420, showlegend=False)
    right.plotly_chart(fig, use_container_width=True)

    st.subheader("Cuisine summary table")
    st.dataframe(cp, use_container_width=True)

# ------------------------------------------------------------------ restaurants
with tab_rest:
    br = busiest_restaurants(orders_f, restaurants, top_n=15)
    left, right = st.columns(2)

    fig = px.bar(br.sort_values("orders"),
                 x="orders", y=br.sort_values("orders").index,
                 orientation="h", title="15 Busiest Restaurants (by orders)",
                 labels={"y": "Restaurant", "x": "Orders"},
                 color="rating", color_continuous_scale="RdYlGn",
                 hover_data=["zone", "cuisines"])
    fig.update_layout(height=520)
    left.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(br, x="orders", y="avg_rating", size="revenue",
                     color="zone", hover_name=br.index,
                     title="Busiest Restaurants — orders vs rating (size = revenue)")
    fig.update_layout(height=520)
    right.plotly_chart(fig, use_container_width=True)

    st.subheader("Busiest restaurants detail")
    st.dataframe(br, use_container_width=True)

# ------------------------------------------------------------------ locations
with tab_geo:
    zd = zone_demand(orders_f, restaurants)

    left, right = st.columns([3, 2])
    fig = px.bar(zd.sort_values("orders"),
                 x="orders", y=zd.sort_values("orders").index,
                 orientation="h", title="Demand by Zone (orders served)",
                 labels={"y": "Zone", "x": "Orders"},
                 color="avg_delivery_time", color_continuous_scale="Turbo",
                 hover_data=["revenue_lakh", "avg_rating"])
    fig.update_layout(height=520)
    left.plotly_chart(fig, use_container_width=True)

    # scatter-geo of restaurant locations, sized by orders received
    rest_orders = (orders_f.groupby("restaurant_id")
                   .agg(orders=("order_id", "count"),
                        revenue=("order_value", "sum"))
                   .join(restaurants.set_index("restaurant_id")))
    fig = px.scatter_map(rest_orders, lat="lat", lon="lon", size="orders",
                            color="rating", zoom=10, height=520,
                            color_continuous_scale="RdYlGn", size_max=18,
                            map_style="open-street-map",
                            hover_name="zone",
                            title="Restaurant Map — size = orders, color = rating")
    right.plotly_chart(fig, use_container_width=True)

    st.subheader("Zone summary")
    st.dataframe(zd, use_container_width=True)

# ------------------------------------------------------------------ peak hours
with tab_time:
    hd = hourly_demand(orders_f)
    left, right = st.columns(2)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=hd.index, y=hd["orders"], name="Orders",
                         marker_color="#636EFA"))
    fig.add_trace(go.Scatter(x=hd.index, y=hd["avg_delivery_time"],
                             name="Avg delivery (min)", mode="lines+markers",
                             yaxis="y2", line=dict(color="#EF553B", width=3)))
    peak_hours = hd["orders"].nlargest(4).index.tolist()
    for h in peak_hours:
        fig.add_vrect(x0=h - 0.5, x1=h + 0.5, fillcolor="red", opacity=0.12,
                      annotation_text=f"peak {h}:00" if h == max(peak_hours) else None)
    fig.update_layout(title="Hourly Demand & Delivery Time (peak hours shaded)",
                      yaxis=dict(title="Orders"),
                      yaxis2=dict(title="Avg delivery (min)", overlaying="y",
                                  side="right"),
                      height=420, legend=dict(orientation="h", y=1.12))
    left.plotly_chart(fig, use_container_width=True)

    delivered = orders_f[orders_f["status"] == "Delivered"]
    heat = (delivered.pivot_table(index="weekday",
                                  columns="order_hour",
                                  values="order_id", aggfunc="count")
            .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                      "Saturday", "Sunday"]))
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                    title="Order Heatmap — weekday × hour",
                    labels={"x": "Hour of day", "y": "", "color": "Orders"})
    fig.update_layout(height=420)
    right.plotly_chart(fig, use_container_width=True)

    st.info("💡 **Peak windows:** lunch 12–14 h and dinner 19–22 h carry the "
            "double burden of highest demand **and** the slowest deliveries "
            "(~43–44 min vs ~35 min off-peak) — prime targets for rider "
            "incentives and surge staffing.")

# ------------------------------------------------------------------ factors
with tab_factors:
    f = delivery_time_factors(orders_f)
    r = rating_factors(orders_f)

    left, right = st.columns(2)

    fig = px.bar(f["by_distance"].reset_index(), x="dist_bucket",
                 y="mean", title="Avg Delivery Time vs Distance",
                 labels={"dist_bucket": "Distance bucket", "mean": "Avg time (min)"},
                 color="mean", color_continuous_scale="OrRd",
                 text="mean")
    fig.update_layout(height=400, showlegend=False)
    left.plotly_chart(fig, use_container_width=True)

    fig = px.bar(r["rating_by_time"].reset_index(), x="time_bucket",
                 y="mean", title="Avg Rating vs Delivery Time",
                 labels={"time_bucket": "Delivery time", "mean": "Avg rating"},
                 color="mean", color_continuous_scale="RdYlGn",
                 text="mean")
    fig.update_layout(height=400, showlegend=False)
    right.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    wx = f["by_weather"].reset_index()
    tx = f["by_traffic"].reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tx["traffic_condition"], y=tx["avg_time"],
                         name="Traffic: avg time (min)", marker_color="#8B5CF6",
                         width=0.5))
    fig.add_trace(go.Bar(x=wx["weather"], y=wx["avg_time"], name="Weather: avg time (min)",
                         marker_color="#636EFA", width=0.5))
    fig.update_layout(title="Conditions Impact on Delivery Time",
                      barmode="group", height=380)
    left.plotly_chart(fig, use_container_width=True)

    delivered_all = orders_f[orders_f["status"] == "Delivered"]
    numeric = delivered_all[["delivery_time_min", "delivery_rating",
                             "distance_km", "order_hour", "order_value"]]
    corr = numeric.corr().round(2)
    fig = px.imshow(corr, aspect="auto", text_auto=".2f",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation Matrix — delivery & rating drivers",
                    labels={"color": "r"})
    fig.update_layout(height=380)
    right.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    fig = px.scatter(delivered_all.sample(min(3000, len(delivered_all)),
                                          random_state=1),
                     x="distance_km", y="delivery_time_min",
                     color="weather",
                     trendline="ols",
                     title="Delivery Time vs Distance (colored by weather, OLS trend)",
                     labels={"distance_km": "Distance (km)",
                             "delivery_time_min": "Delivery time (min)"},
                     opacity=0.35, height=420)
    left.plotly_chart(fig, use_container_width=True)

    time_bins = pd.cut(delivered_all["delivery_time_min"],
                       bins=[0, 20, 40, 60, 120],
                       labels=["<20 min", "20-40 min", "40-60 min", ">60 min"])
    fig = px.box(delivered_all.assign(time_bin=time_bins),
                 x="time_bin", y="delivery_rating", color="time_bin",
                 title="Rating by Delivery-Time Bin",
                 labels={"time_bin": "Delivery time", "delivery_rating": "Rating"},
                 category_orders={"time_bin": ["<20 min", "20-40 min",
                                               "40-60 min", ">60 min"]},
                 height=420)
    right.plotly_chart(fig, use_container_width=True)
    fig.update_layout(height=380)
    right.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Key factor findings**\n\n"
        "- **Distance dominates delivery time** (r ≈ 0.81): 0–2 km orders average "
        "~29 min while 10+ km orders take ~73 min.\n"
        "- **Peak hours add ~9 min** (traffic surge); weekends add ~3.5 min.\n"
        "- **Heavy traffic adds ~4 min**, rain ~9 min over clear/low conditions.\n"
        "- **Delivery time is the #1 rating driver** (r ≈ −0.69): ratings fall from "
        "4.0★ (<25 min) to 1.4★ (>60 min).\n"
        "- Faster zones (Koramangala) rate ~3.3★; slow peripheral zones (Yelahanka) "
        "rate ~2.7★ — logistics, not food quality, explains most of the gap.")

# ------------------------------------------------------------------ what-if (ML)
with tab_whatif:
    st.header("🤖 What-If Simulator — Random Forest predictions")
    st.caption("Models trained on delivered orders. Delivery-time model: "
               "R² 0.987, MAE 1.4 min · Rating model: R² 0.52, MAE 0.61★")

    dt_model, rt_model = train_models(orders, restaurants)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Scenario")
        w_distance = st.slider("Distance (km)", 0.4, 14.0, 4.0, 0.2)
        w_hour = st.slider("Order hour", 0, 23, 20)
        w_weekend = st.checkbox("Weekend", value=True)
        w_weather = st.selectbox("Weather", ["Clear", "Rain"])
        w_traffic = st.selectbox("Traffic", ["Low", "Moderate", "Heavy"], index=2)
        w_prep = st.slider("Restaurant prep time (min)", 8, 45, 18)
        w_cuisine = st.selectbox("Cuisine", sorted(orders["cuisine"].unique()))
        w_zone = st.selectbox("Zone", sorted(restaurants["zone"].unique()))
        w_value = st.slider("Order value (₹)", 79, 2000, 300, 10)

    from modeling import predict as ml_predict

    predicted_time = ml_predict(dt_model, w_distance, w_hour, w_weekend,
                                w_weather, w_traffic, w_prep, w_cuisine, w_zone)

    # predicted rating via the rating model (same encoding scheme)
    row = pd.DataFrame([{
        "delivery_time_min": predicted_time, "distance_km": w_distance,
        "weather": w_weather, "order_value": w_value,
        "promo_code": "NONE", "cuisine": w_cuisine,
        "order_hour": w_hour, "is_weekend": int(w_weekend),
    }])
    encoded = pd.get_dummies(row).reindex(columns=rt_model["feature_columns"],
                                           fill_value=0)
    predicted_rating = float(rt_model["model"].predict(encoded)[0])
    predicted_rating = float(np.clip(predicted_rating, 1, 5))

    with right:
        st.subheader("Prediction")
        k1, k2 = st.columns(2)
        k1.metric("Predicted delivery time", f"{predicted_time} min",
                  help="Random Forest regressor, MAE ±1.4 min")
        k2.metric("Predicted rating", f"{predicted_rating:.1f} ★",
                  help="Random Forest regressor, MAE ±0.61★")

        # sensitivity: predicted time across hours for this scenario
        sweep = pd.DataFrame({
            "order_hour": range(24),
            "distance_km": w_distance, "is_weekend": int(w_weekend),
            "weather": w_weather, "traffic_condition": w_traffic,
            "prep_time_min": w_prep, "cuisine": w_cuisine, "zone": w_zone,
        })
        enc = pd.get_dummies(sweep).reindex(columns=dt_model["feature_columns"],
                                            fill_value=0)
        sweep["predicted_min"] = dt_model["model"].predict(enc)
        fig = px.area(sweep, x="order_hour", y="predicted_min",
                      title="Predicted delivery time by hour of day",
                      labels={"order_hour": "Hour", "predicted_min": "Minutes"})
        fig.add_vline(x=w_hour, line_dash="dash",
                      annotation_text=f"your pick: {w_hour}:00")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

        # sensitivity: predicted time vs distance
        sweep_d = pd.DataFrame({
            "distance_km": np.round(np.arange(0.5, 14.1, 0.5), 1),
            "order_hour": w_hour, "is_weekend": int(w_weekend),
            "weather": w_weather, "traffic_condition": w_traffic,
            "prep_time_min": w_prep, "cuisine": w_cuisine, "zone": w_zone,
        })
        enc_d = pd.get_dummies(sweep_d).reindex(columns=dt_model["feature_columns"],
                                                fill_value=0)
        sweep_d["predicted_min"] = dt_model["model"].predict(enc_d)
        fig = px.line(sweep_d, x="distance_km", y="predicted_min",
                      title="Predicted delivery time vs distance",
                      labels={"distance_km": "Distance (km)",
                              "predicted_min": "Minutes"})
        fig.add_vline(x=w_distance, line_dash="dash",
                      annotation_text=f"your pick: {w_distance} km")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Model quality")
    m1, m2 = st.columns(2)
    m1.markdown("**Delivery-time model**\n\n" + "\n".join(
        f"- `{name}`: MAE **{m['MAE']} min** · RMSE {m['RMSE']} · R² {m['R2']}"
        for name, m in dt_model["metrics"].items()))
    m2.markdown("**Rating model**\n\n" + "\n".join(
        f"- `{name}`: MAE **{m['MAE']}★** · RMSE {m['RMSE']} · R² {m['R2']}"
        for name, m in rt_model["metrics"].items()))

    c1, c2 = st.columns(2)
    fig = px.bar(dt_model["importances"].sort_values(),
                 orientation="h", title="Feature importance — delivery time",
                 labels={"value": "importance", "index": ""}, height=380)
    c1.plotly_chart(fig, use_container_width=True)
    fig = px.bar(rt_model["importances"].sort_values(),
                 orientation="h", title="Feature importance — rating",
                 labels={"value": "importance", "index": ""}, height=380)
    c2.plotly_chart(fig, use_container_width=True)
