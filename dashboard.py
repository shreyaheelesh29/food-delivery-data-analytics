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
from modeling import build_delivery_time_model, build_rating_model

st.set_page_config(page_title="Food Delivery Analytics", page_icon="🍔",
                   layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ data (cached)
@st.cache_data
def load():
    orders, customers, restaurants = load_data()
    return orders, customers, restaurants


@st.cache_resource
def train_models(_orders, _restaurants):
    """Train each prediction model once per Streamlit session cache."""
    return (build_delivery_time_model(_orders, _restaurants),
            build_rating_model(_orders))

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
     "⏰ Peak Hours", "🚚 Delivery & Rating Factors", "🤖 What-If", "🔴 Live"])

# ================================================================== what-if
with tab_whatif:
    st.header("🤖 What-If Simulator")
    st.caption(
        "Enter an order scenario and submit it to estimate delivery time and "
        "customer rating using the trained Random Forest models.")

    with st.form("whatif_scenario"):
        st.subheader("Order scenario")
        col_a, col_b = st.columns(2)
        with col_a:
            scenario_distance = st.number_input(
                "Delivery distance (km)", min_value=0.4, max_value=14.0,
                value=4.0, step=0.2, key="whatif_distance")
            scenario_hour = st.slider(
                "Order hour", min_value=0, max_value=23, value=19,
                help="Use 24-hour time: 19 means 7 PM.")
            scenario_weekend = st.checkbox("Weekend order", value=True)
            scenario_weather = st.selectbox("Weather", ["Clear", "Rain"])
            scenario_traffic = st.selectbox(
                "Traffic", ["Low", "Moderate", "Heavy"], index=1)
        with col_b:
            scenario_zone = st.selectbox(
                "Restaurant zone", sorted(restaurants["zone"].unique()))
            scenario_prep = st.slider(
                "Restaurant prep time (minutes)", 8, 45, 18)
            scenario_cuisine = st.selectbox(
                "Cuisine", sorted(orders["cuisine"].unique()))
            scenario_value = st.number_input(
                "Order value (₹)", min_value=79, max_value=2000,
                value=300, step=10)
            scenario_promo = st.selectbox(
                "Promo code", sorted(orders["promo_code"].unique()))

        submitted = st.form_submit_button(
            "Calculate predictions", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Training or loading models and calculating predictions…"):
            dt_model, rt_model = train_models(orders, restaurants)
            delivery_row = pd.DataFrame([{
                "distance_km": scenario_distance,
                "order_hour": scenario_hour,
                "is_weekend": int(scenario_weekend),
                "weather": scenario_weather,
                "traffic_condition": scenario_traffic,
                "prep_time_min": scenario_prep,
                "cuisine": scenario_cuisine,
                "zone": scenario_zone,
            }])
            delivery_encoded = pd.get_dummies(
                delivery_row, drop_first=True).reindex(
                    columns=dt_model["feature_columns"], fill_value=0)
            predicted_time = float(dt_model["model"].predict(
                delivery_encoded)[0])

            rating_row = pd.DataFrame([{
                "delivery_time_min": predicted_time,
                "distance_km": scenario_distance,
                "weather": scenario_weather,
                "order_value": scenario_value,
                "promo_code": scenario_promo,
                "cuisine": scenario_cuisine,
                "order_hour": scenario_hour,
                "is_weekend": int(scenario_weekend),
            }])
            rating_encoded = pd.get_dummies(
                rating_row, drop_first=True).reindex(
                    columns=rt_model["feature_columns"], fill_value=0)
            predicted_rating = float(np.clip(
                rt_model["model"].predict(rating_encoded)[0], 1, 5))
            st.session_state["whatif_result"] = {
                "delivery_time": predicted_time,
                "rating": predicted_rating,
                "delivery_metrics": dt_model["metrics"]["random_forest"],
                "rating_metrics": rt_model["metrics"]["random_forest"],
                "scenario": {
                    "distance": scenario_distance,
                    "hour": scenario_hour,
                    "weekend": scenario_weekend,
                    "weather": scenario_weather,
                    "traffic": scenario_traffic,
                    "zone": scenario_zone,
                    "prep": scenario_prep,
                    "cuisine": scenario_cuisine,
                    "value": scenario_value,
                    "promo": scenario_promo,
                },
            }

    result = st.session_state.get("whatif_result")
    if result:
        st.subheader("Prediction for submitted scenario")
        result_left, result_right = st.columns(2)
        result_left.metric(
            "Estimated delivery time", f"{result['delivery_time']:.1f} min",
            help="Random Forest estimate. The model's validation MAE is shown below.")
        result_right.metric(
            "Estimated customer rating", f"{result['rating']:.1f} / 5 ★",
            help="Random Forest estimate; rating validation MAE is shown below.")
        detail = result["scenario"]
        day_type = "weekend" if detail["weekend"] else "weekday"
        st.caption(
            f"Scenario: {detail['distance']:.1f} km · {detail['hour']:02d}:00 · "
            f"{day_type} · {detail['weather'].lower()} · {detail['traffic'].lower()} "
            f"traffic · {detail['zone']} · {detail['cuisine']} · "
            f"₹{detail['value']} · promo {detail['promo']}")
        st.info(
            "These are estimates learned from the project's synthetic data. "
            "Traffic, restaurant prep time, and zone are used by the delivery "
            "model; the rating model uses its own trained features and the "
            "estimated delivery time.")

        dt_model, rt_model = train_models(orders, restaurants)
        chart_scenario = detail
        distance_values = np.round(np.arange(0.5, 14.1, 0.5), 1)
        distance_sweep = pd.DataFrame({
            "distance_km": distance_values,
            "order_hour": chart_scenario["hour"],
            "is_weekend": int(chart_scenario["weekend"]),
            "weather": chart_scenario["weather"],
            "traffic_condition": chart_scenario["traffic"],
            "prep_time_min": chart_scenario["prep"],
            "cuisine": chart_scenario["cuisine"],
            "zone": chart_scenario["zone"],
        })
        distance_encoded = pd.get_dummies(
            distance_sweep, drop_first=True).reindex(
                columns=dt_model["feature_columns"], fill_value=0)
        distance_sweep["predicted_delivery_min"] = dt_model["model"].predict(
            distance_encoded)

        rating_sweep = pd.DataFrame({
            "delivery_time_min": distance_sweep["predicted_delivery_min"],
            "distance_km": distance_values,
            "weather": chart_scenario["weather"],
            "order_value": chart_scenario["value"],
            "promo_code": chart_scenario["promo"],
            "cuisine": chart_scenario["cuisine"],
            "order_hour": chart_scenario["hour"],
            "is_weekend": int(chart_scenario["weekend"]),
        })
        rating_encoded = pd.get_dummies(
            rating_sweep, drop_first=True).reindex(
                columns=rt_model["feature_columns"], fill_value=0)
        rating_sweep["predicted_rating"] = np.clip(
            rt_model["model"].predict(rating_encoded), 1, 5)

        chart_left, chart_right = st.columns(2)
        with chart_left:
            delivery_fig = px.line(
                distance_sweep, x="distance_km", y="predicted_delivery_min",
                markers=True, title="Predicted delivery time by distance",
                labels={"distance_km": "Distance (km)",
                        "predicted_delivery_min": "Estimated minutes"})
            delivery_fig.add_vline(
                x=chart_scenario["distance"], line_dash="dash",
                annotation_text=f"Your order: {chart_scenario['distance']:.1f} km")
            delivery_fig.update_layout(height=380)
            st.plotly_chart(delivery_fig, use_container_width=True)
        with chart_right:
            rating_fig = px.line(
                rating_sweep, x="distance_km", y="predicted_rating",
                markers=True, title="Predicted rating by distance",
                labels={"distance_km": "Distance (km)",
                        "predicted_rating": "Estimated rating (stars)"})
            rating_fig.add_vline(
                x=chart_scenario["distance"], line_dash="dash",
                annotation_text=f"Your order: {chart_scenario['distance']:.1f} km")
            rating_fig.update_yaxes(range=[1, 5])
            rating_fig.update_layout(height=380)
            st.plotly_chart(rating_fig, use_container_width=True)
        st.caption(
            "These charts vary delivery distance while holding the other "
            "submitted scenario values constant.")
    else:
        st.info("Choose the scenario values above and select **Calculate predictions**.")

    with st.expander("Model validation details"):
        if result:
            dt_metrics = result["delivery_metrics"]
            rt_metrics = result["rating_metrics"]
            st.markdown(
                f"- Delivery-time model: MAE **{dt_metrics['MAE']} min**, "
                f"R² **{dt_metrics['R2']}**\n"
                f"- Rating model: MAE **{rt_metrics['MAE']} stars**, "
                f"R² **{rt_metrics['R2']}**")
        else:
            st.write("Submit a scenario to see the validation metrics.")

# ================================================================== LIVE tab
# Real-time layer: orders streamed by live_producer.py are tailed from
# data/live_orders.jsonl and merged with today's slice of the historical data
# for rolling KPIs. Refresh is user-triggered so the rest of the dashboard
# does not rerun continuously while the Live tab is not being viewed.
with tab_live:
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
        "Select Refresh to load the latest events.")
    st.button("Refresh live data", key="refresh_live_data")

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
            live_map = live_rest.reset_index()
            fig = px.scatter_geo(
                live_map, lat="lat", lon="lon", size="live_orders",
                color="live_orders", color_continuous_scale="OrRd",
                size_max=24, opacity=0.8, fitbounds="locations",
                projection="mercator", height=400, hover_name="zone",
                hover_data={"restaurant_id": True, "live_orders": True,
                            "lat": False, "lon": False},
                labels={"live_orders": "Live orders"},
                title="Live orders by restaurant location")
            fig.update_geos(
                showland=True, landcolor="#eef2f7", showocean=True,
                oceancolor="#e8f3f8", showcountries=True,
                countrycolor="#cbd5e1", showcoastlines=False)
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

    # Coordinate-based geographic plot avoids external map-tile dependencies.
    rest_orders = (orders_f.groupby("restaurant_id")
                   .agg(orders=("order_id", "count"),
                        revenue=("order_value", "sum"))
                   .join(restaurants.set_index("restaurant_id")))
    restaurant_map = rest_orders.reset_index()
    fig = px.scatter_geo(
        restaurant_map, lat="lat", lon="lon", size="orders", color="rating",
        color_continuous_scale="RdYlGn", size_max=18, opacity=0.75,
        fitbounds="locations", projection="mercator", height=520,
        hover_name="zone",
        hover_data={"restaurant_id": True, "orders": True,
                    "rating": ":.1f", "lat": False, "lon": False},
        labels={"rating": "Restaurant rating", "orders": "Orders"},
        title="Restaurant Locations — size = orders, color = rating")
    fig.update_geos(
        showland=True, landcolor="#eef2f7", showocean=True,
        oceancolor="#e8f3f8", showcountries=True,
        countrycolor="#cbd5e1", showcoastlines=False)
    right.plotly_chart(fig, use_container_width=True)
    right.caption(
        "Each marker is a restaurant. Larger markers mean more orders; "
        "color shows its rating. Select a marker for its zone and details.")

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
