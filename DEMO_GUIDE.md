# 🎤 Demo & Presentation Guide

How to present this project to teachers/examiners — a 10-minute script, the
numbers to memorize, and likely viva questions with answers.

---

## 1. Opening line (30 sec)

> "I built an end-to-end Big Data Analytics project on food-delivery data:
> a reproducible 50,000-order dataset, a full exploratory analysis with pandas,
> predictive models for delivery time and customer ratings, and an interactive
> Streamlit dashboard that lets a stakeholder filter and run what-if scenarios
> live."

## 2. Objectives (30 sec)

- **Exploratory:** popular cuisines, busiest restaurants, high-demand zones, peak hours
- **Diagnostic:** which factors drive delivery time and ratings
- **Predictive:** Random Forest models + a live what-if simulator
- **Deliverable:** interactive dashboard with filters

## 3. Demo script (10 minutes)

### Step 1 — The data (1–2 min)
Open `generate_data.py`. Point out:
- 3 tables: `customers` (5,000), `restaurants` (800), `orders` (50,000) — a relational schema
- It's **synthetic but realistic**: seeded RNG → fully reproducible; realistic correlations planted (distance → time, rain/peak → delay, delay → low rating)
- Delivery-log fields: traffic condition, driver id, delivery fee, promo codes

### Step 2 — The analysis layer (1–2 min)
Run `python analysis.py` in a terminal. Show the console report and say:
> "Every number the dashboard shows comes from this reusable pandas layer —
> group-bys, correlation analysis, and bucketed comparisons. It runs standalone
> for quick checks."

### Step 3 — Dashboard tour (4–5 min)
Run `streamlit run dashboard.py`. Tour the tabs **in this order**, narrating one finding each:

| Tab | What to say |
|---|---|
| 📈 Overview | KPI cards; monthly trend with the December festive bump; weekends ~34% busier |
| 🍽️ Cuisines | North Indian + South Indian + Chinese + Biryani ≈ 51% of orders; use sidebar filters live |
| 📍 Locations | Koramangala & Indiranagar lead; peripheral zones are slower and rated lower — a logistics gap, not a food-quality gap |
| ⏰ Peak Hours | Twin peaks at lunch (12–14) and dinner (19–22); heatmap shows weekend evenings darkest |
| 🚚 Factors | The money slide: ratings collapse from 4.0★ (<25 min) to 1.4★ (>60 min); OLS trend shows time vs distance by weather |
| 🤖 What-If | **The wow moment.** Drag distance to 12 km, switch weather to Rain, traffic to Heavy → prediction jumps live. Explain the model reacts instantly because predictions come from a trained Random Forest |

### Step 4 — The modeling (1–2 min)
Run `python modeling.py`. Show the metrics and explain:

- **Delivery-time model:** Random Forest, R² 0.987, MAE 1.39 min — beats the
  mean-baseline (12.4 min) by ~9×; linear regression shown as benchmark
- **Rating model:** R² 0.52, MAE 0.61★ — honest about the ceiling (ratings are
  integers 1–5, mostly 2–4)
- **Feature importances agree with the EDA** — distance dominates time; time
  dominates rating. Two independent methods telling the same story.

### Step 5 — Business insights (1 min)
Close with 3 recommendations:
1. Position/incentive riders before 12:00 and 19:00 surges
2. Distance-based promise times instead of a flat SLA
3. Rain-mode pricing/bonuses — rain costs ~9 min and ~0.7★

## 4. Numbers to memorize

| Fact | Value |
|---|---|
| Orders / customers / restaurants | 50,000 / 5,000 / 800 |
| Total revenue | ₹94.5 lakh (avg ₹189/order) |
| Avg delivery time / rating | 41 min / ~3.0★ |
| Cancellation rate | 6.1% |
| Distance ↔ time correlation | r ≈ 0.81 |
| Time ↔ rating correlation | r ≈ −0.69 |
| Peak-hour penalty | +9 min (43–44 vs ~35) |
| Rain penalty | +9 min and −0.7★ |
| Heavy-traffic penalty | +4 min |
| Rating across time bins | 4.0★ (<25 min) → 1.4★ (>60 min) |
| RF delivery model | MAE 1.39 min, R² 0.987 (baseline 12.4) |
| RF rating model | MAE 0.61★, R² 0.52 |

## 5. Likely viva questions (and good answers)

**Q: Why synthetic data?**
> Real Kaggle data has inconsistent schemas and unknown quality. Generating data
> with a seeded RNG makes the whole pipeline reproducible and lets me validate
> that my analysis *rediscovers* planted patterns — proof the methodology works.
> The loader is isolated in one function, so real data can be swapped in.

**Q: Is R² = 0.987 suspiciously high?**
> Yes, for real data it would be. Synthetic data has strong signal by design.
> The honest framing: it shows the model captures the data-generating process;
> on real data I'd expect lower R² but the same feature ranking.

**Q: Why Random Forest and not XGBoost/neural nets?**
> RF is robust to outliers, needs little tuning, gives feature importances for
> interpretability, and benchmarks well here (linear is included for
> comparison). Gradient boosting is a natural next step.

**Q: How did you validate models?**
> 80/20 train-test split, MAE/RMSE/R², and comparison against a naive baseline
> (predicting the mean). No leakage: the rating model uses only information
> available at/after delivery, clearly separated.

**Q: Where is the "big data" in a BDA project?**
> 50K rows × 18 columns is beyond spreadsheet scale and demonstrates the full
> pipeline. The pandas layer is intentionally swappable: the same group-by/
> aggregate logic maps 1:1 to Spark DataFrames (PySpark) when data outgrows
> one machine — that's the next milestone.

**Q: What was YOUR contribution vs generated?**
> Everything here is scripted from scratch: the generator with its correlation
> design, the analysis layer, models, and dashboard. Be ready to open any file
> and explain a function — e.g. `cuisine_popularity()` or `haversine()`.

## 6. Backup plans

- **Internet/projector fails:** everything runs locally from the repo — laptop demo
- **Streamlit won't start:** `python analysis.py` + `python modeling.py` still show the full pipeline
- **Time cut short:** go straight to the What-If tab — it demonstrates data + model + UI in one screen
- **They want a document:** this README + `modeling.py` output pasted into the report

## 7. Possible "future scope" answers (examiners love these)

- Swap in a real Swiggy/Zomato Kaggle dataset
- PySpark pipeline for out-of-memory scale
- Gradient boosting + SHAP explainability
- Driver-level scorecards and shift optimization
- Deploy the dashboard to Streamlit Cloud for a public link
