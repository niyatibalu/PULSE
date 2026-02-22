# PULSE: Predictive Urban Logistics & Safety Engine

Turn historical crash data, live weather, and road conditions into actionable safety intelligence — with a built-in simulator to plan before harm occurs.

---

## Overview

PULSE is a decision-support system for urban safety and EMS resource planning. It doesn't just tell you where accidents are likely — it helps city planners, dispatchers, and analysts figure out what to do about it, and what happens if they do something different.

---

## Problem

Most safety tools stop at the heatmap. PULSE is designed around the next question every operator actually asks: *"So what do we do about it?"*

---

## How It Works

1. **Risk Forecasting** — Machine learning models trained on Madison crash data, weather, and road conditions generate short-term risk predictions across Madison, gridded by location and time.

2. **What-If Simulator** — Adjust conditions and proposed interventions (speed limit changes, lighting improvements, event traffic) and see how predicted risk shifts. This is a core feature of PULSE — not just "here's the risk," but "here's what changes if you act."

3. **EMS Deployment Recommendations** — Based on current forecasts, PULSE suggests optimal pre-positioning of limited emergency resources to minimize expected response time and harm.

4. **Equity View** — Overlays census data to surface neighborhoods with persistently high risk and lower service coverage, so resource decisions can account for fairness, not just efficiency.

5. **Explainable Outputs** — Every forecast comes with a plain-language summary of the driving factors: weather, time of day, traffic volume, recent incidents.

---

## Key Features

**Prediction**
- Gradient boosted ML model trained on WisDOT statewide crash records
- Weather-integrated forecasts using NOAA/Open-Meteo historical data
- Grid-level risk scores across Madison with confidence intervals

**Simulation**
- Policy scenario builder: test interventions before deploying them
- Event-aware modeling: Badgers games, festivals, construction zones
- Side-by-side risk comparison across scenarios

**Optimization**
- OR-Tools based EMS pre-positioning under resource constraints
- Equity-constrained allocation with census-based fairness weights
- Travel-time routing via OpenStreetMap road network

**Explainability**
- LLM-generated natural language summaries per forecast
- Feature attribution per prediction
- Audit trail for every recommendation

---

## Tech Stack

**Data**
- WisDOT Crash Records (statewide, severity + coordinates)
- NOAA / Open-Meteo (hourly weather history)
- OpenStreetMap via OSMnx (road network)
- US Census ACS (demographics, income, vehicle access)
- Madison Open Data (fire stations, EMS locations)

**ML & Optimization**
- scikit-learn / XGBoost — crash risk model
- OR-Tools — facility location & resource allocation
- NetworkX / OSMnx — graph-based routing

**Backend**
- Python, FastAPI
- PostGIS / GeoPandas for spatial queries

**Frontend**
- React + Leaflet (interactive risk map)
- Recharts (confidence intervals, scenario comparisons)

---

## Roadmap to scaling

- Real-time 911 dispatch integration
- Multi-city deployment beyond Madison
- Reinforcement learning for adaptive EMS routing
- Public-facing risk transparency dashboard

---

*Built for MadData 2025 — Madison, WI*
