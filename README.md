<div align="center">

<img src="Pulse logo.png" alt="PULSE Logo" width="350"/>

# PULSE
### Predictive Urban Logistics & Safety Engine

*Turn historical crash data, live weather, and road conditions into actionable safety intelligence —*
*with a built-in simulator to plan before harm occurs.*

</div>

---

## 🌟 Overview

PULSE is a decision-support system for urban safety and EMS resource planning. It doesn't just tell you where accidents are likely — it helps city planners, dispatchers, and analysts figure out what to do about it, and what happens if they do something different.

---

## 💡 Problem

Most safety tools stop at the heatmap. PULSE is designed around the next question every operator actually asks:

<div align="center">

**"So what do we do about it?"**

</div>

---

## ⚙️ How It Works

1. 📊 **Risk Forecasting** — Machine learning models trained on Madison crash data, weather, and road conditions generate short-term risk predictions across Madison, gridded by location and time.
   
2. 🚑 **EMS Deployment Recommendations** — Based on current forecasts, PULSE suggests optimal pre-positioning of limited emergency resources to minimize expected response time and harm.
   
3. 💬 **Explainable Outputs** — Every forecast comes with a plain-language summary of the driving factors: weather, time of day, traffic volume, recent incidents.

---

## 🧑‍💻 Tech Stack

**Data**
- WisDOT Crash Records (statewide, severity + coordinates)
- NOAA / Open-Meteo (hourly weather history)
- OpenStreetMap via OSMnx (road network)
- US Census ACS (demographics, income, vehicle access)
- Madison Open Data (fire stations, EMS locations)

**Program**
- scikit-learn / XGBoost — crash risk model
- OR-Tools — facility location & resource allocation
- PyDeck — data visualization
- Python, FastAPI
- PostGIS for spatial queries
- React + Leaflet (interactive risk map)
- Recharts (confidence intervals, scenario comparisons)

---

## 🚀 Roadmap to Scaling

- Real-time 911 dispatch integration
- Multi-city deployment beyond Madison
- Reinforcement learning for adaptive EMS routing
- Public-facing risk transparency dashboard

---

<div align="center">

*Built for MadData 2026 — Madison, WI* 
<br>
*By Abhinav, Niyati, Sahana*

</div>
