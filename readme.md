# Advanced Snowfall Prediction Model

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The **Advanced Snowfall Prediction Model** is a Python-based application that uses machine learning (Random Forest) and deep learning (MLP neural networks via PyTorch) to forecast snowfall events and seasonal totals. It integrates historical weather data from Open-Meteo, climate teleconnection indices (e.g., ENSO/ONI, AO, NAO, PNA, PDO, QBO, SSN, MJO) from NOAA and other sources, and short-term forecasts from the National Weather Service (NWS). The model accumulates data over time for improved accuracy and serves predictions via a lightweight web server on a static HTML page.

Key capabilities:
- Predicts the **first snowfall date** (binary classification with Bayesian fusion for robustness).
- Estimates **seasonal snowfall totals** (regression).
- Displays visualizations: Historical yearly snowfall trends and first snowfall DOYs.
- Includes a **real-time countdown timer** to the next predicted snowfall.
- Updates daily, with predictions prioritizing NWS for imminent events.

Designed for snowy regions (e.g., Lansing, MI by default), it retrains models daily on growing datasets.

## Features

- **Data Accumulation**: Fetches and stores historical daily weather (temp, snowfall, precipitation) from 1940 onward; appends new data daily.
- **Climate Drivers**: Incorporates 8+ teleconnections (ONI, AO, NAO, PNA, PDO, QBO, SSN, MJO amplitude) for long-term patterns.
- **ML/DL Hybrid**: Random Forest for baseline CV evaluation; PyTorch MLPs for predictions.
- **Short-Term Forecasting**: Integrates NWS API for 7-day quantitative snow forecasts.
- **Web Interface**: Self-hosted server at `http://localhost:8080` with:
  - Current drivers table.
  - Predictions with confidence.
  - Live countdown timer (days/hours/minutes/seconds).
  - Graphs: Yearly totals and first snow DOYs (historical + predicted).
  - Last/next update timestamps.
- **Error Resilience**: Handles API failures, NaNs, and retries; logs everything.
- **Configurable**: Set location via `config.json` (e.g., city, ZIP, lat/long).

## Installation

1. **Prerequisites**:
   - Python 3.8+.
   - Git (to clone the repo).

2. **Clone the Repository**:
   ```
   git clone <repo-url>  # Or download as ZIP
   cd advanced-snow-model
   ```

3. **Install Dependencies**:
   The script uses standard libraries + a few extras. Install via pip:
   ```
   pip install numpy requests pandas scikit-learn torch matplotlib scipy
   ```
   - No additional installs needed for core functionality (uses built-in `http.server`).

4. **Verify Setup**:
   - Ensure internet access for API calls (Open-Meteo, NOAA, NWS).
   - Run the script to auto-generate `config.json` if missing.

## Usage

1. **Run the Script**:
   ```
   python test.py
   ```
   - Initial run fetches ~85 years of data (may take 5-10 min; subsequent runs are fast).
   - Starts web server on port 8080 (access: http://localhost:8080).
   - Runs daily predictions; sleeps 24 hours between cycles.

2. **Configuration**:
   - Edit `config.json`:
     ```json
     {"location": "Lansing, MI"}  # Or "48933" (ZIP) or "42.7338,-84.5546" (lat,long)
     ```
   - Restart script for changes.

3. **View Predictions**:
   - Open browser to http://localhost:8080.
   - Refresh page to see updates (auto-refreshes not implemented; manual or add JS if needed).
   - Countdown timer updates live; graphs regenerate per run.

4. **Stopping**:
   - Ctrl+C to interrupt (server stops gracefully).

## How It Works

1. **Data Fetching**:
   - Historical: Open-Meteo archive (1940+; chunks by year).
   - Drivers: NOAA/FU Berlin/SIDC APIs (monthly/yearly indices).
   - Current: Open-Meteo forecast + NWS grid data.

2. **Processing**:
   - Identifies first snow days (temp ≤32°F, snow >0.1in).
   - Features: Cyclic DOY + scaled drivers + precip proxy.
   - Stores in `historical_weather.csv` and `teleconnections.json`.

3. **Modeling**:
   - **First Snow**: MLP classifier (PyTorch) + Bayesian fusion with historical median.
   - **Seasonal Total**: MLP regressor.
   - Retrains daily for incremental learning.

4. **Output**:
   - Logs to console/`advanced_snow_model.log`.
   - HTML page with tables, graphs (matplotlib PNGs), and JS countdown.

5. **Accuracy**:
   - CV AUC ~0.93 for first snow; MAE ~7.5in for totals (based on Lansing data).
   - Improves with more data.

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Array operations |
| requests | API calls |
| pandas | Data handling |
| scikit-learn | Scaling, CV, baselines |
| torch | DL models |
| matplotlib | Plots |
| scipy | Optimization (unused but imported) |

No extras needed for web server.

## Troubleshooting

- **Port Conflict**: Change `PORT = 8080` in `run_server()`.
- **API Errors**: Check logs; fallback to defaults if offline.
- **No Data**: Initial fetch may fail on old dates—script retries daily.
- **Graph Missing**: Ensure matplotlib backend works; delete PNGs to regenerate.
- **Countdown Wrong**: Verify predicted date in logs.

## Contributing

Fork, PR improvements (e.g., more drivers, NN architectures). Report issues for API changes.

## License

MIT License. See LICENSE file for details.

---

*Built with ❤️ by xAI-inspired Grok. Last updated: Nov 3, 2025.*
