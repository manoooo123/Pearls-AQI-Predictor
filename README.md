# Pearls AQI Predictor

An AI-powered environmental intelligence platform predicting Air Quality Index (AQI) for major urban centers (Lahore, Islamabad, Faisalabad) across 24-hour, 48-hour, and 72-hour forecast horizons.

## Features

- **Multi-Horizon Forecasting**: Machine learning models predicting 24h, 48h, and 72h AQI.
- **Live Telemetry & Ingestion**: OpenAQ v3 API sensor ingestion combined with Open-Meteo weather telemetry.
- **Interactive Streamlit Dashboard**: Real-time telemetry, model benchmarks, SHAP explainability, and user alerts.
- **REST API Backend**: Flask endpoints for authentication, forecasting, and telemetry access.
- **Feature Store & Pipeline**: Automated feature engineering, validation, and historical parquet storage.

## Project Structure

```
Pearls-AQI-Predictor/
├── streamlit_app.py         # Primary Streamlit Dashboard entry point
├── main.py                  # Unified CLI orchestrator
├── app/                     # Flask REST API backend
├── config/                  # Configuration settings & paths
├── utils/                   # Database, security, and feature store utilities
├── models/                  # Trained ML models (.joblib)
├── feature_pipeline/        # Ingestion, validation & feature store refresh
├── training_pipeline/       # Training, evaluation & backtesting scripts
├── explainability/          # SHAP explainability analysis
├── reports/                 # Quality reports, metrics & charts
├── tests/                   # Automated pytest suite
├── requirements.txt         # Minimal production dependencies
└── .streamlit/config.toml   # Streamlit UI configuration
```

## Quick Start (Local Setup)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/manoooo123/Pearls-AQI-Predictor.git
   cd Pearls-AQI-Predictor
   ```

2. **Set up virtual environment & install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in required keys:
   ```bash
   cp .env.example .env
   ```

4. **Launch Streamlit Dashboard**:
   ```bash
   streamlit run streamlit_app.py
   ```
   Or via the CLI launcher:
   ```bash
   python main.py --mode app
   ```

5. **Run Test Suite**:
   ```bash
   python -m pytest
   ```

## Deploying to Streamlit Community Cloud

1. Push your repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io).
3. Click **New App**, select your repository (`Pearls-AQI-Predictor`), branch (`main`).
4. Set **Main file path** to `streamlit_app.py`.
5. Under **Advanced Settings** -> **Secrets**, add any required secrets (e.g. `OPENAQ_API_KEY`).
6. Click **Deploy**.
