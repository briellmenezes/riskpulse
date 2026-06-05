◈ RISKPULSE
Institutional Portfolio Risk Scoring Engine
ML-Driven Quantitative Analysis + RAG-Augmented AI Synthesis

RISKPULSE is a high-fidelity analytics tool designed to bridge the gap between backward-looking machine learning models and forward-looking market intelligence. Built for portfolio managers and risk analysts, it evaluates ETFs across four asset classes using a Random Forest regressor and synthesizes results using Gemini  2.0 Flash.

Prerequisites
Before running the engine, ensure you have Python 3.9+ installed. You will also need a Gemini API Key from Google AI Studio.


1. Install Dependencies
This project relies on streamlit for the UI, yfinance for live market data, and scikit-learn for the ML backend.

Bash
pip install streamlit pandas numpy yfinance google-genai scikit-learn joblib

Execution
To launch the dashboard, run the following command in your terminal:

Bash
streamlit run final.py
The app will automatically open in your default browser at http://localhost:8501.

System Architecture
The tool operates on a Three-Layer Risk Stack:

Quantitative Layer: Fetches 16 years of historical data via yfinance. It computes 6 core metrics: Annualized Volatility, Max Drawdown, Sharpe Ratio, Annualized Return, Beta, and VIX Correlation.

ML Inference Layer: Processes features through a pre-trained Random Forest Regressor (CV R²: 0.807) to generate a proprietary Risk Score (0-100).

AI Synthesis Layer (RAG): * Stage 1 (Retrieval): Gemini retrieves real-time market context and news for the specific ticker.

Stage 2 (Synthesis): Gemini reconciles the ML score with the qualitative news and the User Mandate (provided via the UI text box) to create a final institutional memo.

Project Structure
app.py: The main Streamlit application logic and UI.

model.pkl: Trained Random Forest model.

scaler.pkl: Saved MinMaxScaler for feature normalization.

secrets.py: (Excluded from Git) Should contain GEMINI_API_KEY = "your_key".