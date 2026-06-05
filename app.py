# ============================================================
# RISKPULSE — Institutional Portfolio Risk Scoring Tool
# Stack: Streamlit + Random Forest (scikit-learn) + Gemini AI
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import yfinance as yf
from google import genai

# ── API key — loaded from secrets.py (never commit this file) ──
# For Streamlit Cloud deployment, add GEMINI_API_KEY to
# App Settings → Secrets as: GEMINI_API_KEY = "your-key"
try:
    from secrets import GEMINI_API_KEY
except ImportError:
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        GEMINI_API_KEY = ""

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="RISKPULSE | Portfolio Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Instrument+Sans:wght@400;500;600&display=swap');

:root {
    --berry:    #6B1A3A;
    --pink:     #E8538A;
    --blush:    #F2A7C3;
    --black:    #0D0D0D;
    --charcoal: #1A1A1A;
    --graphite: #2C2C2C;
    --white:    #FFFFFF;
    --muted:    #888888;
    --border:   rgba(232,83,138,0.18);
}

html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif; background-color: var(--black); color: var(--white); }

.stApp {
    background: var(--black);
    background-image:
        radial-gradient(ellipse 80% 50% at 10% 0%, rgba(107,26,58,0.35) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 100%, rgba(232,83,138,0.15) 0%, transparent 55%);
}

[data-testid="stSidebar"] { background: var(--charcoal) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--white) !important; }

.rp-logo { font-family: 'DM Serif Display', serif; font-size: 2.6rem; letter-spacing: -1px; background: linear-gradient(135deg, var(--pink) 0%, var(--blush) 60%, var(--white) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1; display: inline-block; }
.rp-tag { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--pink); letter-spacing: 3px; border: 1px solid var(--border); padding: 2px 8px; border-radius: 2px; vertical-align: middle; margin-left: 12px; }
.rp-subtitle { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; margin-bottom: 32px; margin-top: 4px; }

.rp-card { background: var(--charcoal); border: 1px solid var(--border); border-radius: 8px; padding: 24px; margin-bottom: 16px; position: relative; overflow: hidden; }
.rp-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--berry), var(--pink), var(--blush)); }
.rp-card-label { font-family: 'DM Mono', monospace; font-size: 0.6rem; letter-spacing: 3px; color: var(--pink); text-transform: uppercase; margin-bottom: 12px; }

.score-ring-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px 0; }
.score-number { font-family: 'DM Serif Display', serif; font-size: 5.5rem; line-height: 1; background: linear-gradient(135deg, var(--white) 0%, var(--blush) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.score-denom { font-family: 'DM Mono', monospace; font-size: 1rem; color: var(--muted); }
.score-category { font-family: 'DM Mono', monospace; font-size: 0.75rem; letter-spacing: 4px; text-transform: uppercase; padding: 6px 20px; border-radius: 2px; margin-top: 10px; }
.cat-low    { background: rgba(40,200,100,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
.cat-medium { background: rgba(250,180,0,0.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.cat-high   { background: rgba(232,83,138,0.15); color: var(--pink); border: 1px solid var(--border); }

.gauge-wrap { margin: 16px 0 8px; }
.gauge-track { height: 6px; background: var(--graphite); border-radius: 3px; overflow: hidden; }
.gauge-fill  { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #4ade80, #fbbf24, var(--pink)); }
.gauge-labels { display: flex; justify-content: space-between; font-family: 'DM Mono', monospace; font-size: 0.6rem; color: var(--muted); margin-top: 4px; }

.metric-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin: 12px 0; }
.metric-cell { background: var(--graphite); border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 12px 14px; }
.metric-label { font-family: 'DM Mono', monospace; font-size: 0.55rem; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px; }
.metric-value { font-family: 'DM Serif Display', serif; font-size: 1.3rem; color: var(--white); line-height: 1.1; }
.metric-value.pos { color: #4ade80; }
.metric-value.neg { color: var(--pink); }

.feat-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.feat-name { font-family: 'DM Mono', monospace; font-size: 0.62rem; color: var(--muted); width: 120px; flex-shrink: 0; text-transform: uppercase; letter-spacing: 1px; }
.feat-bar-track { flex: 1; height: 4px; background: var(--graphite); border-radius: 2px; overflow: hidden; }
.feat-bar-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--berry), var(--pink)); }
.feat-pct { font-family: 'DM Mono', monospace; font-size: 0.6rem; color: var(--blush); width: 36px; text-align: right; }

.warn-flag { display: flex; align-items: flex-start; gap: 10px; background: rgba(232,83,138,0.08); border: 1px solid rgba(232,83,138,0.3); border-radius: 6px; padding: 10px 14px; margin: 6px 0; }
.warn-icon { font-size: 0.9rem; flex-shrink: 0; margin-top: 1px; }
.warn-text { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--blush); line-height: 1.5; }
.warn-title { color: var(--pink); font-weight: 500; display: block; margin-bottom: 2px; }

.peer-row { display: flex; align-items: center; gap: 12px; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
.peer-ticker { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: var(--white); width: 60px; flex-shrink: 0; font-weight: 500; }
.peer-bar-track { flex: 1; height: 8px; background: var(--graphite); border-radius: 4px; overflow: hidden; }
.peer-bar-fill { height: 100%; border-radius: 4px; }
.peer-score { font-family: 'DM Mono', monospace; font-size: 0.7rem; width: 45px; text-align: right; }
.peer-cat { font-family: 'DM Mono', monospace; font-size: 0.55rem; letter-spacing: 1px; width: 55px; text-align: right; }

.ai-wrap { background: linear-gradient(135deg, rgba(107,26,58,0.25) 0%, rgba(26,26,26,0.8) 100%); border: 1px solid var(--border); border-radius: 8px; padding: 24px; position: relative; }
.ai-wrap::before { content: '◈ AI ASSESSMENT · RAG-AUGMENTED'; font-family: 'DM Mono', monospace; font-size: 0.6rem; letter-spacing: 3px; color: var(--pink); display: block; margin-bottom: 16px; }
.ai-text { font-family: 'Instrument Sans', sans-serif; font-size: 0.875rem; line-height: 1.75; color: rgba(255,255,255,0.85); }

.sidebar-section { font-family: 'DM Mono', monospace; font-size: 0.6rem; letter-spacing: 3px; color: var(--pink); text-transform: uppercase; margin: 20px 0 8px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
.ticker-badge { display: inline-block; font-family: 'DM Mono', monospace; font-size: 1rem; font-weight: 500; background: linear-gradient(135deg, var(--berry), var(--pink)); padding: 4px 14px; border-radius: 3px; letter-spacing: 2px; margin-bottom: 16px; }
.scenario-active { font-family: 'DM Mono', monospace; font-size: 0.6rem; letter-spacing: 2px; color: #fbbf24; background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.3); border-radius: 3px; padding: 3px 10px; display: inline-block; margin-bottom: 12px; }

.stButton > button { background: linear-gradient(135deg, var(--berry), var(--pink)) !important; color: white !important; border: none !important; border-radius: 4px !important; font-family: 'DM Mono', monospace !important; font-size: 0.7rem !important; letter-spacing: 2px !important; text-transform: uppercase !important; padding: 10px 24px !important; width: 100% !important; }
.stButton > button:hover { opacity: 0.85 !important; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea { background: var(--graphite) !important; border: 1px solid var(--border) !important; color: var(--white) !important; border-radius: 4px !important; font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; }
.stSelectbox > div > div { background: var(--graphite) !important; border: 1px solid var(--border) !important; color: var(--white) !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

# Feature importances from trained Random Forest — displayed in UI
# and passed to Gemini so AI explains the model's actual logic
FEATURE_IMPORTANCES = [
    ("Ann. Volatility",  0.5909),
    ("Max Drawdown",     0.3599),
    ("Sharpe Ratio",     0.0152),
    ("Ann. Return",      0.0136),
    ("Beta",             0.0070),
    ("VIX Correlation",  0.0068),
]

# Stress scenarios — feature values derived from percentile ranks
# of the training data distribution (Option B approach).
# Each scenario represents a market regime, not a specific year's raw numbers.
SCENARIOS = {
    "2008 Financial Crisis": {
        "ann_return": -0.355, "ann_vol":      0.369,
        "sharpe":     -0.870, "max_drawdown": -0.590,
        "beta":        1.230, "vix_corr":    -0.130,
        "description": "95th percentile vol/drawdown — peak crisis conditions"
    },
    "COVID Crash (2020)": {
        "ann_return": -0.120, "ann_vol":      0.294,
        "sharpe":     -0.440, "max_drawdown": -0.420,
        "beta":        1.098, "vix_corr":    -0.160,
        "description": "85th percentile — rapid drawdown, VIX spike"
    },
    "2022 Rate Hike Cycle": {
        "ann_return": -0.180, "ann_vol":      0.204,
        "sharpe":     -0.850, "max_drawdown": -0.370,
        "beta":        0.980, "vix_corr":    -0.480,
        "description": "70th percentile vol — bond/equity selloff, rising rates"
    },
    "Bull Market (2019)": {
        "ann_return":  0.135, "ann_vol":      0.154,
        "sharpe":      0.520, "max_drawdown": -0.200,
        "beta":        0.867, "vix_corr":    -0.660,
        "description": "20th percentile vol — calm conditions, strong returns"
    },
}

# Peer groups for benchmarking — scored alongside the main ticker
PEER_GROUPS = {
    "equity":    ["QQQ", "IWM", "VTV", "EFA"],
    "bond":      ["TLT", "HYG", "LQD", "EMB"],
    "commodity": ["GLD", "USO", "DJP", "PDBC"],
    "alt":       ["VNQ", "IYR", "AOR", "AOA"],
}

# Ticker → asset class lookup lists
BOND_TICKERS      = ['AGG','BND','TLT','IEF','SHY','HYG','LQD','EMB','MUB','BNDX']
COMMODITY_TICKERS = ['GLD','SLV','USO','DJP','PDBC','DBA','CPER']
ALT_TICKERS       = ['VNQ','IYR','AOM','AOR','AOA','AOK']


# ============================================================
# MODEL LOADING
# @st.cache_resource — loaded once, reused across all sessions
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

@st.cache_resource
def load_scaler():
    return joblib.load('scaler.pkl')

try:
    model  = load_model()
    scaler = load_scaler()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Ensure model.pkl and scaler.pkl are in the same folder as app.py.")
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_asset_class(ticker: str) -> str:
    """Classify a ticker into equity / bond / commodity / alt."""
    if ticker in BOND_TICKERS:      return 'bond'
    if ticker in COMMODITY_TICKERS: return 'commodity'
    if ticker in ALT_TICKERS:       return 'alt'
    return 'equity'


@st.cache_data(show_spinner=False)
def fetch_etf_features(ticker: str) -> dict | None:
    """
    Download 2010-2026 price history and compute 6 risk features.
    Cached — repeated calls for the same ticker skip the download.
    Returns None if data is missing or too short to be reliable.
    """
    try:
        data = yf.download(ticker, start='2010-01-01', end='2026-01-01',
                           auto_adjust=True, progress=False)['Close']
        if data.empty or len(data) < 100:
            return None

        spy = yf.download('SPY',  start='2010-01-01', end='2026-01-01',
                          auto_adjust=True, progress=False)['Close']
        vix = yf.download('^VIX', start='2010-01-01', end='2026-01-01',
                          auto_adjust=True, progress=False)['Close']

        r     = data.pct_change().dropna()
        spy_r = spy.pct_change().dropna()
        vix_r = vix.pct_change().dropna()

        ann_return = float(r.mean() * 252)
        ann_vol    = float(r.std() * np.sqrt(252))
        sharpe     = float((ann_return - 0.044) / ann_vol) if ann_vol > 0 else 0.0

        cum    = (1 + r).cumprod()
        max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())

        aligned = pd.concat([r, spy_r], axis=1).dropna()
        cov     = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
        beta    = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 1.0

        vix_al   = pd.concat([r, vix_r], axis=1).dropna()
        vix_corr = float(vix_al.corr().iloc[0, 1])

        return dict(ann_return=ann_return, ann_vol=ann_vol, sharpe=sharpe,
                    max_drawdown=max_dd, beta=beta, vix_corr=vix_corr)
    except Exception:
        return None


def predict_score(features: dict, asset_class: str) -> float:
    """
    Scale features with the saved MinMaxScaler then run through Random Forest.
    Column order is enforced via model.feature_names_in_ to prevent mismatch errors.
    Score is clamped to [0, 100].
    """
    row = pd.DataFrame([{
        'ann_return':      features['ann_return'],
        'ann_vol':         features['ann_vol'],
        'sharpe':          features['sharpe'],
        'max_drawdown':    features['max_drawdown'],
        'beta':            features['beta'],
        'vix_corr':        features['vix_corr'],
        'asset_equity':    asset_class == 'equity',
        'asset_bond':      asset_class == 'bond',
        'asset_commodity': asset_class == 'commodity',
        'asset_alt':       asset_class == 'alt',
    }])
    row_scaled = pd.DataFrame(scaler.transform(row), columns=row.columns)
    row_scaled = row_scaled[model.feature_names_in_]
    return float(max(0.0, min(100.0, model.predict(row_scaled)[0])))


def get_warnings(features: dict, score: float) -> list:
    """
    Flag any metrics that cross institutional risk thresholds.
    Returns list of (title, message) tuples — empty list if no flags.
    """
    flags = []
    if features['max_drawdown'] < -0.45:
        flags.append(("EXTREME DRAWDOWN",
            f"Max drawdown of {features['max_drawdown']:.1%} exceeds -45% threshold. Severe capital loss risk."))
    if features['ann_vol'] > 0.30:
        flags.append(("HIGH VOLATILITY",
            f"Annualized vol of {features['ann_vol']:.1%} is significantly above the ~20% market average."))
    if features['sharpe'] < 0:
        flags.append(("NEGATIVE SHARPE",
            f"Sharpe of {features['sharpe']:.2f} — returns do not compensate for risk taken."))
    if features['beta'] > 1.5:
        flags.append(("HIGH BETA",
            f"Beta of {features['beta']:.2f} amplifies market moves by {features['beta']:.1f}x."))
    if score > 75:
        flags.append(("OVERALL RISK FLAG",
            f"Score of {score:.0f}/100 is in the top quartile of risk across the training universe."))
    return flags

def get_similar_etfs(features: dict, asset_class: str, top_n: int = 3) -> list:
    """
    Compute cosine similarity between the queried ticker's feature vector
    and all 53 ETFs in the training universe.
    Returns top_n most similar tickers with their similarity scores.
    This is the embedding/similarity component of the RAG pipeline —
    instead of a vector database, we use the training feature matrix directly.
    """
    # Training universe feature vectors (from your Colab notebook output)
    # These are the scaled feature values for all 53 ETFs
    training_data = {
        'SPY':  [0.142, 0.177,  0.552, -0.337,  1.000, -0.727, False, False, False, False],
        'QQQ':  [0.194, 0.219,  0.684, -0.351,  1.153, -0.694, False, False, False, False],
        'IWM':  [0.106, 0.225,  0.275, -0.411,  1.098, -0.656, False, False, False, False],
        'VTI':  [0.138, 0.181,  0.521, -0.350,  1.014, -0.724, False, False, False, False],
        'VTV':  [0.114, 0.166,  0.420, -0.368,  0.868, -0.667, False, False, False, False],
        'VUG':  [0.171, 0.209,  0.607, -0.356,  1.127, -0.698, False, False, False, False],
        'IVW':  [0.166, 0.201,  0.607, -0.327,  1.095, -0.705, False, False, False, False],
        'IVE':  [0.111, 0.169,  0.397, -0.370,  0.894, -0.671, False, False, False, False],
        'EFA':  [0.083, 0.172,  0.227, -0.342,  0.821, -0.658, False, False, False, False],
        'EEM':  [0.070, 0.204,  0.126, -0.398,  0.866, -0.612, False, False, False, False],
        'VEA':  [0.088, 0.172,  0.256, -0.357,  0.829, -0.663, False, False, False, False],
        'VWO':  [0.071, 0.196,  0.139, -0.364,  0.821, -0.607, False, False, False, False],
        'IEFA': [0.086, 0.172,  0.243, -0.348,  0.818, -0.656, False, False, False, False],
        'AGG':  [0.020, 0.052, -0.458, -0.184,  0.027, -0.011, False, True,  False, False],
        'BND':  [0.021, 0.053, -0.435, -0.186,  0.031, -0.015, False, True,  False, False],
        'TLT':  [0.010, 0.150, -0.229, -0.484, -0.159,  0.156, False, True,  False, False],
        'IEF':  [0.016, 0.066, -0.425, -0.239, -0.061,  0.163, False, True,  False, False],
        'SHY':  [0.016, 0.015, -1.886, -0.057, -0.007,  0.115, False, True,  False, False],
        'HYG':  [0.044, 0.083,  0.006, -0.220,  0.360, -0.512, False, True,  False, False],
        'LQD':  [0.031, 0.084, -0.152, -0.250,  0.127, -0.142, False, True,  False, False],
        'EMB':  [0.038, 0.097, -0.064, -0.287,  0.313, -0.397, False, True,  False, False],
        'GLD':  [0.124, 0.149,  0.538, -0.220,  0.036,  0.007, False, False, True,  False],
        'SLV':  [0.168, 0.274,  0.453, -0.428,  0.336, -0.149, False, False, True,  False],
        'USO':  [-0.034, 0.388, -0.201, -0.928, 0.669, -0.243, False, False, True,  False],
        'DJP':  [0.023, 0.168, -0.125, -0.537,  0.321, -0.277, False, False, True,  False],
        'VNQ':  [0.072, 0.204,  0.137, -0.424,  0.834, -0.473, False, False, False, True],
        'IYR':  [0.072, 0.200,  0.140, -0.423,  0.823, -0.481, False, False, False, True],
        'PDBC': [0.030, 0.178, -0.076, -0.493,  0.323, -0.277, False, False, True,  False],
        'XLK':  [0.216, 0.236,  0.728, -0.336,  1.239, -0.682, False, False, False, False],
        'XLF':  [0.135, 0.218,  0.418, -0.429,  1.043, -0.617, False, False, False, False],
        'XLE':  [0.085, 0.294,  0.139, -0.673,  1.030, -0.449, False, False, False, False],
        'XLV':  [0.106, 0.168,  0.369, -0.284,  0.751, -0.582, False, False, False, False],
        'XLI':  [0.129, 0.195,  0.435, -0.423,  0.980, -0.646, False, False, False, False],
        'XLU':  [0.106, 0.189,  0.326, -0.361,  0.598, -0.324, False, False, False, False],
        'XLP':  [0.081, 0.145,  0.257, -0.245,  0.574, -0.478, False, False, False, False],
        'XLY':  [0.147, 0.214,  0.483, -0.397,  1.086, -0.656, False, False, False, False],
        'AOM':  [0.055, 0.078,  0.144, -0.200,  0.383, -0.656, False, False, False, True],
        'AOR':  [0.074, 0.105,  0.284, -0.229,  0.555, -0.707, False, False, False, True],
        'AOA':  [0.093, 0.134,  0.363, -0.284,  0.725, -0.729, False, False, False, True],
        'AOK':  [0.046, 0.065,  0.028, -0.189,  0.294, -0.582, False, False, False, True],
        'SPLV': [0.092, 0.154,  0.314, -0.363,  0.700, -0.542, False, False, False, False],
        'USMV': [0.107, 0.144,  0.437, -0.331,  0.734, -0.627, False, False, False, False],
        'SPHB': [0.162, 0.279,  0.424, -0.468,  1.390, -0.649, False, False, False, False],
        'HDV':  [0.090, 0.156,  0.294, -0.370,  0.722, -0.565, False, False, False, False],
        'VGK':  [0.091, 0.188,  0.248, -0.372,  0.861, -0.627, False, False, False, False],
        'VPL':  [0.082, 0.169,  0.224, -0.339,  0.786, -0.647, False, False, False, False],
        'MUB':  [0.024, 0.047, -0.433, -0.137,  0.057, -0.032, False, True,  False, False],
        'BNDX': [0.022, 0.040, -0.555, -0.162,  0.021, -0.011, False, True,  False, False],
        'DBA':  [0.023, 0.130, -0.165, -0.478,  0.162, -0.203, False, False, True,  False],
        'CPER': [0.105, 0.343,  0.177, -0.420,  0.468, -0.205, False, False, True,  False],
        'MTUM': [0.152, 0.202,  0.532, -0.341,  1.028, -0.681, False, False, False, False],
        'QUAL': [0.137, 0.179,  0.519, -0.341,  0.991, -0.704, False, False, False, False],
        'SIZE': [0.117, 0.183,  0.399, -0.392,  0.958, -0.649, False, False, False, False],
    }

    # Build query vector from current features
    query_vec = np.array([
        features['ann_return'], features['ann_vol'],   features['sharpe'],
        features['max_drawdown'], features['beta'],    features['vix_corr'],
        asset_class == 'equity',  asset_class == 'bond',
        asset_class == 'commodity', asset_class == 'alt'
    ], dtype=float)

    # Compute cosine similarity against all training ETFs
    similarities = {}
    for etf, vec in training_data.items():
        train_vec = np.array(vec, dtype=float)
        # Cosine similarity: dot product / (magnitude * magnitude)
        dot     = np.dot(query_vec, train_vec)
        norm_q  = np.linalg.norm(query_vec)
        norm_t  = np.linalg.norm(train_vec)
        if norm_q > 0 and norm_t > 0:
            similarities[etf] = dot / (norm_q * norm_t)

    # Return top_n most similar, excluding the ticker itself if it's in training
    sorted_etfs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [(etf, round(sim, 3)) for etf, sim in sorted_etfs[:top_n + 1]
            if etf != (ticker_input if 'ticker_input' in dir() else '')][:top_n]

def get_ai_explanation(
    ticker: str,
    score: float,
    category: str,
    features: dict,
    description: str,
    scenario_name: str | None = None,
    asset_class: str = 'equity'
) -> str:
    """
    TWO-STAGE RAG AI PIPELINE:

    Stage 1 — Retrieval: ask Gemini to recall recent market developments,
              analyst sentiment, and qualitative risk signals for the ticker.
              This leverages Gemini's built-in knowledge base as a retrieval
              source — more reliable than external news APIs for a deployed app.

    Stage 2 — Generation: synthesize the retrieved qualitative context with
              the ML model output (score, features, importances) into a
              structured 3-paragraph institutional risk assessment.

    The prompt adapts by asset class so bonds get analyzed through a
    duration/credit lens, commodities through supply/demand, etc.
    All edge cases (no ticker, scenario mode, manual input) are handled.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)


    # ── Embedding similarity — find closest training ETFs ──
    similar_etfs = get_similar_etfs(features, asset_class)
    similar_str  = ", ".join([f"{etf} (similarity: {sim:.2f})"
                            for etf, sim in similar_etfs])
    # ── Stage 1: Retrieve qualitative market context ──
    # Gemini's knowledge base acts as our retrieval source.
    # We ask for specific, factual, risk-relevant context — not generic summaries.
    news_context = "No specific market context available. Analysis based on quantitative data only."

    if ticker:
        try:
            retrieval_prompt = f"""You are a financial analyst with access to recent market data.
For the ETF or asset {ticker}, provide 3-4 specific bullet points covering:
- Recent price action or performance trend
- Any notable analyst upgrades, downgrades, or sentiment shifts
- Macro or sector-level risks currently affecting this asset
- Any specific news events that have impacted or could impact risk

Be factual and specific. If you are uncertain about recent events, say so.
Do not be generic. Focus on risk-relevant information only."""

            retrieval_response = client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=retrieval_prompt
            )
            news_context = retrieval_response.text.strip()
        except Exception as e:
            news_context = f"Market context retrieval failed ({str(e)}). Quantitative analysis only."

    # ── Stage 2: Asset-class-specific risk lens ──
    # Each asset class has fundamentally different risk drivers.
    # We tell Gemini which lens to apply so the output is substantive, not generic.
    asset_context = {
        'bond': (
            "duration risk, credit spread widening, interest rate sensitivity, and yield curve dynamics",
            "investment-grade bonds typically score 15-35; high yield 30-50"
        ),
        'commodity': (
            "supply/demand imbalances, geopolitical exposure, USD sensitivity, and futures roll costs",
            "commodities typically score 35-70 depending on volatility profile"
        ),
        'alt': (
            "correlation to equities, liquidity risk, interest rate sensitivity, and sector concentration",
            "alternatives typically score 30-55"
        ),
        'equity': (
            "earnings volatility, sector concentration, macro beta, and valuation sensitivity",
            "broad equity ETFs score 45-65; sector/factor ETFs 55-80"
        ),
    }
    risk_lens, benchmark = asset_context.get(asset_class, asset_context['equity'])

    # Identify primary quantitative red flag to anchor the narrative
    red_flags = []
    if features['max_drawdown'] < -0.45:
        red_flags.append(f"extreme max drawdown of {features['max_drawdown']:.1%}")
    if features['ann_vol'] > 0.30:
        red_flags.append(f"high annualized volatility of {features['ann_vol']:.1%}")
    if features['sharpe'] < 0:
        red_flags.append(f"negative Sharpe ratio of {features['sharpe']:.2f}")
    if features['beta'] > 1.5:
        red_flags.append(f"elevated beta of {features['beta']:.2f}")
    primary_flag = red_flags[0] if red_flags else f"annualized volatility of {features['ann_vol']:.1%}"

    # Scenario framing — if stress test active, tell Gemini to frame accordingly
    scenario_context = ""
    if scenario_name:
        desc = SCENARIOS[scenario_name].get("description", scenario_name)
        scenario_context = (
            f"\nSTRESS SCENARIO ACTIVE: {scenario_name} ({desc}). "
            "These features reflect simulated crisis conditions, not live market data. "
            "Frame the assessment as a stress test result and note what the score implies "
            "about portfolio resilience under this regime."
        )

    ticker_label = ticker if ticker else "Custom Manual Portfolio"

    # ── Stage 2: Main synthesis prompt ──
    prompt = f"""You are a senior portfolio risk strategist at BNY Mellon's CAO division.
Your role is to synthesize quantitative ML model output with qualitative market intelligence.
{scenario_context}

ASSET PROFILE:
- Ticker: {ticker_label}
- Asset Class: {asset_class.upper()} — analyze through the lens of {risk_lens}
- Risk Score: {score:.1f}/100 ({category})
- Benchmark context: {benchmark}
- Primary quantitative driver: {primary_flag}

QUANTITATIVE METRICS (Random Forest, CV R²=0.807, trained on 53 ETFs 2010-2026):
- Annualized Return:  {features['ann_return']:.2%}
- Annualized Vol:     {features['ann_vol']:.2%}    [59% of model weight]
- Sharpe Ratio:       {features['sharpe']:.2f}
- Max Drawdown:       {features['max_drawdown']:.2%} [36% of model weight]
- Beta to S&P 500:    {features['beta']:.2f}
- VIX Correlation:    {features['vix_corr']:.2f}

QUALITATIVE MARKET INTELLIGENCE (retrieved via RAG):
{news_context}

USER CONTEXT: {description if description else 'None provided'}

Write approx 3 sections. Use headers, bullet points, whatever you need to help investor understand. Do not use markdown!!!:
Do not make a literal memo. Make it decently short and to the point. 
CORE STRATEGIC GUIDELINES:
1. FOCUS ON DISCREPANCY: Contrast the ML Risk Score ({score:.1f}) with current news. 
   Does the qualitative context suggest the model is OVER or UNDER-estimating risk?
2. PEER COMPARISON: Use the similarity data {similar_str}. 
   Explain why {ticker_label} is a better/worse risk play than its nearest neighbors.
3. NO FLUFF: End every section with a "PM TAKEAWAY."

SECTION 1: QUANTITATIVE IMPLICATIONS
Interpret the 59% weight on Volatility and 36% on Drawdown. 
What does the current {features['ann_vol']:.1%} vol tell us about the 'regime shift' this asset is in?
PM TAKEAWAY: [One sentence action item]

SECTION 2: QUALITATIVE ALPHA & DISCREPANCY
Use this news: {news_context}. 
Does this news confirm the ML score or signal a "Tail Risk" the historical data is missing? 
Highlight if current sector concentration (like AI/NVDA) makes the historical Beta ({features['beta']:.2f}) unreliable.

SECTION 3: STRATEGIC SUITABILITY
Identify the 'ideal' market condition for this asset and one 'exit trigger' based on VIX correlation ({features['vix_corr']:.2f}).

SIMILAR ETFs FROM TRAINING UNIVERSE (cosine similarity on feature vectors):
{similar_str}
— Use these as behavioral comparables when framing the assessment.

Tone: precise, institutional, skeptical. internal memo, not a chatbot."""

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return (
            f"AI synthesis unavailable: {str(e)}. "
            "Please verify your Gemini API key."
        )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="font-family:\'DM Serif Display\',serif;font-size:1.4rem;color:#E8538A;">◈ RISKPULSE</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="font-family:\'DM Mono\',monospace;font-size:0.55rem;letter-spacing:3px;'
        'color:#888;margin-bottom:20px;">PORTFOLIO ANALYTICS ENGINE</div>',
        unsafe_allow_html=True
    )

    # ① Ticker — drives the live yfinance data fetch
    st.markdown('<div class="sidebar-section">① ETF Ticker</div>', unsafe_allow_html=True)
    ticker_input = st.text_input(
        "", placeholder="e.g. SPY, HYG, ARKK",
        label_visibility="collapsed"
    ).upper().strip()
    st.caption("Auto-populates all features from Yahoo Finance (2010–2026)")

    # ② Stress scenario — overrides live features with regime-based values
    st.markdown('<div class="sidebar-section">② Stress Scenario</div>', unsafe_allow_html=True)
    scenario_choice = st.selectbox(
        "", ["None (use live/manual data)"] + list(SCENARIOS.keys()),
        label_visibility="collapsed"
    )
    if scenario_choice != "None (use live/manual data)":
        st.caption(f"📌 {SCENARIOS[scenario_choice]['description']}")

    # ③ Manual sliders — auto-filled from ticker, tweakable for what-if analysis
    st.markdown('<div class="sidebar-section">③ Adjust / Override</div>', unsafe_allow_html=True)
    st.caption("Auto-filled from ticker. Adjust for custom stress testing.")
    ann_return_s  = st.slider('Ann. Return',     -0.50, 0.50,  0.10, format="%.2f")
    ann_vol_s     = st.slider('Ann. Volatility',  0.05, 0.60,  0.15, format="%.2f")
    sharpe_s      = st.slider('Sharpe Ratio',    -1.00, 2.00,  0.50, format="%.2f")
    max_dd_s      = st.slider('Max Drawdown',    -0.80, 0.00, -0.20, format="%.2f")
    beta_s        = st.slider('Beta to SPY',     -0.50, 2.00,  1.00, format="%.2f")
    vix_corr_s    = st.slider('VIX Correlation', -0.50, 0.50, -0.10, format="%.2f")
    asset_class_s = st.selectbox('Asset Class', ['equity', 'bond', 'commodity', 'alt'])

    # ④ Optional context — passed to Gemini to personalize the assessment
    st.markdown('<div class="sidebar-section">④ Context (Optional)</div>', unsafe_allow_html=True)
    description = st.text_area(
        "", placeholder="Describe the fund mandate or your concern...",
        height=80, label_visibility="collapsed"
    )

    analyze = st.button("◈  Run Risk Analysis")


# ============================================================
# MAIN LAYOUT
# ============================================================
st.markdown(
    '<div><span class="rp-logo">RISKPULSE</span>'
    '<span class="rp-tag">INSTITUTIONAL GRADE</span></div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="rp-subtitle">Portfolio Risk Intelligence · ML + RAG-Augmented AI</div>',
    unsafe_allow_html=True
)

# ── Landing state ──
if not analyze:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="rp-card"><div class="rp-card-label">① Enter a Ticker</div>
            <p style="font-size:0.8rem;color:rgba(255,255,255,0.7);line-height:1.7;margin:0">
            Type any ETF ticker — SPY, HYG, ARKK — and the app pulls 16 years of market
            data and auto-computes all 6 risk features instantly.</p></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="rp-card"><div class="rp-card-label">② Stress Test</div>
            <p style="font-size:0.8rem;color:rgba(255,255,255,0.7);line-height:1.7;margin:0">
            Apply 2008, COVID, or 2022 regime scenarios to see how the risk score
            changes under historical crisis conditions. Use sliders for custom what-if analysis.</p></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="rp-card"><div class="rp-card-label">③ AI Assessment</div>
            <p style="font-size:0.8rem;color:rgba(255,255,255,0.7);line-height:1.7;margin:0">
            Gemini retrieves qualitative market context and synthesizes it with the ML
            output into an institutional risk narrative with peer benchmarking.</p></div>""",
            unsafe_allow_html=True)

    st.markdown("""<div style="text-align:center;margin-top:60px;opacity:0.3;">
        <div style="font-family:'DM Serif Display',serif;font-size:4rem;color:#E8538A;">◈</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:3px;
                    color:#888;margin-top:8px;">ENTER A TICKER IN THE SIDEBAR TO BEGIN</div>
    </div>""", unsafe_allow_html=True)

# ── Results state ──
else:
    # ── Step 1: Resolve feature source ──
    # Priority: live ticker fetch > manual sliders > scenario override
    features      = None
    data_source   = "manual"
    asset_class   = asset_class_s
    scenario_name = None

    if ticker_input:
        with st.spinner(f"Fetching {ticker_input} from Yahoo Finance..."):
            features = fetch_etf_features(ticker_input)
        if features:
            data_source = "live"
            asset_class = get_asset_class(ticker_input)
        else:
            st.warning(f"Could not fetch **{ticker_input}** — using manual slider inputs.")

    if not features:
        features = dict(
            ann_return=ann_return_s, ann_vol=ann_vol_s, sharpe=sharpe_s,
            max_drawdown=max_dd_s,   beta=beta_s,        vix_corr=vix_corr_s
        )

    # Scenario overrides features but preserves asset class for peer comparison
    if scenario_choice != "None (use live/manual data)":
        scenario_name = scenario_choice
        s = SCENARIOS[scenario_choice]
        features = {k: s[k] for k in
                    ['ann_return','ann_vol','sharpe','max_drawdown','beta','vix_corr']}
        data_source = "scenario"

    # ── Step 2: ML prediction ──
    score = predict_score(features, asset_class)

    if score <= 33:   category, cat_class, emoji = 'LOW RISK',    'cat-low',    '▲'
    elif score <= 66: category, cat_class, emoji = 'MEDIUM RISK', 'cat-medium', '◆'
    else:             category, cat_class, emoji = 'HIGH RISK',   'cat-high',   '▼'

    warnings = get_warnings(features, score)

    left, right = st.columns([1, 1.6], gap="large")

    # ============================================================
    # LEFT COLUMN — quantitative ML output
    # ============================================================
    with left:
        if ticker_input:
            st.markdown(f'<div class="ticker-badge">{ticker_input}</div>',
                        unsafe_allow_html=True)
        if scenario_name:
            st.markdown(
                f'<div class="scenario-active">⚡ SCENARIO: {scenario_name.upper()}</div>',
                unsafe_allow_html=True
            )

        # Risk score + gauge
        st.markdown(f"""<div class="rp-card">
            <div class="rp-card-label">Risk Score</div>
            <div class="score-ring-wrap">
                <div>
                    <span class="score-number">{score:.0f}</span>
                    <span class="score-denom">/ 100</span>
                </div>
                <div class="score-category {cat_class}">{emoji} {category}</div>
            </div>
            <div class="gauge-wrap">
                <div class="gauge-track">
                    <div class="gauge-fill" style="width:{score}%"></div>
                </div>
                <div class="gauge-labels">
                    <span>CONSERVATIVE</span><span>MODERATE</span><span>AGGRESSIVE</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Warning flags — only rendered if any thresholds are triggered
        if warnings:
            warn_html = '<div class="rp-card"><div class="rp-card-label">⚠ Risk Flags</div>'
            for title, msg in warnings:
                warn_html += (
                    f'<div class="warn-flag">'
                    f'<div class="warn-icon">⚠</div>'
                    f'<div class="warn-text">'
                    f'<span class="warn-title">{title}</span>{msg}'
                    f'</div></div>'
                )
            warn_html += '</div>'
            st.markdown(warn_html, unsafe_allow_html=True)

        # Color coding helper: green = healthy, pink = risky
        def cc(val, invert=False):
            return "pos" if (val > 0) != invert else "neg"

        # Six-metric grid
        st.markdown(f"""<div class="rp-card">
            <div class="rp-card-label">Quantitative Profile</div>
            <div class="metric-grid">
                <div class="metric-cell">
                    <div class="metric-label">Ann. Return</div>
                    <div class="metric-value {cc(features['ann_return'])}">{features['ann_return']:.1%}</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-label">Volatility</div>
                    <div class="metric-value neg">{features['ann_vol']:.1%}</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-label">Sharpe</div>
                    <div class="metric-value {cc(features['sharpe'])}">{features['sharpe']:.2f}</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value neg">{features['max_drawdown']:.1%}</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-label">Beta</div>
                    <div class="metric-value">{features['beta']:.2f}</div>
                </div>
                <div class="metric-cell">
                    <div class="metric-label">VIX Corr</div>
                    <div class="metric-value {cc(features['vix_corr'], invert=True)}">{features['vix_corr']:.2f}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Feature importance bars — shows what the model actually weighted
        rows_html = "".join([
            f'<div class="feat-row">'
            f'<div class="feat-name">{n}</div>'
            f'<div class="feat-bar-track">'
            f'<div class="feat-bar-fill" style="width:{v*100:.1f}%"></div>'
            f'</div>'
            f'<div class="feat-pct">{v*100:.1f}%</div>'
            f'</div>'
            for n, v in FEATURE_IMPORTANCES
        ])
        st.markdown(
            f'<div class="rp-card">'
            f'<div class="rp-card-label">Model Feature Importance</div>'
            f'{rows_html}</div>',
            unsafe_allow_html=True
        )

    # ============================================================
    # RIGHT COLUMN — peers, AI assessment, metadata
    # ============================================================
    with right:
        # Data source badge
        src_map = {
            "live":     (f"◉ LIVE DATA — {ticker_input}", "#4ade80"),
            "manual":   ("◎ MANUAL INPUT",                "#fbbf24"),
            "scenario": (f"⚡ STRESS SCENARIO — {scenario_name}", "#E8538A"),
        }
        src_label, src_color = src_map[data_source]
        st.markdown(
            f'<div style="font-family:DM Mono,monospace;font-size:0.6rem;'
            f'letter-spacing:2px;color:{src_color};margin-bottom:16px;">'
            f'{src_label}</div>',
            unsafe_allow_html=True
        )

        # Peer comparison — fetch and score 3 ETFs from the same asset class
        # Cached so re-runs don't re-download
        peers = [t for t in PEER_GROUPS.get(asset_class, PEER_GROUPS['equity'])
                 if t != ticker_input][:3]

        with st.spinner("Scoring peer ETFs..."):
            peer_results = [(ticker_input or "INPUT", score, cat_class)]
            for pt in peers:
                pf = fetch_etf_features(pt)
                if pf:
                    ps = predict_score(pf, get_asset_class(pt))
                    pc = ('cat-low'    if ps <= 33 else
                          'cat-medium' if ps <= 66 else 'cat-high')
                    peer_results.append((pt, ps, pc))

        color_map = {'cat-low': '#4ade80', 'cat-medium': '#fbbf24', 'cat-high': '#E8538A'}
        cat_label  = {'cat-low': 'LOW',    'cat-medium': 'MED',     'cat-high': 'HIGH'}

        peer_html = "".join([
            f'<div class="peer-row">'
            f'<div class="peer-ticker">{pt}</div>'
            f'<div class="peer-bar-track">'
            f'<div class="peer-bar-fill" '
            f'style="width:{ps:.0f}%;background:{color_map[pc]};opacity:0.8">'
            f'</div></div>'
            f'<div class="peer-score" style="color:{color_map[pc]}">{ps:.0f}</div>'
            f'<div class="peer-cat"   style="color:{color_map[pc]}">{cat_label[pc]}</div>'
            f'</div>'
            for pt, ps, pc in sorted(peer_results, key=lambda x: x[1])
        ])
        st.markdown(
            f'<div class="rp-card">'
            f'<div class="rp-card-label">Peer Comparison</div>'
            f'{peer_html}</div>',
            unsafe_allow_html=True
        )

        # AI assessment — two-stage RAG pipeline using hardcoded Gemini key
        if GEMINI_API_KEY:
            with st.spinner("Retrieving market context & generating assessment..."):
                try:
                    explanation = get_ai_explanation(
                        ticker_input, score, category, features,
                        description, scenario_name, asset_class
                    )
                    paragraphs = [p.strip() for p in explanation.split('\n\n') if p.strip()]
                    formatted  = "".join([
                        f"<p style='margin-bottom:14px'>{p}</p>"
                        for p in paragraphs
                    ])
                    st.markdown(
                        f'<div class="ai-wrap">'
                        f'<div class="ai-text">{formatted}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"AI assessment failed: {e}")
        else:
            st.markdown("""
                <div class="ai-wrap" style="opacity:0.4;text-align:center;padding:48px 24px;">
                    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                                letter-spacing:2px;color:#888;">
                        GEMINI API KEY NOT CONFIGURED<br>
                        ADD TO secrets.py OR STREAMLIT SECRETS
                    </div>
                </div>""", unsafe_allow_html=True)

        # Analysis metadata card
        st.markdown(f"""<div class="rp-card" style="margin-top:16px">
            <div class="rp-card-label">Analysis Metadata</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px">
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;
                                color:#888;letter-spacing:2px;text-transform:uppercase">
                        Asset Class</div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.8rem;
                                color:white;margin-top:2px">{asset_class.upper()}</div>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;
                                color:#888;letter-spacing:2px;text-transform:uppercase">
                        Model</div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.8rem;
                                color:white;margin-top:2px">RANDOM FOREST</div>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;
                                color:#888;letter-spacing:2px;text-transform:uppercase">
                        CV R²</div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.8rem;
                                color:#E8538A;margin-top:2px">0.807 ± 0.094</div>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;
                                color:#888;letter-spacing:2px;text-transform:uppercase">
                        Training Period</div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.8rem;
                                color:white;margin-top:2px">2010 – 2026</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)