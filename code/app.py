import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="VisionCheck", page_icon="👁", layout="centered")

# ── Load artifact ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifact():
    path = os.path.join(os.path.dirname(__file__), "model_artifact.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

artifact        = load_artifact()
pipeline        = artifact["pipeline"]
threshold       = artifact["threshold"]
feature_config  = artifact["feature_config"]
train_columns   = artifact["train_columns"]
model_name      = artifact["model_name"]

# ── Feature engineering (mirrors notebook) ───────────────────────────────────
def engineer_features(s: dict, config: dict) -> pd.DataFrame:
    df = pd.DataFrame([s])

    if config.get("age_squared"):
        df["age"] = df["age"] ** 2
    if config.get("near_work_intensity"):
        df["near_work_intensity"] = df["screen_time_hours"] * (1 / df["screen_distance_cm"])
    if config.get("light_dose_near"):
        df["light_dose_near"] = df["screen_time_hours"] * df["screen_brightness_avg"]
    if config.get("night_mode_ratio"):
        df["night_mode_ratio"] = df["night_mode_usage"] / (df["screen_time_hours"] + 0.1)
    if config.get("mh_age_interaction"):
        df["mh_age_interaction"] = df["mental_health_score"] * df["age"]
    if config.get("screen_bin"):
        df["screen_bin"] = pd.cut(
            df["screen_time_hours"],
            bins=[0, 1, 4, 8, np.inf],
            labels=["<=1h", "1-4h", "4-8h", ">8h"]
        )

    return df[train_columns]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: -apple-system, 'Inter', 'Helvetica Neue', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background: #eaf0fb;
    background-image:
        radial-gradient(ellipse 80% 60% at 15% 10%, rgba(120,180,255,0.4) 0%, transparent 60%),
        radial-gradient(ellipse 60% 70% at 85% 85%, rgba(180,140,255,0.3) 0%, transparent 55%),
        radial-gradient(ellipse 50% 60% at 10% 75%, rgba(100,220,180,0.25) 0%, transparent 55%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; max-width: 460px; }

.app-header { text-align: center; margin-bottom: 32px; }
.app-icon {
    width: 76px; height: 76px;
    background: linear-gradient(145deg, #007AFF, #5AC8FA);
    border-radius: 20px;
    margin: 0 auto 12px;
    font-size: 34px;
    line-height: 76px;
    text-align: center;
    box-shadow: 0 16px 48px rgba(0,122,255,0.3), 0 0 0 0.5px rgba(255,255,255,0.4) inset;
}
.app-title {
    font-size: 26px; font-weight: 700;
    color: rgba(0,0,0,0.85); letter-spacing: -0.5px; margin: 0;
}
.app-sub { font-size: 13px; color: rgba(0,0,0,0.4); margin: 4px 0 0; }
.model-badge {
    display: inline-block;
    background: rgba(0,122,255,0.1); color: #007AFF;
    font-size: 11px; font-weight: 600; letter-spacing: 0.3px;
    padding: 3px 10px; border-radius: 20px; margin-top: 8px;
}

.card {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(40px) saturate(180%);
    -webkit-backdrop-filter: blur(40px) saturate(180%);
    border: 0.5px solid rgba(255,255,255,0.6);
    border-radius: 20px; padding: 20px 22px 8px; margin-bottom: 12px;
    box-shadow: 0 2px 0 rgba(255,255,255,0.85) inset, 0 16px 40px rgba(0,0,0,0.07);
}
.card-label {
    font-size: 11px; font-weight: 600; color: rgba(0,0,0,0.38);
    letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 4px;
}

div[data-testid="stSlider"] label {
    font-size: 15px !important; font-weight: 400 !important;
    color: rgba(0,0,0,0.82) !important; letter-spacing: -0.2px !important;
}

.stButton > button {
    width: 100% !important; background: #007AFF !important;
    color: white !important; border: none !important;
    border-radius: 14px !important; padding: 15px !important;
    font-size: 16px !important; font-weight: 600 !important;
    letter-spacing: -0.2px !important;
    box-shadow: 0 10px 28px rgba(0,122,255,0.32) !important;
    margin-top: 4px !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.result { border-radius: 20px; padding: 28px 24px 22px; margin-top: 14px; text-align: center; }
.result-positive { background: rgba(255,149,0,0.07); border: 0.5px solid rgba(255,149,0,0.28); }
.result-negative { background: rgba(52,199,89,0.07); border: 0.5px solid rgba(52,199,89,0.28); }
.result-emoji { font-size: 46px; margin-bottom: 10px; }
.result-headline { font-size: 21px; font-weight: 700; letter-spacing: -0.4px; margin: 0 0 5px; }
.result-positive .result-headline { color: #E08600; }
.result-negative .result-headline { color: #28A745; }
.result-sub { font-size: 14px; color: rgba(0,0,0,0.45); line-height: 1.5; margin: 0 0 22px; }

.conf-row {
    display: flex; justify-content: space-between;
    font-size: 11px; font-weight: 600; color: rgba(0,0,0,0.38);
    letter-spacing: 0.4px; text-transform: uppercase; margin-bottom: 7px;
}
.conf-positive { color: #E08600; }
.conf-negative { color: #28A745; }
.conf-track { height: 5px; background: rgba(0,0,0,0.07); border-radius: 3px; overflow: hidden; margin-bottom: 20px; }
.conf-fill-positive { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #FF9500, #FFCC00); }
.conf-fill-negative { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #34C759, #30D158); }

section[data-testid="stMain"] > div > div > div > div > div[data-testid="stVerticalBlock"] {
    background: rgba(255,255,255,0.06);
    border: 0.5px solid rgba(255,255,255,0.92);
    border-radius: 24px;
    padding: 24px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 1px 0 rgba(255,255,255,0.08) inset, 0 16px 40px rgba(0,0,0,0.3);
}

.factor {
    display: flex; align-items: center; gap: 10px;
    background: rgba(0,0,0,0.04); border-radius: 10px;
    padding: 9px 13px; margin-bottom: 7px;
    font-size: 13px; color: rgba(0,0,0,0.7); text-align: left;
}
.dot { font-size: 8px; }
.dot-positive { color: #FF9500; }
.dot-negative { color: #34C759; }

.summary {
    background: rgba(0,122,255,0.06); border-radius: 12px;
    padding: 13px 15px; font-size: 13px; color: rgba(0,0,0,0.5);
    line-height: 1.6; text-align: left; margin-top: 4px;
}
.thr-note { font-size: 11px; color: rgba(0,0,0,0.3); text-align: center; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
    <div class="app-icon">👁</div>
    <p class="app-title">VisionCheck</p>
    <p class="app-sub">Find out if you're likely to require Glasses</p>
</div>
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-label">🏃 Lifestyle</div>', unsafe_allow_html=True)
exercise_hours      = st.slider("Exercise hours / week", 0.0, 30.0, 3.0, 0.5)
mental_health_score = st.slider("Mental health score", 0, 100, 65)
age                 = st.slider("Age", 5, 90, 30)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-label">📱 Screen Habits</div>', unsafe_allow_html=True)
screen_time_hours     = st.slider("Screen time / day (h)", 0.0, 18.0, 8.0, 0.5)
screen_brightness_avg = st.slider("Screen brightness (%)", 0, 100, 70)
night_mode_usage      = st.slider("Night mode usage (%)", 0, 100, 50)
screen_distance_cm    = st.slider("Screen distance (cm)", 10, 120, 50)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-label">🌿 Environment</div>', unsafe_allow_html=True)
outdoor_light_exposure_hours = st.slider("Outdoor light / day (h)", 0.0, 12.0, 2.0, 0.5)
st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Analyse Vision Risk"):
    raw = {
        "exercise_hours":               exercise_hours,
        "mental_health_score":          float(mental_health_score),
        "screen_time_hours":            screen_time_hours,
        "screen_brightness_avg":        float(screen_brightness_avg),
        "age":                          float(age),
        "outdoor_light_exposure_hours": outdoor_light_exposure_hours,
        "night_mode_usage":             float(night_mode_usage),
        "screen_distance_cm":           float(screen_distance_cm),
    }

    X           = engineer_features(raw, feature_config)
    proba       = pipeline.predict_proba(X)[0][1]
    prediction  = 1 if proba >= threshold else 0
    is_positive = prediction == 1
    conf_pct    = int(proba * 100)

    factors_scored = [
        (f"{screen_time_hours}h screen time / day",         screen_time_hours / 18),
        (f"Screen distance {screen_distance_cm}cm",         1 - (screen_distance_cm - 10) / 110),
        (f"{outdoor_light_exposure_hours}h outdoors / day", 1 - outdoor_light_exposure_hours / 12),
        (f"Age {age}",                                      age / 90),
        (f"Screen brightness {screen_brightness_avg}%",     screen_brightness_avg / 100),
        (f"Night mode {night_mode_usage}%",                 (1 - night_mode_usage/100) if is_positive else night_mode_usage/100),
        (f"{exercise_hours}h exercise / week",              (1 - exercise_hours/30) if is_positive else exercise_hours/30),
    ]
    top3 = [f for f, _ in sorted(factors_scored, key=lambda x: x[1], reverse=True)[:3]]

    rc      = "result-positive" if is_positive else "result-negative"
    fc      = "conf-fill-positive" if is_positive else "conf-fill-negative"
    cc      = "conf-positive" if is_positive else "conf-negative"
    dc      = "dot-positive" if is_positive else "dot-negative"
    emoji   = "👓" if is_positive else "✓"
    hl      = "Likely to need glasses" if is_positive else "Low glasses risk"
    sub     = (
        "Your profile matches patterns associated with corrective lens use."
        if is_positive else
        "Your lifestyle patterns suggest low risk of needing glasses."
    )
    summary = (
        "The model flags elevated risk based on your screen habits and age profile. "
        "Consider reducing screen time and increasing outdoor exposure."
        if is_positive else
        "Your current lifestyle shows no strong risk indicators. "
        "Maintaining outdoor activity and screen distance helps protect long-term eye health."
    )

    factors_html = "".join([
        f'<div class="factor"><span class="dot {dc}">●</span>{f}</div>'
        for f in top3
    ])

    st.markdown(f"""
    <div class="result {rc}">
        <div class="summary">{summary}</div>
        <div class="result-emoji">{emoji}</div>
        <div class="result-headline">{hl}</div>
        <div class="result-sub">{sub}</div>
        <div class="conf-row">
            <span>Model confidence</span>
            <span class="{cc}">{conf_pct}%</span>
        </div>
        <div class="conf-track">
            <div class="{fc}" style="width:{conf_pct}%"></div>
        </div>
        {factors_html}
    </div>
    """, unsafe_allow_html=True)
