"""
Air Quality Dashboard — Streamlit App
======================================
Run with:
    streamlit run app.py
Make sure the FastAPI backend is also running on port 8000.
"""

import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AirSense — Air Quality Monitor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background: #0D1117; color: #E6EDF3; }

    /* Metric cards */
    .metric-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card .label {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8B949E;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 36px;
        font-weight: 700;
        color: #E6EDF3;
        line-height: 1.1;
    }
    .metric-card .sub {
        font-size: 13px;
        color: #8B949E;
        margin-top: 6px;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.05em;
    }

    /* Header */
    .page-header {
        padding: 32px 0 20px 0;
        border-bottom: 1px solid #30363D;
        margin-bottom: 28px;
    }
    .page-header h1 {
        font-size: 32px;
        font-weight: 800;
        color: #E6EDF3;
        margin: 0;
    }
    .page-header p {
        color: #8B949E;
        margin: 6px 0 0 0;
        font-size: 15px;
    }

    /* Section titles */
    .section-title {
        font-size: 16px;
        font-weight: 700;
        color: #E6EDF3;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #21262D;
    }

    /* Hide streamlit default elements */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161B22;
        border-right: 1px solid #30363D;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌫️ AirSense")
    st.markdown("---")
    st.markdown("### 📡 Backend Status")

    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        if health.get("models_loaded"):
            st.success("API online — models loaded")
        else:
            st.warning("API online — models not loaded")
    except Exception:
        st.error("API offline — run: `uvicorn api:app --reload`")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This dashboard uses three ML models:
    - 📈 **Regressor** — predicts AQI value
    - 🏷️ **Classifier** — assigns risk level
    - 🔵 **K-Means** — detects pollution pattern
    """)
    st.markdown("---")
    st.caption("Air Quality Prediction System · Graduation Project")


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>🌫️ AirSense — Air Quality Monitor</h1>
    <p>Enter pollutant readings to predict the Air Quality Index, classify risk level, and detect pollution patterns.</p>
</div>
""", unsafe_allow_html=True)


# ── Input Section ─────────────────────────────────────────────
st.markdown('<p class="section-title">📥 Pollutant Input</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    co_aqi = st.number_input(
        "🔴 CO AQI Value",
        min_value=0.0, max_value=500.0, value=2.0, step=0.5,
        help="Carbon Monoxide AQI reading"
    )
with col2:
    ozone_aqi = st.number_input(
        "🔵 Ozone AQI Value",
        min_value=0.0, max_value=500.0, value=45.0, step=0.5,
        help="Ozone AQI reading"
    )
with col3:
    no2_aqi = st.number_input(
        "🟡 NO₂ AQI Value",
        min_value=0.0, max_value=500.0, value=12.0, step=0.5,
        help="Nitrogen Dioxide AQI reading"
    )
with col4:
    pm25_aqi = st.number_input(
        "🟠 PM2.5 AQI Value",
        min_value=0.0, max_value=500.0, value=68.0, step=0.5,
        help="Fine particulate matter AQI reading"
    )

st.markdown("")
_, center_col, _ = st.columns([2, 1, 2])
with center_col:
    predict_btn = st.button("⚡ Run Analysis", use_container_width=True, type="primary")


# ── Prediction ───────────────────────────────────────────────
if predict_btn:
    payload = {
        "co_aqi": co_aqi,
        "ozone_aqi": ozone_aqi,
        "no2_aqi": no2_aqi,
        "pm25_aqi": pm25_aqi
    }

    with st.spinner("Running models..."):
        try:
            response = requests.post(f"{API_URL}/predict/all", json=payload, timeout=10)
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API. Make sure it's running on port 8000.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.stop()

    if "detail" in result:
        st.error(f"API Error: {result['detail']}")
        st.stop()

    # ── Results ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📊 Analysis Results</p>', unsafe_allow_html=True)

    aqi_val   = result["aqi_value"]
    aqi_desc  = result["description"]
    risk_level = result["risk_level"]
    cluster_id = result["cluster_id"]
    color_code = result["color"]

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">📈 Predicted AQI Value</div>
            <div class="value">{aqi_val}</div>
            <div class="sub">{aqi_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">🏷️ Risk Level</div>
            <div class="value" style="color: {color_code};">{risk_level}</div>
            <div class="sub">Classifier output</div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">🔵 Pollution Cluster</div>
            <div class="value">Group {cluster_id}</div>
            <div class="sub">K-Means cluster</div>
        </div>
        """, unsafe_allow_html=True)
    # ── Charts ───────────────────────────────────────────────
    st.markdown("")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**🌡️ AQI Gauge**")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=aqi_val,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 300], "tickcolor": "#8B949E"},
                "bar": {"color": color_code},
                "steps": [
                    {"range": [0, 50],   "color": "#1a3a1a"},
                    {"range": [50, 100], "color": "#2a3a1a"},
                    {"range": [100, 150],"color": "#3a3a1a"},
                    {"range": [150, 200],"color": "#3a2a1a"},
                    {"range": [200, 300],"color": "#3a1a1a"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": aqi_val
                }
            },
            number={"font": {"color": "#E6EDF3", "size": 40}}
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#161B22",
            font={"color": "#8B949E"},
            height=280,
            margin=dict(l=20, r=20, t=20, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with chart_col2:
        st.markdown("**📡 Pollutant Radar**")
        labels   = ["CO AQI", "Ozone AQI", "NO₂ AQI", "PM2.5 AQI"]
        values   = [co_aqi, ozone_aqi, no2_aqi, pm25_aqi]

    import matplotlib.colors as mcolors

    rgba_color = mcolors.to_rgba(color_code, 0.25)

    fig_radar = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor=f"rgba({int(rgba_color[0]*255)}, {int(rgba_color[1]*255)}, {int(rgba_color[2]*255)}, 0.25)",
        line={"color": color_code, "width": 2},
        name="Pollutants"
    ))
    fig_radar.update_layout(
            polar={
                "radialaxis": {"visible": True, "color": "#8B949E"},
                "angularaxis": {"color": "#8B949E"},
                "bgcolor": "#161B22"
            },
            paper_bgcolor="#161B22",
            font={"color": "#8B949E"},
            height=280,
            margin=dict(l=40, r=40, t=20, b=20),
            showlegend=False
        )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── AQI Scale Reference ──────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📏 AQI Reference Scale</p>', unsafe_allow_html=True)

    scale_data = pd.DataFrame({
        "Category":    ["Good", "Moderate", "Unhealthy (Sensitive)", "Unhealthy", "Very Unhealthy", "Hazardous"],
        "AQI Range":   ["0 – 50", "51 – 100", "101 – 150", "151 – 200", "201 – 300", "301+"],
        "Who's at risk": [
            "No one", "Sensitive individuals", "Sensitive groups", "General public",
            "Everyone", "Everyone — emergency"
        ],
        "Color": ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C", "#8E44AD", "#5D1A1A"]
    })

    for _, row in scale_data.iterrows():
        is_current = (
            ("Good" in row["Category"] and aqi_val <= 50) or
            ("Moderate" in row["Category"] and 51 <= aqi_val <= 100) or
            ("Sensitive" in row["Category"] and "(" in row["Category"] and 101 <= aqi_val <= 150) or
            ("Unhealthy" == row["Category"] and 151 <= aqi_val <= 200) or
            ("Very" in row["Category"] and 201 <= aqi_val <= 300) or
            ("Hazardous" in row["Category"] and aqi_val > 300)
        )
        border = f"2px solid {row['Color']}" if is_current else "1px solid #30363D"
        arrow  = " ◀ current" if is_current else ""

        st.markdown(f"""
        <div style="
            background: #161B22; border: {border}; border-radius: 8px;
            padding: 10px 16px; margin-bottom: 6px;
            display: flex; align-items: center; gap: 14px;
        ">
            <div style="width:12px; height:12px; border-radius:50%; background:{row['Color']}; flex-shrink:0;"></div>
            <span style="font-weight:600; color:#E6EDF3; width:160px;">{row['Category']}</span>
            <span style="color:#8B949E; width:90px;">{row['AQI Range']}</span>
            <span style="color:#8B949E; flex:1;">{row['Who\'s at risk']}</span>
            <span style="color:{row['Color']}; font-weight:700;">{arrow}</span>
        </div>
        """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("")
    st.markdown("""
    <div style="
        text-align: center; padding: 60px 20px;
        border: 1px dashed #30363D; border-radius: 12px;
        color: #8B949E;
    ">
        <div style="font-size: 48px; margin-bottom: 16px;">🌡️</div>
        <div style="font-size: 18px; font-weight: 600; color: #E6EDF3;">Ready to analyze</div>
        <div style="margin-top: 8px;">Enter pollutant values above and click <strong>Run Analysis</strong></div>
    </div>
    """, unsafe_allow_html=True)
