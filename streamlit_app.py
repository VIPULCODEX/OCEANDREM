"""
OceanEmbed - Interactive Dashboard

NOT YET UPDATED for the real-data-only pipeline. This file was originally
built around synthetic/ocean_pipeline_demo.py (now deleted -- the project
dropped the synthetic data track entirely, see README.md) and several of
the model-training wrapper functions it imported from models.dl_pipeline
were removed in the same change, since they were synthetic-pipeline-only.
Porting this Streamlit app to the real-data pipeline (real/real_training.py)
is a separate, larger task -- it duplicates significant dashboard logic
independently of public/app.js, rather than reusing it. The currently
working option is the static dashboard: see public/ and README.md's
"Two ways to run this" section.

This file still parses, but will exit immediately with a clear error if
actually run (`streamlit run streamlit_app.py`), rather than failing with
a raw traceback on the missing imports below.

Run with:
    streamlit run streamlit_app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from pathlib import Path

FAVICON_PATH = Path(__file__).parent / "favicon.svg"
st.set_page_config(
    page_title="Sea Green | OceanEmbed",
    layout="wide",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else None,
)

# Inject custom Sea Green design system
CSS_FILE = Path(__file__).parent / "streamlit_custom.css"
if CSS_FILE.exists():
    st.html(CSS_FILE)

# ── Ocean photo background — injected as base64 data URI (no static server needed) ──
import base64 as _b64
_bg_path = Path(__file__).parent / "beautiful-shot-fishes-swimming-ocean.jpg"
if _bg_path.exists():
    _bg_b64 = _b64.b64encode(_bg_path.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        /* ── Step 1: Ocean photo on .stApp root ── */
        [data-testid="stAppViewContainer"],
        .stApp {{
          background-color: #061412 !important;
          background-image:
            /* faint wave-line watermark — scrolls with page */
            url("data:image/svg+xml,%3Csvg width='160' height='44' viewBox='0 0 160 44' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 22 Q 40 8, 80 22 T 160 22' fill='none' stroke='%233fcf8e' stroke-width='0.8' opacity='0.025'/%3E%3C/svg%3E"),
            /* ocean photograph — fixed, full-viewport */
            url("data:image/jpeg;base64,{_bg_b64}") !important;
          background-attachment: scroll, fixed !important;
          background-repeat: repeat-x, no-repeat !important;
          background-size: 160px 44px, cover !important;
          background-position: bottom center, center center !important;
          position: relative !important;
        }}

        /* ── Step 2: Dedicated dark scrim — sits above photo, behind all content ── */
        .stApp::before {{
          content: "" !important;
          position: fixed !important;
          inset: 0 !important;          /* top:0 right:0 bottom:0 left:0 */
          background: linear-gradient(
            180deg,
            rgba(6, 20, 18, 0.55) 0%,
            rgba(6, 20, 18, 0.70) 100%
          ) !important;
          pointer-events: none !important;
          z-index: 0 !important;
        }}

        /* ── Step 3: Lift all Streamlit content above the scrim ── */
        .stApp > * {{
          position: relative !important;
          z-index: 1 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Inject Google Fonts — Fraunces & Lora (warm humanist serif display) + Space Grotesk & IBM Plex Mono
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400..700;1,9..144,400..700&family=Lora:ital,wght@0,400..700;1,400..700&family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# STROKE-BASED INLINE SVG ICONS (stroke-width: 1.5, size: 18-20px, accent color)
# ----------------------------------------------------------------------
ICON_WAVE = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#e8f5f0" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 6c.6.5 1.2 1 2.5 1C7 7 7 5 9.5 5c2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/><path d="M2 12c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/><path d="M2 18c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/></svg>"""

ICON_ALERT = """<svg class="sg-icon-alert" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#e2543f" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>"""

ICON_THERMOMETER = """<span class="sg-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3fcf8e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/><path d="M10 9h4"/><path d="M12 9v5"/></svg></span>"""

ICON_SATELLITE = """<span class="sg-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3fcf8e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 10a7.97 7.97 0 0 1 6-6"/><path d="M2 14a11.96 11.96 0 0 1 10-10"/><path d="m8 20 4-4"/><path d="m12 16 7-7a2.83 2.83 0 1 0-4-4l-7 7Z"/><path d="m17 7 3-3"/></svg></span>"""

ICON_BRAIN = """<span class="sg-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3fcf8e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect width="6" height="6" x="16" y="16" rx="1"/><rect width="6" height="6" x="2" y="16" rx="1"/><rect width="6" height="6" x="9" y="2" rx="1"/><path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3"/><path d="M12 12V8"/></svg></span>"""

ICON_BAR_CHART = """<span class="sg-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3fcf8e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg></span>"""

ICON_PROFILE = """<span class="sg-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3fcf8e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="1" x2="7" y1="14" y2="14"/><line x1="9" x2="15" y1="8" y2="8"/><line x1="17" x2="23" y1="16" y2="16"/></svg></span>"""

ICON_SCATTER = """<span class="sg-icon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3fcf8e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M22 12h-4"/><path d="M6 12H2"/><path d="M12 6V2"/><path d="M12 22v-4"/></svg></span>"""


def apply_theme(fig):
    """Applies the Sea Green dark ocean / scientific-instrument theme to Plotly figures."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 24, 40, 0.75)",
        font=dict(
            color="#ffffff",
            family="'Space Grotesk', system-ui, sans-serif",
            size=12,
        ),
        legend=dict(
            font=dict(color="#ffffff", size=11),
            bgcolor="rgba(10, 24, 42, 0.85)",
            bordercolor="rgba(100, 180, 255, 0.25)",
            borderwidth=1,
        ),
        margin=dict(l=60, r=30, t=40, b=50),
    )
    # Clear graph boundary lines and grid — sharp high contrast
    _grid = "rgba(255, 255, 255, 0.15)"
    _axis_line = "rgba(255, 255, 255, 0.45)"  # Strong boundary line for graph axes
    fig.update_xaxes(
        gridcolor=_grid,
        zerolinecolor=_grid,
        showline=True,
        linewidth=2,
        linecolor=_axis_line,
        mirror=True,
        tickfont=dict(family="'Space Grotesk', sans-serif", color="#ffffff", size=11),
        title=dict(font=dict(color="#ffffff", size=13, family="'Space Grotesk'")),
    )
    fig.update_yaxes(
        gridcolor=_grid,
        zerolinecolor=_grid,
        showline=True,
        linewidth=2,
        linecolor=_axis_line,
        mirror=True,
        tickfont=dict(family="'Space Grotesk', sans-serif", color="#ffffff", size=11),
        title=dict(font=dict(color="#ffffff", size=13, family="'Space Grotesk'")),
    )
    return fig


def render_skeleton_slot(placeholder, label_text="Loading..."):
    """Renders a pulsing CSS skeleton card matching metric dimensions inside an st.empty() slot."""
    placeholder.markdown(
        f"""
        <div class="skeleton-metric-card">
            <div class="skeleton-bar skeleton-bar-label"></div>
            <div class="skeleton-bar skeleton-bar-value"></div>
            <div class="skeleton-bar skeleton-bar-sub"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    from synthetic.ocean_pipeline_demo import (
        USE_SYNTHETIC_DATA,
        LAT_RANGE,
        LON_RANGE,
        DEPTH_LEVELS,
        N_DAYS,
        get_satellite_grid,
        build_training_table,
        train_and_evaluate,
        marine_heatwave_series,
    )
    from models.dl_pipeline import (
        train_pooled_ffnn_and_evaluate, train_cnn_and_evaluate, train_lstm_and_evaluate,
        train_vit_and_evaluate, train_autoencoder_and_evaluate, train_gnn_and_evaluate,
    )
except ImportError as e:
    st.error(
        "This Streamlit app has not yet been updated for the real-data-only "
        "pipeline (the synthetic track it was built around was removed). "
        "Use the static dashboard instead: run `python -m real.export_real_results` "
        "then open public/index.html, or see README.md's \"Two ways to run this\" "
        f"section.\n\nImport error: {e}"
    )
    st.stop()

# Every model trained/tested, same time-based split, same test set (mirrors
# export_data.py's MODELS list -- keep the two in sync).
MODELS = [
    ("rf", "Random Forest", train_and_evaluate),
    ("cnn", "CNN (satellite patches)", train_cnn_and_evaluate),
    ("vit", "ViT (attention over patches)", train_vit_and_evaluate),
    ("gnn", "GNN (k-NN graph)", train_gnn_and_evaluate),
    ("autoencoder", "Autoencoder (unsupervised embedding)", train_autoencoder_and_evaluate),
    ("lstm", "LSTM (depth decoder)", train_lstm_and_evaluate),
    ("ffnn", "FFNN (headline)", train_pooled_ffnn_and_evaluate),
]

# ----------------------------------------------------------------------
# CMOCEAN-INSPIRED HIGH-CONTRAST COLOUR PALETTE
# ----------------------------------------------------------------------
PALETTE = {
    # ---- thermal scale samples (temperature) ----
    "th_deep":   "#12082b",   # darkest navy
    "th_cool":   "#2a4858",   # deep teal-navy
    "th_teal":   "#3fcf8e",   # bright mint teal — Argo actual
    "th_warm":   "#e58a3a",   # warm amber — predicted / accent
    "th_hot":    "#e2543f",   # brick red  — extreme events
    "th_line":   "#3fcf8e",   # SST anomaly main line

    # ---- haline scale samples (model comparison, 8 slots - vibrant high contrast) ----
    "ha_0":  "#e2543f",   # Naive baseline  (coral red)
    "ha_1":  "#e3c23c",   # Random Forest   (bright yellow)
    "ha_2":  "#35b0e6",   # CNN             (cyan blue)
    "ha_3":  "#9b51e0",   # ViT             (bright purple)
    "ha_4":  "#f2994a",   # GNN             (vibrant orange)
    "ha_5":  "#27ae60",   # Autoencoder     (emerald green)
    "ha_6":  "#e056fd",   # LSTM            (neon magenta)
    "ha_7":  "#3fcf8e",   # FFNN headline   (mint green)

    # ---- neutrals ----
    "ref":    "#e2543f",  # perfect-prediction dashed line
    "argo_x": "#ffdd66",  # Argo float markers on map
}

# ----------------------------------------------------------------------
# CACHED WRAPPERS around the existing pipeline (so tabs don't retrain
# the model / regenerate data on every interaction)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Training Random Forest + 5 neural-network architectures (same test split, for a fair comparison)...")
def cached_train():
    X, Y, argo_full = build_training_table()

    results = {}
    for key, label, trainer in MODELS:
        _, X_test, Y_test, preds, metrics = trainer(X, Y)
        results[key] = {"preds": preds, "metrics": metrics}

    # FFNN is the headline model (wins overall -- see PROJECT_REPORT.txt);
    # its predictions drive the profile/scatter charts below.
    preds, metrics = results["ffnn"]["preds"], results["ffnn"]["metrics"]
    for key in results:
        if key == "ffnn":
            continue
        metrics = metrics.merge(
            results[key]["metrics"][["depth", "rmse_model"]].rename(columns={"rmse_model": f"rmse_{key}"}),
            on="depth",
        )

    model_summary = pd.DataFrame(
        [{"name": "Naive guess", "avg_rmse": metrics["rmse_baseline"].mean()}]
        + [{"name": label, "avg_rmse": results[key]["metrics"]["rmse_model"].mean()} for key, label, _ in MODELS]
    )
    return X_test, Y_test, preds, metrics, model_summary


@st.cache_data(show_spinner="Fetching satellite grid...")
def cached_grid(day):
    return get_satellite_grid(day)


@st.cache_data(show_spinner="Scanning for marine heatwave events...")
def cached_heatwave():
    return marine_heatwave_series()


X_test, Y_test, preds, metrics, model_summary = cached_train()
heatwave = cached_heatwave()

# ----------------------------------------------------------------------
# STICKY TOPBAR — team ID + one-line pitch
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="sg-topbar">
        <span class="sg-topbar-id">PS&nbsp;26066&ensp;&mdash;&ensp;OceanEmbed</span>
        <span class="sg-topbar-sep" aria-hidden="true">&middot;</span>
        <span class="sg-topbar-pitch">Predicting ocean subsurface temperature from satellite surface data</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# BRAND HEADER & DEMO BANNER
# ----------------------------------------------------------------------
_hero_left, _hero_right = st.columns([2, 1], gap="large")

with _hero_left:
    st.markdown(
        f"""
        <header class="sg-header">
            <div class="sg-header-row">
                <div class="sg-brand">
                    <div class="sg-logo">{ICON_WAVE}</div>
                    <div class="sg-title-group">
                        <h1>Sea Green &mdash; OceanEmbed</h1>
                        <p>North Indian Ocean marine heatwave monitoring &amp; 3D subsurface temperature reconstruction</p>
                    </div>
                </div>
                <div class="sg-badge-container">
                    <div class="sg-badge">
                        <span class="sg-dot"></span>
                        <span>{'Simulated Demo' if USE_SYNTHETIC_DATA else 'Live Feed'}</span>
                    </div>
                </div>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )

with _hero_right:
    # Sparkline: last 30 days of basin SST anomaly — glanceable trend shape only
    st.markdown('<div class="sg-sparkline-card">', unsafe_allow_html=True)
    _spark_df = heatwave.tail(30).copy()
    _fig_spark = go.Figure()
    _fig_spark.add_trace(
        go.Scatter(
            x=_spark_df["day"],
            y=_spark_df["anomaly"],
            mode="lines",
            line=dict(color="#3fcf8e", width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(63, 207, 142, 0.15)",
            hoverinfo="skip",
        )
    )
    _fig_spark.update_layout(
        margin=dict(l=4, r=4, t=4, b=4),
        height=85,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        showlegend=False,
    )
    # st.plotly_chart(_fig_spark, use_container_width=True, config={"displayModeBar": False})
    # st.markdown('<p class="sg-sparkline-caption">30-day SST anomaly trend</p></div>', unsafe_allow_html=True)

# st.markdown(
#     f"""
#     <div class="sg-note-card">
#         <div class="sg-note-header">
#             <span class="sg-note-icon">{ICON_ALERT}</span>
#             <span class="sg-note-title">Demonstration Note &middot; Synthetic Data Pipeline</span>
#         </div>
#         <p class="sg-note-body">
#             Pipeline validated on simulated data (USE_SYNTHETIC_DATA = {USE_SYNTHETIC_DATA}). Real Argo and MOSDAC integration points are designated in <em>ocean_pipeline_demo.py</em> (see <em>get_satellite_grid</em> and <em>get_argo_data</em>). To activate real-time telemetry, toggle the configuration flag and initialize the <em>argopy</em> and <em>copernicusmarine</em> connectors.
#         </p>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["Heatwave Monitor", "Spatial Observations", "Model Benchmarks", "Vertical Profiles", "Depth Scatter"]
)

# ----------------------------------------------------------------------
# TAB 0 - MARINE HEATWAVE MONITOR
# ----------------------------------------------------------------------
with tab0:
    st.markdown(
        f'<div class="tab-header sg-tab-heatwave">{ICON_THERMOMETER} <h3>Basin-averaged SST Anomaly vs. Climatology</h3></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.35, 1.0, 0.95], gap="medium")
    slot_peak = c1.empty()
    slot_sst = c2.empty()
    slot_days = c3.empty()

    render_skeleton_slot(slot_peak, "Peak Anomaly")
    render_skeleton_slot(slot_sst, "Basin SST")
    render_skeleton_slot(slot_days, "Days Monitored")

    peak = heatwave.loc[heatwave["anomaly"].idxmax()]
    basin_sst_peak = 28.5 + float(peak["anomaly"])

    slot_peak.metric("Peak anomaly", f"{peak['anomaly']:.2f} °C", peak["category"])
    slot_sst.metric("Peak basin SST", f"{basin_sst_peak:.2f} °C", "thermal peak")
    slot_days.metric("Days monitored", N_DAYS, f"Day {int(peak['day'])} peak")

    color_map = {
        "Normal": "#3fcf8e", "Watch": "#e3c23c", "Warning": "#e58a3a",
        "Severe": "#e2543f", "Extreme": "#b5203a",
    }
    fig_hw = go.Figure()
    fig_hw.add_trace(
        go.Scatter(
            x=heatwave["day"], y=heatwave["anomaly"], mode="lines",
            line=dict(color="#3fcf8e", width=3), name="SST anomaly",
        )
    )
    fig_hw.add_trace(
        go.Scatter(
            x=heatwave["day"], y=heatwave["anomaly"], mode="markers",
            marker=dict(size=7, color=heatwave["category"].map(color_map), line=dict(width=1, color="#061412")),
            name="Category", showlegend=False,
        )
    )
    for thresh, label in [(0.5, "Watch"), (1.0, "Warning"), (1.5, "Severe"), (2.0, "Extreme")]:
        fig_hw.add_hline(
            y=thresh, line_dash="dot", line_color=color_map[label], line_width=1.5,
            annotation_text=label, annotation_position="right",
            annotation_font=dict(color="#f4f7f6", size=11),
        )
    fig_hw.update_layout(
        title="Heatwave Detector(Test Run)",
        xaxis_title="Day of season window", yaxis_title="SST anomaly (&deg;C)",
        height=450, showlegend=False,
    )
    st.plotly_chart(apply_theme(fig_hw), width="stretch")
    # st.markdown(
    #     '<div class="sg-graph-caption">Categories follow the Hobday marine-heatwave scale (Watch &ge; 0.5&deg;C, '
    #     'Warning &ge; 1.0&deg;C, Severe &ge; 1.5&deg;C, Extreme &ge; 2.0&deg;C above climatology).</div>',
    #     unsafe_allow_html=True
    # )
    st.markdown(
    '<div class="sg-graph-caption">Categories follow the Hobday marine-heatwave scale(Watch ≥ 0.5°C, Warning ≥ 1.0°C, Severe ≥ 1.5°C, Extreme ≥ 2.0°C above climatology).'
    '<br>'
    'The Hobday scale is a standard system used to measure how intense a marine heatwave is. It compares the current sea temperature with the normal temperature for that location and time of year. The greater the temperature difference, the higher the heatwave category.</div>',
    unsafe_allow_html=True
)

# ----------------------------------------------------------------------
# TAB 1 - OVERVIEW
# ----------------------------------------------------------------------
with tab1:
    st.markdown(
        f'<div class="tab-header sg-tab-spatial">{ICON_SATELLITE} <h3>Surface Satellite Observations &amp; Argo Float Distribution</h3></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Argo floats measure subsurface ocean temperature directly, but they are
        sparse in space and time. Satellites see the surface (Sea Surface Temperature(SST), Sea Surface Height(SSH), salinity,
        wind) continuously and everywhere. This pipeline learns the relationship
        between surface satellite observations and subsurface temperature at
        several depths, so we can estimate subsurface structure in places and
        times where no Argo float was present.
        """
    )

    day_for_map = st.slider("Day (of the season window)", 0, N_DAYS - 1, 45)

    s1, s2, s3 = st.columns([1.2, 0.9, 1.15], gap="medium")
    slot_map_sst = s1.empty()
    slot_map_fl = s2.empty()
    slot_map_time = s3.empty()

    render_skeleton_slot(slot_map_sst, "Basin Mean SST")
    render_skeleton_slot(slot_map_fl, "Test Floats")
    render_skeleton_slot(slot_map_time, "Satellite Pass")

    grid = cached_grid(day_for_map)

    slot_map_sst.metric("Basin mean SST", f"{grid['sst'].mean():.2f} °C", "satellite grid")
    slot_map_fl.metric("Argo profiles", f"{len(X_test)}", "in-situ test locations")
    slot_map_time.metric("Satellite pass timestamp", f"Day {day_for_map} / {N_DAYS}", "window observation")

    fig_map = go.Figure()
    fig_map.add_trace(
        go.Scatter(
            x=grid["lon"], y=grid["lat"],
            mode="markers",
            marker=dict(
                size=8,
                color=grid["sst"],
                colorscale="Thermal",
                colorbar=dict(title=dict(text="SST (&deg;C)", font=dict(color="#f4f7f6")), tickfont=dict(color="#f4f7f6")),
            ),
            name="Satellite SST grid",
            hovertemplate="lon %{x:.1f}, lat %{y:.1f}<br>SST %{marker.color:.2f}&deg;C<extra></extra>",
        )
    )
    fig_map.add_trace(
        go.Scatter(
            x=X_test["lon"], y=X_test["lat"],
            mode="markers",
            marker=dict(size=10, color=PALETTE["argo_x"], symbol="x", line=dict(width=2, color="#000000")),
            name="Argo profiles (test set)",
            hovertemplate="Argo float<br>lon %{x:.2f}, lat %{y:.2f}<extra></extra>",
        )
    )
    fig_map.update_layout(
        title=dict(text=f"Surface SST + Argo float locations (day {day_for_map})", font=dict(color="#f4f7f6", size=15)),
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        xaxis=dict(range=list(LON_RANGE)),
        yaxis=dict(range=list(LAT_RANGE)),
        legend=dict(orientation="h", y=-0.15),
        height=550,
    )
    st.plotly_chart(apply_theme(fig_map), width="stretch")

# ----------------------------------------------------------------------
# TAB 2 - MODEL BENCHMARKS
# ----------------------------------------------------------------------
with tab2:
    st.markdown(
        f'<div class="tab-header sg-tab-benchmarks">{ICON_BRAIN} <h3>Comparative Architecture Benchmarks</h3></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sg-graph-caption">Seven independently-trained approaches (one classical, six neural-network designs), '
        'same held-out test data. Shorter bar = less error.</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3 = st.columns([1.4, 1.0, 1.0], gap="medium")
    slot_skill = k1.empty()
    slot_floats = k2.empty()
    slot_corr = k3.empty()

    render_skeleton_slot(slot_skill, "Error Reduction")
    render_skeleton_slot(slot_floats, "Test Floats")
    render_skeleton_slot(slot_corr, "Correlation")

    baseline_rmse = metrics["rmse_baseline"].mean()
    headline_rmse = metrics["rmse_model"].mean()
    skill_pct = ((baseline_rmse - headline_rmse) / baseline_rmse) * 100
    avg_corr = metrics["correlation"].mean()

    slot_skill.metric("Error reduction", f"{skill_pct:.1f}%", "-54% vs climatology")
    slot_floats.metric("Test float count", f"{len(X_test)}", "held-out evaluation")
    slot_corr.metric("Correlation (r)", f"{avg_corr:.3f}", "1.0 = perfect match")
    fig_summary = px.bar(
        model_summary, x="name", y="avg_rmse",
        color=model_summary["avg_rmse"] == model_summary["avg_rmse"].min(),
        color_discrete_map={True: PALETTE["ha_7"], False: PALETTE["ha_2"]},
        labels={"name": "", "avg_rmse": "Mean RMSE (&deg;C)"},
        text=model_summary["avg_rmse"].round(3),
    )
    fig_summary.update_traces(textposition="outside", textfont=dict(color="#f4f7f6", size=12))
    fig_summary.update_layout(title="Model Comparision",showlegend=False, height=350)
    st.plotly_chart(apply_theme(fig_summary), width="stretch")

    st.markdown(
        f'<div class="tab-header">{ICON_BAR_CHART} <h3>RMSE per Depth: All Seven Models vs. Naive Baseline</h3></div>',
        unsafe_allow_html=True,
    )

    metrics_long = metrics.melt(
        id_vars="depth",
        value_vars=["rmse_baseline", "rmse_rf", "rmse_cnn", "rmse_vit", "rmse_gnn", "rmse_autoencoder", "rmse_lstm", "rmse_model"],
        var_name="method",
        value_name="rmse",
    )
    metrics_long["method"] = metrics_long["method"].map(
        {
            "rmse_baseline": "Naive baseline (climatology)",
            "rmse_rf": "Random Forest",
            "rmse_cnn": "CNN (satellite patches)",
            "rmse_vit": "ViT (attention over patches)",
            "rmse_gnn": "GNN (k-NN graph)",
            "rmse_autoencoder": "Autoencoder (unsupervised embedding)",
            "rmse_lstm": "LSTM (depth decoder)",
            "rmse_model": "Neural Network (FFNN, headline)",
        }
    )

    fig_bar = px.line(
        metrics_long,
        x="depth", y="rmse", color="method", markers=True,
        color_discrete_map={
            "Naive baseline (climatology)": PALETTE["ha_0"],
            "Random Forest":                PALETTE["ha_1"],
            "CNN (satellite patches)":      PALETTE["ha_2"],
            "ViT (attention over patches)": PALETTE["ha_3"],
            "GNN (k-NN graph)":             PALETTE["ha_4"],
            "Autoencoder (unsupervised embedding)": PALETTE["ha_5"],
            "LSTM (depth decoder)":         PALETTE["ha_6"],
            "Neural Network (FFNN, headline)": PALETTE["ha_7"],
        },
        labels={"depth": "Depth (m)", "rmse": "RMSE (&deg;C)", "method": ""},
    )
    fig_bar.update_traces(line=dict(width=3), marker=dict(size=8))
    fig_bar.update_layout(title="Error by Depth",legend=dict(orientation="h", y=-0.3), height=480)
    st.plotly_chart(apply_theme(fig_bar), width="stretch")

    st.markdown(
        f'<div class="tab-header">{ICON_BAR_CHART} <h3>Metrics Table</h3></div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        metrics.rename(
            columns={
                "depth": "Depth",
                "rmse_baseline": "RMSE - Naive baseline",
                "rmse_rf": "RMSE - Random Forest",
                "rmse_cnn": "RMSE - CNN",
                "rmse_vit": "RMSE - ViT",
                "rmse_gnn": "RMSE - GNN",
                "rmse_autoencoder": "RMSE - Autoencoder",
                "rmse_lstm": "RMSE - LSTM",
                "rmse_model": "RMSE - FFNN (headline)",
                "correlation": "Correlation (r)",
                "bias": "Bias (model - actual)",
            }
        )[[
            "Depth", "RMSE - Naive baseline", "RMSE - Random Forest", "RMSE - CNN", "RMSE - ViT", "RMSE - GNN",
            "RMSE - Autoencoder", "RMSE - LSTM", "RMSE - FFNN (headline)", "Correlation (r)", "Bias (model - actual)",
        ]],
        width="stretch",
        hide_index=True,
    )

# ----------------------------------------------------------------------
# TAB 3 - VERTICAL PROFILES (vertical profile, one Argo float at a time)
# ----------------------------------------------------------------------
with tab3:
    st.markdown(
        f'<div class="tab-header sg-tab-profiles">{ICON_PROFILE} <h3>Single-Float Subsurface Reconstruction</h3></div>',
        unsafe_allow_html=True,
    )

    profile_options = list(X_test.index)
    labels = {
        idx: f"{X_test.loc[idx, 'lat']:.2f}N, {X_test.loc[idx, 'lon']:.2f}E "
             f"(day {int(X_test.loc[idx, 'day'])})"
        for idx in profile_options
    }
    chosen_idx = st.selectbox(
        "Argo profile (test set)", profile_options, format_func=lambda i: labels[i]
    )

    depth_cols = [f"temp_{z}m" for z in DEPTH_LEVELS]
    actual_profile = Y_test.loc[chosen_idx, depth_cols].values
    pred_profile = preds.loc[chosen_idx, depth_cols].values

    fig_profile = go.Figure()
    fig_profile.add_trace(
        go.Scatter(
            x=actual_profile, y=DEPTH_LEVELS, mode="lines+markers",
            name="Actual (Argo)", line=dict(color="#3fcf8e", width=3),
            marker=dict(size=9, color="#3fcf8e", line=dict(width=1, color="#061412")),
        )
    )
    fig_profile.add_trace(
        go.Scatter(
            x=pred_profile, y=DEPTH_LEVELS, mode="lines+markers",
            name="Predicted (from surface only)", line=dict(color="#e58a3a", width=3, dash="dash"),
            marker=dict(size=9, symbol="square", color="#e58a3a", line=dict(width=1, color="#061412")),
        )
    )
    fig_profile.update_layout(
        title="Pick a Float - See the Guess vs. Reality",
        xaxis_title="Temperature (&deg;C)",
        yaxis_title="Depth (m)",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=-0.15),
        height=500,
    )
    st.plotly_chart(apply_theme(fig_profile), width="stretch")

# ----------------------------------------------------------------------
# TAB 4 - DEPTH SCATTER (per depth level)
# ----------------------------------------------------------------------
with tab4:
    st.markdown(
        f'<div class="tab-header sg-tab-depth">{ICON_SCATTER} <h3>Cross-Basin Correlation by Depth Level</h3></div>',
        unsafe_allow_html=True,
    )

    depth_choice = st.selectbox("Depth level", DEPTH_LEVELS, index=1, format_func=lambda z: f"{z} m")
    col = f"temp_{depth_choice}m"

    lo = float(min(Y_test[col].min(), preds[col].min()))
    hi = float(max(Y_test[col].max(), preds[col].max()))

    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=Y_test[col], y=preds[col], mode="markers",
            marker=dict(
                size=9,
                color="#09544e",  # Darker, rich teal spot color for high contrast
                opacity=0.95,     # Increased opacity (was 0.65)
                line=dict(width=1.2, color="#3fcf8e"),  # Crisp border line around spots
            ),
            name="Argo profiles (test set)",
            hovertemplate="actual %{x:.2f}&deg;C<br>predicted %{y:.2f}&deg;C<extra></extra>",
        )
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="#e2543f", dash="dash", width=2.5),
            name="Perfect prediction",
        )
    )
    fig_scatter.update_layout(
        title="Guess Vs Reality",
        xaxis_title=f"Actual temp at {depth_choice}m (&deg;C)",
        yaxis_title=f"Predicted temp at {depth_choice}m (&deg;C)",
        legend=dict(orientation="h", y=-0.15),
        height=500,
    )
    st.plotly_chart(apply_theme(fig_scatter), width="stretch")
