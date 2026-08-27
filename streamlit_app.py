"""
OceanEmbed - Interactive Dashboard
Streamlit UI around the existing pipeline in ocean_pipeline_demo.py.

Reuses (does not reimplement) the data generation / model training / evaluation
logic from ocean_pipeline_demo.py. See that file for the real-data integration
points (argopy / copernicusmarine) and the USE_SYNTHETIC_DATA flag.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from ocean_pipeline_demo import (
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
from dl_pipeline import train_pooled_ffnn_and_evaluate

st.set_page_config(page_title="Sea Green | OceanEmbed", layout="wide", page_icon="🌊")

# ----------------------------------------------------------------------
# CACHED WRAPPERS around the existing pipeline (so tabs don't retrain
# the model / regenerate data on every interaction)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Training Random Forest + FFNN (same test split, for a fair comparison)...")
def cached_train():
    X, Y, argo_full = build_training_table()
    _, _, _, _, rf_metrics = train_and_evaluate(X, Y)
    # FFNN is the headline model (beats RF once N_ARGO_PROFILES >= ~600 --
    # see PROJECT_REPORT.txt); its predictions drive every chart below.
    model, X_test, Y_test, preds, metrics = train_pooled_ffnn_and_evaluate(X, Y)
    metrics = metrics.merge(
        rf_metrics[["depth", "rmse_model"]].rename(columns={"rmse_model": "rmse_rf"}),
        on="depth",
    )
    return X_test, Y_test, preds, metrics


@st.cache_data(show_spinner="Fetching satellite grid...")
def cached_grid(day):
    return get_satellite_grid(day)


@st.cache_data(show_spinner="Scanning for marine heatwave events...")
def cached_heatwave():
    return marine_heatwave_series()


X_test, Y_test, preds, metrics = cached_train()
heatwave = cached_heatwave()

# ----------------------------------------------------------------------
# BANNER - always visible, do not remove.
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <div style="background-color:#7a1f1f; color:white; padding:12px 18px;
                border-radius:6px; margin-bottom:16px; font-size:15px;">
        <b>&#9888; SYNTHETIC DEMO DATA</b> &mdash; pipeline validated on simulated data
        (USE_SYNTHETIC_DATA = {USE_SYNTHETIC_DATA}); real Argo/MOSDAC integration point
        marked in <code>ocean_pipeline_demo.py</code> (see <code>get_satellite_grid()</code>
        / <code>get_argo_data()</code>). Flip the flag + fill in the commented
        <code>argopy</code> / <code>copernicusmarine</code> calls to go live.
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("🌊 Sea Green &mdash; OceanEmbed Subsurface Temperature Reconstruction")
st.caption("North Indian Ocean marine heatwave monitoring, from surface satellite data to subsurface structure.")

tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["Marine Heatwave", "Overview", "Model Performance", "Explore Predictions", "Predicted vs Actual"]
)

# ----------------------------------------------------------------------
# TAB 0 - MARINE HEATWAVE MONITOR
# ----------------------------------------------------------------------
with tab0:
    st.subheader("Basin-averaged SST anomaly vs. climatology")

    peak = heatwave.loc[heatwave["anomaly"].idxmax()]
    c1, c2, c3 = st.columns(3)
    c1.metric("Peak anomaly", f"{peak['anomaly']:.2f} °C", peak["category"])
    c2.metric("Days monitored", N_DAYS)
    c3.metric(
        "Days at Watch level or above",
        int((heatwave["category"] != "Normal").sum()),
    )

    color_map = {
        "Normal": "#3fcf8e", "Watch": "#e3c23c", "Warning": "#e58a3a",
        "Severe": "#e2543f", "Extreme": "#b5203a",
    }
    fig_hw = go.Figure()
    fig_hw.add_trace(
        go.Scatter(
            x=heatwave["day"], y=heatwave["anomaly"], mode="lines",
            line=dict(color="#2e8b57", width=2), name="SST anomaly",
        )
    )
    fig_hw.add_trace(
        go.Scatter(
            x=heatwave["day"], y=heatwave["anomaly"], mode="markers",
            marker=dict(size=6, color=heatwave["category"].map(color_map)),
            name="Category", showlegend=False,
        )
    )
    for thresh, label in [(0.5, "Watch"), (1.0, "Warning"), (1.5, "Severe"), (2.0, "Extreme")]:
        fig_hw.add_hline(y=thresh, line_dash="dot", line_color=color_map[label],
                          annotation_text=label, annotation_position="right")
    fig_hw.update_layout(
        xaxis_title="Day of season window", yaxis_title="SST anomaly (&deg;C)",
        height=450, showlegend=False,
    )
    st.plotly_chart(fig_hw, width="stretch")
    st.caption(
        "Categories follow the Hobday marine-heatwave scale (Watch ≥ 0.5°C, "
        "Warning ≥ 1.0°C, Severe ≥ 1.5°C, Extreme ≥ 2.0°C above climatology)."
    )

# ----------------------------------------------------------------------
# TAB 1 - OVERVIEW
# ----------------------------------------------------------------------
with tab1:
    st.markdown(
        """
        Argo floats measure subsurface ocean temperature directly, but they are
        sparse in space and time. Satellites see the surface (SST, SSH, salinity,
        wind) continuously and everywhere. This pipeline learns the relationship
        between surface satellite observations and subsurface temperature at
        several depths, so we can estimate subsurface structure in places and
        times where no Argo float was present.
        """
    )

    day_for_map = st.slider("Day (of the season window)", 0, N_DAYS - 1, 45)
    grid = cached_grid(day_for_map)

    fig_map = go.Figure()
    fig_map.add_trace(
        go.Scatter(
            x=grid["lon"], y=grid["lat"],
            mode="markers",
            marker=dict(
                size=8,
                color=grid["sst"],
                colorscale="Thermal",
                colorbar=dict(title="SST (&deg;C)"),
            ),
            name="Satellite SST grid",
            hovertemplate="lon %{x:.1f}, lat %{y:.1f}<br>SST %{marker.color:.2f}&deg;C<extra></extra>",
        )
    )
    fig_map.add_trace(
        go.Scatter(
            x=X_test["lon"], y=X_test["lat"],
            mode="markers",
            marker=dict(size=9, color="black", symbol="x"),
            name="Argo profiles (test set)",
            hovertemplate="Argo float<br>lon %{x:.2f}, lat %{y:.2f}<extra></extra>",
        )
    )
    fig_map.update_layout(
        title=f"Surface SST + Argo float locations (day {day_for_map})",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        xaxis=dict(range=list(LON_RANGE)),
        yaxis=dict(range=list(LAT_RANGE)),
        legend=dict(orientation="h", y=-0.15),
        height=550,
    )
    st.plotly_chart(fig_map, width="stretch")

# ----------------------------------------------------------------------
# TAB 2 - MODEL PERFORMANCE
# ----------------------------------------------------------------------
with tab2:
    st.subheader("RMSE per depth: model vs naive baseline")

    metrics_long = metrics.melt(
        id_vars="depth",
        value_vars=["rmse_baseline", "rmse_rf", "rmse_model"],
        var_name="method",
        value_name="rmse",
    )
    metrics_long["method"] = metrics_long["method"].map(
        {
            "rmse_baseline": "Naive baseline (climatology)",
            "rmse_rf": "Random Forest",
            "rmse_model": "Neural Network (FFNN)",
        }
    )

    fig_bar = px.bar(
        metrics_long,
        x="depth", y="rmse", color="method",
        barmode="group",
        color_discrete_map={
            "Naive baseline (climatology)": "#9aa0a6",
            "Random Forest": "#4fa3e3",
            "Neural Network (FFNN)": "#2f6fed",
        },
        labels={"depth": "Depth", "rmse": "RMSE (&deg;C)", "method": ""},
    )
    fig_bar.update_layout(legend=dict(orientation="h", y=-0.2), height=450)
    st.plotly_chart(fig_bar, width="stretch")

    st.subheader("Metrics table")
    st.dataframe(
        metrics.rename(
            columns={
                "depth": "Depth",
                "rmse_baseline": "RMSE - Naive baseline",
                "rmse_rf": "RMSE - Random Forest",
                "rmse_model": "RMSE - Neural Network",
                "correlation": "Correlation (r)",
                "bias": "Bias (model - actual)",
            }
        )[["Depth", "RMSE - Naive baseline", "RMSE - Random Forest", "RMSE - Neural Network", "Correlation (r)", "Bias (model - actual)"]],
        width="stretch",
        hide_index=True,
    )

# ----------------------------------------------------------------------
# TAB 3 - EXPLORE PREDICTIONS (vertical profile, one Argo float at a time)
# ----------------------------------------------------------------------
with tab3:
    st.subheader("Predicted vs actual vertical temperature profile")

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
            name="Actual (Argo)", line=dict(color="#2f6fed", width=2),
            marker=dict(size=9),
        )
    )
    fig_profile.add_trace(
        go.Scatter(
            x=pred_profile, y=DEPTH_LEVELS, mode="lines+markers",
            name="Predicted (from surface only)", line=dict(color="#e07b39", width=2, dash="dash"),
            marker=dict(size=9, symbol="square"),
        )
    )
    fig_profile.update_layout(
        xaxis_title="Temperature (&deg;C)",
        yaxis_title="Depth (m)",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=-0.15),
        height=500,
    )
    st.plotly_chart(fig_profile, width="stretch")

# ----------------------------------------------------------------------
# TAB 4 - PREDICTED VS ACTUAL SCATTER (per depth level)
# ----------------------------------------------------------------------
with tab4:
    st.subheader("Predicted vs actual, per depth level")

    depth_choice = st.selectbox("Depth level", DEPTH_LEVELS, index=1, format_func=lambda z: f"{z} m")
    col = f"temp_{depth_choice}m"

    lo = float(min(Y_test[col].min(), preds[col].min()))
    hi = float(max(Y_test[col].max(), preds[col].max()))

    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=Y_test[col], y=preds[col], mode="markers",
            marker=dict(size=8, color="#2f6fed", opacity=0.65),
            name="Argo profiles (test set)",
            hovertemplate="actual %{x:.2f}&deg;C<br>predicted %{y:.2f}&deg;C<extra></extra>",
        )
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="#9aa0a6", dash="dash", width=2),
            name="Perfect prediction",
        )
    )
    fig_scatter.update_layout(
        xaxis_title=f"Actual temp at {depth_choice}m (&deg;C)",
        yaxis_title=f"Predicted temp at {depth_choice}m (&deg;C)",
        legend=dict(orientation="h", y=-0.15),
        height=500,
    )
    st.plotly_chart(fig_scatter, width="stretch")
