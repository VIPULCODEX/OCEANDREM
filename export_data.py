"""
Export a static JSON data bundle from the OceanEmbed pipeline for the
Vercel-deployed static dashboard (public/data.json). Run this whenever the
pipeline logic changes:

    python export_data.py

Keeps the site a plain static build (no server / API routes needed), which
is what makes it trivially Vercel-compatible.
"""
import json
import numpy as np
import pandas as pd

from ocean_pipeline_demo import (
    LAT_RANGE, LON_RANGE, DEPTH_LEVELS, N_DAYS, N_ARGO_PROFILES, N_CLUSTERS,
    USE_SYNTHETIC_DATA,
    get_satellite_grid, build_training_table, train_and_evaluate,
    marine_heatwave_series, spatiotemporal_clusters,
)
from dl_pipeline import (
    train_pooled_ffnn_and_evaluate, train_cnn_and_evaluate, train_lstm_and_evaluate,
    train_vit_and_evaluate, train_autoencoder_and_evaluate,
)

# Every model we've trained and tested, same time-based split, same test
# set, same metrics -- (key used in exported JSON, display name, trainer).
# FFNN is the headline model (see PROJECT_REPORT.txt for why); everything
# else is an honestly-reported comparison, not cherry-picked.
MODELS = [
    ("rf", "Random Forest", train_and_evaluate),
    ("cnn", "CNN (satellite patches)", train_cnn_and_evaluate),
    ("vit", "ViT (attention over patches)", train_vit_and_evaluate),
    ("autoencoder", "Autoencoder (unsupervised embedding)", train_autoencoder_and_evaluate),
    ("lstm", "LSTM (depth decoder)", train_lstm_and_evaluate),
    ("ffnn", "FFNN (headline)", train_pooled_ffnn_and_evaluate),
]

GRID_SAMPLE_STEP_DAYS = 3   # animate through every 3rd day (30 frames)
GRID_POINT_STRIDE = 2       # coarsen the lat/lon grid for export payload size


def round_list(arr, nd=3):
    return [round(float(v), nd) for v in arr]


def main():
    print("Building training table...")
    X, Y, argo = build_training_table()

    results = {}
    for key, label, trainer in MODELS:
        print(f"Training {label}...")
        _, X_test, Y_test, preds, metrics = trainer(X, Y)
        results[key] = {"preds": preds, "metrics": metrics}

    # FFNN drives the profile/scatter/map panels (it's the headline model);
    # X_test/Y_test are identical across all models (same time_based_split).
    preds, metrics = results["ffnn"]["preds"], results["ffnn"]["metrics"]
    rf_preds = results["rf"]["preds"]

    print("Computing marine heatwave series...")
    heatwave = marine_heatwave_series()

    print("Sampling satellite grids for animation frames...")
    frame_days = list(range(0, N_DAYS, GRID_SAMPLE_STEP_DAYS))
    grids = []
    cluster_model = None
    # refit a small cluster model on argo for consistent grid overlay coloring
    cluster_labels, cluster_model = spatiotemporal_clusters(argo)

    for day in frame_days:
        g = get_satellite_grid(day)
        g = g.iloc[::GRID_POINT_STRIDE].reset_index(drop=True)
        gclusters = spatiotemporal_clusters(
            g.assign(day=day), fit_on=cluster_model
        )
        grids.append({
            "day": day,
            "lat": round_list(g["lat"], 2),
            "lon": round_list(g["lon"], 2),
            "sst": round_list(g["sst"], 2),
            "ssh": round_list(g["ssh"], 4),
            "cluster": [int(c) for c in gclusters],
        })

    print("Assembling Argo test-set predictions...")
    depth_cols = [f"temp_{z}m" for z in DEPTH_LEVELS]
    argo_test = []
    for idx in X_test.index:
        argo_test.append({
            "id": int(idx),
            "lat": round(float(X_test.loc[idx, "lat"]), 3),
            "lon": round(float(X_test.loc[idx, "lon"]), 3),
            "day": int(X_test.loc[idx, "day"]),
            "cluster": int(argo.loc[idx, "cluster"]),
            "actual": round_list(Y_test.loc[idx, depth_cols], 3),
            "predicted": round_list(preds.loc[idx, depth_cols], 3),
            "predicted_rf": round_list(rf_preds.loc[idx, depth_cols], 3),
        })

    rmse_by_depth = {
        key: dict(zip(r["metrics"]["depth"], r["metrics"]["rmse_model"]))
        for key, r in results.items()
    }
    metrics_out = [
        {
            "depth": int(row["depth"].replace("temp_", "").replace("m", "")),
            "rmse_model": round(float(row["rmse_model"]), 4),
            **{f"rmse_{key}": round(float(rmse_by_depth[key][row["depth"]]), 4) for key in results},
            "rmse_baseline": round(float(row["rmse_baseline"]), 4),
            "correlation": round(float(row["correlation"]), 4),
            "bias": round(float(row["bias"]), 4),
        }
        for _, row in metrics.iterrows()
    ]

    # Compact per-model summary for the Results tab's overview chart --
    # every architecture we actually trained and tested, same test set.
    model_summary = [{"name": "Naive guess", "avg_rmse": round(float(metrics["rmse_baseline"].mean()), 4), "avg_correlation": None}]
    for key, label, _ in MODELS:
        m = results[key]["metrics"]
        model_summary.append({
            "name": label,
            "avg_rmse": round(float(m["rmse_model"].mean()), 4),
            "avg_correlation": round(float(m["correlation"].mean()), 4),
        })

    heatwave_out = [
        {"day": int(r.day), "anomaly": round(float(r.anomaly), 3), "category": r.category}
        for r in heatwave.itertuples()
    ]

    peak = heatwave.loc[heatwave["anomaly"].idxmax()]

    bundle = {
        "meta": {
            "team": "Sea Green",
            "title": "Sea Green — Indian Ocean Marine Heatwave & Subsurface Intelligence",
            "region": {"lat_range": LAT_RANGE, "lon_range": LON_RANGE},
            "depth_levels": DEPTH_LEVELS,
            "n_days": N_DAYS,
            "n_argo_profiles": N_ARGO_PROFILES,
            "n_clusters": N_CLUSTERS,
            "use_synthetic_data": USE_SYNTHETIC_DATA,
            "peak_event": {
                "day": int(peak["day"]),
                "anomaly": round(float(peak["anomaly"]), 2),
                "category": peak["category"],
            },
            "model_name": "Neural Network (FFNN)",
            "baseline_model_name": "Random Forest",
            "avg_rmse_improvement_pct": round(
                100 * (1 - metrics["rmse_model"].mean() / metrics["rmse_baseline"].mean()), 1
            ),
            "avg_rmse_vs_rf_pct": round(
                100 * (1 - metrics["rmse_model"].mean() / results["rf"]["metrics"]["rmse_model"].mean()), 1
            ),
            "avg_correlation": round(float(metrics["correlation"].mean()), 3),
        },
        "heatwave_series": heatwave_out,
        "grids": grids,
        "argo_test": argo_test,
        "metrics": metrics_out,
        "model_summary": model_summary,
    }

    out_path = "public/data.json"
    with open(out_path, "w") as f:
        json.dump(bundle, f)

    import os
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path} ({size_kb:.0f} KB) -- "
          f"{len(grids)} grid frames, {len(argo_test)} test profiles.")


if __name__ == "__main__":
    main()
