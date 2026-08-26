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

GRID_SAMPLE_STEP_DAYS = 3   # animate through every 3rd day (30 frames)
GRID_POINT_STRIDE = 2       # coarsen the lat/lon grid for export payload size


def round_list(arr, nd=3):
    return [round(float(v), nd) for v in arr]


def main():
    print("Building training table + training model...")
    X, Y, argo = build_training_table()
    model, X_test, Y_test, preds, metrics = train_and_evaluate(X, Y)

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
        })

    metrics_out = [
        {
            "depth": int(row["depth"].replace("temp_", "").replace("m", "")),
            "rmse_model": round(float(row["rmse_model"]), 4),
            "rmse_baseline": round(float(row["rmse_baseline"]), 4),
        }
        for _, row in metrics.iterrows()
    ]

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
            "avg_rmse_improvement_pct": round(
                100 * (1 - metrics["rmse_model"].mean() / metrics["rmse_baseline"].mean()), 1
            ),
        },
        "heatwave_series": heatwave_out,
        "grids": grids,
        "argo_test": argo_test,
        "metrics": metrics_out,
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
