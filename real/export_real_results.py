"""
Bakes the real-data 7-model comparison (Random Forest + 6 neural nets,
trained by real.real_training on genuine MOSDAC/CMEMS-derived surface
and subsurface data) into public/data.json for the static dashboard.
This is the real-data-only replacement for the deleted
synthetic/export_data.py -- there is no synthetic data path in this
project any more, so this script is now the sole producer of
public/data.json.

Run with: python real/export_real_results.py
     (or: python -m real.export_real_results)
"""
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from real.real_training import (
    build_real_training_table, train_real_and_evaluate,
    LAT_RANGE, LON_RANGE, STANDARD_DEPTHS,
)
from real.real_data import HEATWAVE_CATEGORIES, _classify

# Every model real_training.py trains, in the order shown in the
# Results tab's comparison chart/table. Random Forest is the headline
# (empirically the best real-data performer -- see README.md).
MODELS = [
    ("rf", "Random Forest"),
    ("cnn", "CNN (satellite patches)"),
    ("vit", "ViT (attention over patches)"),
    ("gnn", "GNN (k-NN graph)"),
    ("autoencoder", "Autoencoder (unsupervised embedding)"),
    ("lstm", "LSTM (depth decoder)"),
    ("ffnn", "FFNN"),
]
HEADLINE_KEY = "rf"
HEADLINE_LABEL = "Random Forest"

GRID_POINT_STRIDE = 130  # flattened-point stride per frame; picked so each
                          # frame lands in the same few-hundred-point ballpark
                          # as the old synthetic export (native real
                          # resolution is 0.083 deg, far denser than that
                          # was, so a much larger stride is needed here).


def round_list(arr, nd=3):
    return [round(float(v), nd) for v in arr]


def build_clusters(X):
    """K-means over z-scored (lat, lon, day), n_clusters=6 to match
    CLUSTER_PALETTE's length in app.js. Real features have no ssh, so the
    old (lat, lon, day, ssh) weighted-clustering can't be reused as-is --
    this is a simpler direct replacement, not a port of that logic."""
    feats = X[["lat", "lon", "day"]].values.astype(np.float64)
    scaler = StandardScaler().fit(feats)
    feats_s = scaler.transform(feats)
    km = KMeans(n_clusters=6, random_state=42, n_init=10).fit(feats_s)
    counts = np.bincount(km.labels_, minlength=6)
    print(f"Cluster point counts (day not downweighted): {counts.tolist()}")
    return scaler, km


def main():
    print("Building real training table (surface features + real subsurface target)...")
    X, Y, grids = build_real_training_table(n_samples=800, seed=42, return_grids=True)

    print("Training all 7 models on real data...")
    X_test, Y_test, results = train_real_and_evaluate(X, Y, grids)

    headline_preds = results[HEADLINE_KEY]["preds"]
    headline_metrics = results[HEADLINE_KEY]["metrics"]

    print("Fitting region clusters (lat, lon, day) for the map overlay...")
    cluster_scaler, cluster_model = build_clusters(X)

    def cluster_for(lat, lon, day):
        f = cluster_scaler.transform([[lat, lon, day]])
        return int(cluster_model.predict(f)[0])

    print("Assembling per-depth metrics table...")
    depth_cols = [f"temp_{z}m" for z in STANDARD_DEPTHS]
    # rmse_baseline (the naive-guess baseline) is identical across every
    # model's own metrics frame, since it's computed from the same
    # Y_train.mean() regardless of which model produced the predictions --
    # take it from the headline model's frame, no need for a separate one.
    metrics_out = []
    for i, depth_col in enumerate(depth_cols):
        row = {
            "depth": STANDARD_DEPTHS[i],
            "rmse_model": round(float(headline_metrics.iloc[i]["rmse_model"]), 4),
            "rmse_baseline": round(float(headline_metrics.iloc[i]["rmse_baseline"]), 4),
            "correlation": round(float(headline_metrics.iloc[i]["correlation"]), 4),
            "bias": round(float(headline_metrics.iloc[i]["bias"]), 4),
        }
        for key, _ in MODELS:
            row[f"rmse_{key}"] = round(float(results[key]["metrics"].iloc[i]["rmse_model"]), 4)
        metrics_out.append(row)

    print("Assembling model_summary...")
    model_summary = [{
        "name": "Naive guess",
        "avg_rmse": round(float(headline_metrics["rmse_baseline"].mean()), 4),
        "avg_correlation": None,
    }]
    for key, label in MODELS:
        m = results[key]["metrics"]
        model_summary.append({
            "name": label,
            "avg_rmse": round(float(m["rmse_model"].mean()), 4),
            "avg_correlation": round(float(m["correlation"].mean()), 4),
        })

    print("Assembling test-point (argo_test) table...")
    argo_test = []
    for idx in X_test.index:
        lat, lon, day = float(X_test.loc[idx, "lat"]), float(X_test.loc[idx, "lon"]), int(X_test.loc[idx, "day"])
        argo_test.append({
            "id": int(idx),
            "lat": round(lat, 3),
            "lon": round(lon, 3),
            "day": day,
            "cluster": cluster_for(lat, lon, day),
            "actual": round_list(Y_test.loc[idx, depth_cols], 3),
            "predicted": round_list(headline_preds.loc[idx, depth_cols], 3),
        })

    print("Building day-indexed grid animation frames + dense heatwave series...")
    sst_daily = grids["sst_daily"]
    n_days = len(sst_daily.time)
    lat2d, lon2d = np.meshgrid(sst_daily.latitude.values, sst_daily.longitude.values, indexing="ij")
    lat_flat, lon_flat = lat2d.ravel(), lon2d.ravel()

    basin_means = []
    grid_frames = []
    for t_idx in range(n_days):
        vals = sst_daily.isel(time=t_idx).values.ravel()
        mask = np.isfinite(vals)
        basin_means.append(float(np.nanmean(sst_daily.isel(time=t_idx).values)))

        lat_s = lat_flat[mask][::GRID_POINT_STRIDE]
        lon_s = lon_flat[mask][::GRID_POINT_STRIDE]
        val_s = vals[mask][::GRID_POINT_STRIDE]
        clusters_s = [cluster_for(float(la), float(lo), t_idx) for la, lo in zip(lat_s, lon_s)]

        grid_frames.append({
            "day": t_idx,
            "lat": round_list(lat_s, 2),
            "lon": round_list(lon_s, 2),
            "sst": round_list(val_s, 2),
            "cluster": clusters_s,
        })

    window_mean = float(np.mean(basin_means))
    heatwave_out = []
    for t_idx, mean_sst in enumerate(basin_means):
        anomaly = round(mean_sst - window_mean, 3)
        heatwave_out.append({"day": t_idx, "anomaly": anomaly, "category": _classify(anomaly)})

    peak = max(heatwave_out, key=lambda r: r["anomaly"])

    bundle = {
        "meta": {
            "team": "Sea Green",
            "title": "Sea Green — Indian Ocean Marine Heatwave & Subsurface Intelligence",
            "region": {"lat_range": list(LAT_RANGE), "lon_range": list(LON_RANGE)},
            "depth_levels": STANDARD_DEPTHS,
            "n_days": n_days,
            "n_argo_profiles": int(len(X)),
            "n_clusters": 6,
            "use_synthetic_data": False,
            "peak_event": {
                "day": peak["day"],
                "anomaly": peak["anomaly"],
                "category": peak["category"],
            },
            "model_name": HEADLINE_LABEL,
            "baseline_model_name": HEADLINE_LABEL,
            "avg_rmse_improvement_pct": round(
                100 * (1 - headline_metrics["rmse_model"].mean() / headline_metrics["rmse_baseline"].mean()), 1
            ),
            "avg_correlation": round(float(headline_metrics["correlation"].mean()), 3),
        },
        "heatwave_series": heatwave_out,
        "grids": grid_frames,
        "argo_test": argo_test,
        "metrics": metrics_out,
        "model_summary": model_summary,
    }

    out_path = _REPO_ROOT / "public" / "data.json"
    with open(out_path, "w") as f:
        json.dump(bundle, f)

    import os
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nWrote {out_path} ({size_kb:.0f} KB) -- "
          f"{len(grid_frames)} grid frames, {len(argo_test)} test points.")

    print(f"\n{'Model':30s} {'Mean RMSE':>10s} {'Correlation':>12s}")
    print(f"{'Naive guess':30s} {model_summary[0]['avg_rmse']:>10.4f} {'--':>12s}")
    for row in model_summary[1:]:
        print(f"{row['name']:30s} {row['avg_rmse']:>10.4f} {row['avg_correlation']:>12.4f}")


if __name__ == "__main__":
    main()
