"""
Sea Green -- deep learning reconstruction, following the actual method in
Loo et al. 2026 (arXiv:2605.00860): cluster first (by depth AND by time),
then train a separate small model per spatiotemporal sub-block, instead of
one model on everything pooled together. Their finding: clustering-first
beats pooled training by 12-27% RMSE, consistently, across six backbone
architectures (DP-CNN, Attention U-Net, ViT, FFNN, LSTM, OCNN).

This reproduces that comparison at hackathon scale, on our synthetic data,
using the simplest of their six backbones (FFNN):

  1. Vertical clustering: hierarchical clustering (contiguity-constrained,
     so depth bands stay physically sensible) on the mean temperature
     profile -- groups the 15 standard depths into a few bands that behave
     similarly (their Section 4.2).
  2. Temporal clustering: their paper uses PCA + change-point detection on
     the annual cycle (Section 4.3). We don't have `ruptures` available
     (no C compiler in this environment to build it), and we already have
     something arguably more directly useful for our data: a per-day
     marine-heatwave category from ocean_pipeline_demo.py. So temporal
     phases here are just "quiet" (Normal) vs. "event" (Watch or above) --
     a threshold-crossing segmentation, which is actually how real marine
     heatwave detection (Hobday et al.) defines event boundaries anyway.
  3. Train one small FFNN per (vertical band x temporal phase) sub-block,
     each predicting only its band's depths, only from its phase's rows.
     At test time, reassemble each test profile from whichever sub-block
     models match its vertical bands and temporal phase.
  4. Compare RMSE/correlation for: Random Forest (existing baseline),
     pooled FFNN (one network, no clustering), and clustered FFNN
     (this framework) -- on the same held-out test set.

Run with: python dl_pipeline.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from ocean_pipeline_demo import (
    DEPTH_LEVELS, RANDOM_SEED, build_training_table, train_and_evaluate,
    time_based_split, marine_heatwave_series,
)

torch.manual_seed(RANDOM_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_COLS = ["lat", "lon", "day", "sst", "ssh", "sss", "wind"]
N_VERTICAL_CLUSTERS = 3


# ----------------------------------------------------------------------
# 1. VERTICAL CLUSTERING (contiguous depth bands)
# ----------------------------------------------------------------------
def vertical_clusters(argo, n_clusters=N_VERTICAL_CLUSTERS):
    """Groups DEPTH_LEVELS into `n_clusters` contiguous bands by similarity
    of the mean profile, using a tridiagonal connectivity matrix so a
    depth can only merge with its immediate neighbor (keeps bands
    physically contiguous, e.g. no "0m + 500m, but not 50m" cluster)."""
    cols = [f"temp_{z}m" for z in DEPTH_LEVELS]
    profile = argo[cols].mean().values.reshape(-1, 1)

    n = len(DEPTH_LEVELS)
    connectivity = np.zeros((n, n))
    for i in range(n - 1):
        connectivity[i, i + 1] = connectivity[i + 1, i] = 1

    model = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity, linkage="ward")
    labels = model.fit_predict(profile)

    bands = {}
    for band_id in np.unique(labels):
        depths = [DEPTH_LEVELS[i] for i in range(n) if labels[i] == band_id]
        bands[int(band_id)] = depths
    return bands


# ----------------------------------------------------------------------
# 2. TEMPORAL PHASES (quiet vs. event, from the heatwave detector)
# ----------------------------------------------------------------------
def temporal_phases():
    """Returns {day: phase} for every day in the season, where phase is
    'quiet' (Normal) or 'event' (Watch or above)."""
    hw = marine_heatwave_series()
    return {
        int(row.day): ("quiet" if row.category == "Normal" else "event")
        for row in hw.itertuples()
    }


# ----------------------------------------------------------------------
# 3. MODELS
# ----------------------------------------------------------------------
class FFNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_ffnn(X, Y, epochs=300, lr=1e-2, weight_decay=1e-4):
    """Trains one FFNN. X, Y are already-scaled numpy arrays."""
    model = FFNN(X.shape[1], Y.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(epochs // 3, 1), gamma=0.3)
    loss_fn = nn.MSELoss()

    xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(Y, dtype=torch.float32, device=DEVICE)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict_ffnn(model, X):
    model.eval()
    xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    return model(xt).cpu().numpy()


def train_pooled_ffnn_and_evaluate(X, Y):
    """
    Same interface/return shape as ocean_pipeline_demo.train_and_evaluate(),
    so the export script and dashboard can treat this as a drop-in second
    model: (model, X_test, Y_test, preds, metrics_df). Uses the identical
    time-based split as the Random Forest for a fair, apples-to-apples
    comparison. This is the model that ended up winning (see dl_pipeline.py
    module docstring / PROJECT_REPORT.txt for the full comparison and why
    the paper's clustering trick was dropped from the headline claim).
    """
    X_train, X_test, Y_train, Y_test = time_based_split(X, Y)

    x_scaler = StandardScaler().fit(X_train[FEATURE_COLS])
    y_scaler = StandardScaler().fit(Y_train)

    model = train_ffnn(
        x_scaler.transform(X_train[FEATURE_COLS]).astype(np.float32),
        y_scaler.transform(Y_train).astype(np.float32),
    )
    raw_preds = y_scaler.inverse_transform(predict_ffnn(model, x_scaler.transform(X_test[FEATURE_COLS]).astype(np.float32)))
    preds = pd.DataFrame(raw_preds, columns=Y_test.columns, index=Y_test.index)

    baseline = pd.DataFrame(
        np.tile(Y_train.mean().values, (len(Y_test), 1)), columns=Y_test.columns, index=Y_test.index
    )

    results = []
    for col in Y_test.columns:
        rmse_model = mean_squared_error(Y_test[col], preds[col]) ** 0.5
        rmse_base = mean_squared_error(Y_test[col], baseline[col]) ** 0.5
        corr = np.corrcoef(Y_test[col], preds[col])[0, 1]
        bias = float((preds[col] - Y_test[col]).mean())
        results.append({
            "depth": col, "rmse_model": rmse_model, "rmse_baseline": rmse_base,
            "correlation": corr, "bias": bias,
        })

    return model, X_test, Y_test, preds, pd.DataFrame(results)


# ----------------------------------------------------------------------
# 4. EXPERIMENT: pooled vs. clustered FFNN, vs. the existing Random Forest
# ----------------------------------------------------------------------
def run_comparison():
    X, Y, argo = build_training_table()

    # Same time-based split the Random Forest uses, for a fair comparison.
    order = X["day"].argsort()
    X, Y = X.iloc[order].reset_index(drop=True), Y.iloc[order].reset_index(drop=True)
    split = int(0.75 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    Y_train, Y_test = Y.iloc[:split], Y.iloc[split:]

    x_scaler = StandardScaler().fit(X_train[FEATURE_COLS])
    y_scaler = StandardScaler().fit(Y_train)

    def xs(df):
        return x_scaler.transform(df[FEATURE_COLS]).astype(np.float32)

    def ys(df):
        return y_scaler.transform(df).astype(np.float32)

    print(f"Device: {DEVICE} | train rows: {len(X_train)} | test rows: {len(X_test)}")

    # --- Random Forest (existing baseline, unchanged) ---
    print("\n[1/3] Random Forest (existing baseline)...")
    _, _, _, rf_preds, rf_metrics = train_and_evaluate(X, Y)

    # --- Pooled FFNN: one network, no clustering ---
    print("[2/3] Pooled FFNN (one network on everything)...")
    pooled_model = train_ffnn(xs(X_train), ys(Y_train))
    pooled_preds = y_scaler.inverse_transform(predict_ffnn(pooled_model, xs(X_test)))
    pooled_preds = pd.DataFrame(pooled_preds, columns=Y_test.columns, index=Y_test.index)

    # --- Clustered FFNN: the paper's actual method ---
    print("[3/3] Clustered FFNN (per depth-band x time-phase sub-block)...")
    bands = vertical_clusters(argo)
    phases = temporal_phases()
    print(f"  vertical bands: { {k: v for k, v in bands.items()} }")
    phase_series = X_train["day"].map(phases)

    sub_models = {}
    for band_id, depths in bands.items():
        target_cols = [f"temp_{z}m" for z in depths]
        for phase in ["quiet", "event"]:
            rows = phase_series == phase
            n_rows = int(rows.sum())
            if n_rows < 5:
                sub_models[(band_id, phase)] = None
                print(f"  band {band_id} {depths} / {phase}: only {n_rows} rows, skipping (fallback to pooled)")
                continue
            xb = x_scaler.transform(X_train.loc[rows, FEATURE_COLS]).astype(np.float32)
            yb_scaler = StandardScaler().fit(Y_train.loc[rows, target_cols])
            yb = yb_scaler.transform(Y_train.loc[rows, target_cols]).astype(np.float32)
            sub_models[(band_id, phase)] = (train_ffnn(xb, yb, epochs=300), yb_scaler, target_cols)
            print(f"  band {band_id} {depths} / {phase}: {n_rows} rows trained")

    test_phase = X_test["day"].map(phases)
    clustered_preds = pd.DataFrame(index=Y_test.index, columns=Y_test.columns, dtype=float)
    for band_id, depths in bands.items():
        target_cols = [f"temp_{z}m" for z in depths]
        for phase in ["quiet", "event"]:
            rows = test_phase == phase
            if not rows.any():
                continue
            entry = sub_models.get((band_id, phase))
            if entry is None:
                # not enough training rows for this sub-block -- fall back
                # to the pooled model's prediction for these depths/rows
                clustered_preds.loc[rows, target_cols] = pooled_preds.loc[rows, target_cols].values
                continue
            model, yb_scaler, _ = entry
            xb = x_scaler.transform(X_test.loc[rows, FEATURE_COLS]).astype(np.float32)
            preds = yb_scaler.inverse_transform(predict_ffnn(model, xb))
            clustered_preds.loc[rows, target_cols] = preds

    # --- Compare ---
    def rmse_per_depth(preds):
        return {col: float(np.sqrt(np.mean((Y_test[col] - preds[col]) ** 2))) for col in Y_test.columns}

    rf_rmse = dict(zip(rf_metrics["depth"], rf_metrics["rmse_model"]))
    pooled_rmse = rmse_per_depth(pooled_preds)
    clustered_rmse = rmse_per_depth(clustered_preds)

    print("\n" + "=" * 78)
    print(f"{'Depth':>10} | {'RF (RMSE)':>10} | {'Pooled FFNN':>12} | {'Clustered FFNN':>15}")
    print("-" * 78)
    for col in Y_test.columns:
        print(f"{col:>10} | {rf_rmse[col]:>10.3f} | {pooled_rmse[col]:>12.3f} | {clustered_rmse[col]:>15.3f}")

    avg_rf = np.mean(list(rf_rmse.values()))
    avg_pooled = np.mean(list(pooled_rmse.values()))
    avg_clustered = np.mean(list(clustered_rmse.values()))
    print("-" * 78)
    print(f"{'MEAN':>10} | {avg_rf:>10.3f} | {avg_pooled:>12.3f} | {avg_clustered:>15.3f}")
    print("=" * 78)
    improvement = 100 * (1 - avg_clustered / avg_pooled)
    print(f"\nClustering-first vs. pooled FFNN: {improvement:+.1f}% RMSE change "
          f"(paper reports +12.4% to +27.2% improvement)")

    return {
        "bands": bands, "rf_rmse": rf_rmse, "pooled_rmse": pooled_rmse,
        "clustered_rmse": clustered_rmse, "avg_rf": avg_rf,
        "avg_pooled": avg_pooled, "avg_clustered": avg_clustered,
    }


if __name__ == "__main__":
    run_comparison()
