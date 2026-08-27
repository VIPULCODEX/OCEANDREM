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
    DEPTH_LEVELS, LAT_RANGE, LON_RANGE, GRID_STEP, RANDOM_SEED,
    build_training_table, train_and_evaluate, get_satellite_grid,
    time_based_split, marine_heatwave_series,
)

torch.manual_seed(RANDOM_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_COLS = ["lat", "lon", "day", "sst", "ssh", "sss", "wind"]
PATCH_CHANNELS = ["sst", "ssh", "sss", "wind"]
PATCH_HALF = 2  # 5x5 patch
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
    torch.manual_seed(RANDOM_SEED)  # deterministic regardless of what trained before this call
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


def _metrics_frame(Y_train, Y_test, preds):
    """Shared metrics computation (RMSE/baseline/correlation/bias per depth),
    used identically by every model below so their outputs are directly
    comparable -- same formula, same baseline, same test rows."""
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
    return pd.DataFrame(results)


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

    return model, X_test, Y_test, preds, _metrics_frame(Y_train, Y_test, preds)


# ----------------------------------------------------------------------
# 3b. CNN -- operates on a real spatial patch of the satellite grid
#     (not flattened tabular features), the closest thing in this repo to
#     the "satellite embeddings" the problem statement actually asks for.
# ----------------------------------------------------------------------
class PatchCNN(nn.Module):
    """Small conv net over a (channels, patch, patch) satellite image patch,
    with the scalar `day` feature concatenated in after pooling (day isn't
    spatial, so it doesn't belong inside the convolution)."""

    def __init__(self, in_channels, out_dim, hidden=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, patch, day_scalar):
        embedding = self.conv(patch).flatten(1)          # (batch, hidden) -- the "satellite embedding"
        return self.head(torch.cat([embedding, day_scalar], dim=1))


def _day_grid_arrays(day, _cache={}):
    """2D (lat x lon) arrays per channel for one day's satellite grid,
    cached since many profiles share the same day."""
    if day not in _cache:
        g = get_satellite_grid(day)
        lats = np.arange(LAT_RANGE[0], LAT_RANGE[1] + GRID_STEP, GRID_STEP)
        lons = np.arange(LON_RANGE[0], LON_RANGE[1] + GRID_STEP, GRID_STEP)
        arrays = {ch: g[ch].values.reshape(len(lats), len(lons)) for ch in PATCH_CHANNELS}
        _cache[day] = (lats, lons, arrays)
    return _cache[day]


def extract_patch(lat, lon, day, half=PATCH_HALF):
    """(len(PATCH_CHANNELS), 2*half+1, 2*half+1) patch of the satellite grid
    centered on the nearest grid cell to (lat, lon), for the given day.
    Edge profiles get edge-replicated padding so every patch is the same
    fixed size."""
    lats, lons, arrays = _day_grid_arrays(int(day))
    i = int(np.argmin(np.abs(lats - lat)))
    j = int(np.argmin(np.abs(lons - lon)))
    ii = np.clip(np.arange(i - half, i + half + 1), 0, len(lats) - 1)
    jj = np.clip(np.arange(j - half, j + half + 1), 0, len(lons) - 1)
    return np.stack([arrays[ch][np.ix_(ii, jj)] for ch in PATCH_CHANNELS])


def train_cnn_and_evaluate(X, Y, epochs=300, lr=1e-2):
    """Same interface as train_pooled_ffnn_and_evaluate(). Builds a real
    spatial patch per profile (see extract_patch) instead of using the
    flattened surface features directly."""
    torch.manual_seed(RANDOM_SEED)
    X_train, X_test, Y_train, Y_test = time_based_split(X, Y)

    def build_patches(df):
        return np.stack([extract_patch(r.lat, r.lon, r.day) for r in df.itertuples()]).astype(np.float32)

    patch_train, patch_test = build_patches(X_train), build_patches(X_test)
    # normalize each channel using training-set stats
    ch_mean = patch_train.mean(axis=(0, 2, 3), keepdims=True)
    ch_std = patch_train.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    patch_train = (patch_train - ch_mean) / ch_std
    patch_test = (patch_test - ch_mean) / ch_std

    day_scaler = StandardScaler().fit(X_train[["day"]])
    day_train = day_scaler.transform(X_train[["day"]]).astype(np.float32)
    day_test = day_scaler.transform(X_test[["day"]]).astype(np.float32)

    y_scaler = StandardScaler().fit(Y_train)
    y_train = y_scaler.transform(Y_train).astype(np.float32)

    model = PatchCNN(in_channels=len(PATCH_CHANNELS), out_dim=Y.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(epochs // 3, 1), gamma=0.3)
    loss_fn = nn.MSELoss()

    pt = torch.tensor(patch_train, device=DEVICE)
    dt = torch.tensor(day_train, device=DEVICE)
    yt = torch.tensor(y_train, device=DEVICE)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(pt, dt), yt)
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        raw_preds = model(torch.tensor(patch_test, device=DEVICE), torch.tensor(day_test, device=DEVICE)).cpu().numpy()
    preds = pd.DataFrame(y_scaler.inverse_transform(raw_preds), columns=Y_test.columns, index=Y_test.index)

    return model, X_test, Y_test, preds, _metrics_frame(Y_train, Y_test, preds)


# ----------------------------------------------------------------------
# 3c. LSTM -- decodes the 15-depth profile as a sequence, one step per
#     depth, conditioned on the encoded surface features + which depth
#     the current step is predicting.
# ----------------------------------------------------------------------
class LSTMDecoder(nn.Module):
    def __init__(self, in_dim, n_steps, hidden=32):
        super().__init__()
        self.n_steps = n_steps
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.lstm = nn.LSTM(input_size=hidden + 1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, depth_embed):
        enc = self.encoder(x)                                    # (batch, hidden)
        enc_rep = enc.unsqueeze(1).repeat(1, self.n_steps, 1)     # (batch, steps, hidden)
        depth_rep = depth_embed.view(1, self.n_steps, 1).repeat(x.shape[0], 1, 1)
        out, _ = self.lstm(torch.cat([enc_rep, depth_rep], dim=-1))
        return self.head(out).squeeze(-1)                         # (batch, steps)


def train_lstm_and_evaluate(X, Y, epochs=300, lr=1e-2):
    """Same interface as train_pooled_ffnn_and_evaluate()."""
    torch.manual_seed(RANDOM_SEED)
    X_train, X_test, Y_train, Y_test = time_based_split(X, Y)

    x_scaler = StandardScaler().fit(X_train[FEATURE_COLS])
    y_scaler = StandardScaler().fit(Y_train)
    x_train = x_scaler.transform(X_train[FEATURE_COLS]).astype(np.float32)
    x_test = x_scaler.transform(X_test[FEATURE_COLS]).astype(np.float32)
    y_train = y_scaler.transform(Y_train).astype(np.float32)

    depth_embed = torch.tensor(
        (np.array(DEPTH_LEVELS) / max(DEPTH_LEVELS)).astype(np.float32), device=DEVICE
    )

    model = LSTMDecoder(in_dim=len(FEATURE_COLS), n_steps=len(DEPTH_LEVELS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(epochs // 3, 1), gamma=0.3)
    loss_fn = nn.MSELoss()

    xt = torch.tensor(x_train, device=DEVICE)
    yt = torch.tensor(y_train, device=DEVICE)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt, depth_embed), yt)
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        raw_preds = model(torch.tensor(x_test, device=DEVICE), depth_embed).cpu().numpy()
    preds = pd.DataFrame(y_scaler.inverse_transform(raw_preds), columns=Y_test.columns, index=Y_test.index)

    return model, X_test, Y_test, preds, _metrics_frame(Y_train, Y_test, preds)


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
