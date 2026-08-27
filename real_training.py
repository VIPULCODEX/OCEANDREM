"""
Sea Green -- REAL training data, not synthetic.

Builds actual (surface, subsurface) training pairs from the real files in
this folder:
  - Sea Surface temp.nc                      real SST  (CMEMS thetao, ~0.49m)
  - Sea surface salinity.nc                   real SSS  (CMEMS so)
  - cmems_mod_glo_phy-cur_..._.nc             real surface currents (uo, vo)
  - cmems_mod_glo_phy-thetao_..._0.49-902m.nc real SUBSURFACE temperature,
                                               35 native depth levels,
                                               0.49m to 902m -- THIS is the
                                               piece that was missing before:
                                               a real training TARGET.

Honest scope: only SST, SSS, and surface currents are used as input
features here, because those are the only ones with real data. No
wind/curl/SSH -- those stay synthetic-only (see ocean_pipeline_demo.py)
until a real wind/altimetry file is obtained. This is a smaller feature
set than the synthetic pipeline's, on purpose -- real features only, no
mixing.

Depths: the real subsurface file's native levels don't land exactly on the
spec's 15 standard depths, so each standard depth is matched to its
NEAREST native level (see DEPTH_MATCH). The deepest native level actually
available (given how the CMEMS subset was requested) is ~902m, so the
"1000m" target uses ~902m as an approximation -- documented, not hidden.

Run with: python real_training.py
"""
import glob
import numpy as np
import pandas as pd
import xarray as xr

LAT_RANGE = (5, 30)
LON_RANGE = (45, 99.9)  # capped where SST/SSS/currents actually have data
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
REAL_FEATURE_COLS = ["lat", "lon", "day", "sst", "sss", "uo", "vo"]


def _find(pattern_contains_all):
    """Find the .nc file whose name contains all given substrings (case-sens)."""
    for f in glob.glob("*.nc"):
        if all(s in f for s in pattern_contains_all):
            return f
    raise FileNotFoundError(f"No .nc file matching {pattern_contains_all}")


def build_real_training_table(n_samples=800, seed=42, return_grids=False):
    """Returns (X, Y) DataFrames of REAL training pairs: real surface
    features -> real subsurface temperature at (nearest-matched) standard
    depths. n_samples random valid ocean (lat, lon, day) points, sampled
    from the actual data (not generated)."""
    thetao_path = _find(["thetao", "902"])  # the multi-depth file
    sst_path = "Sea Surface temp.nc"
    sss_path = "Sea surface salinity.nc"
    cur_path = _find(["phy-cur"])

    thetao = xr.open_dataset(thetao_path)["thetao"].sel(
        latitude=slice(*LAT_RANGE), longitude=slice(*LON_RANGE)
    )
    sst_raw = xr.open_dataset(sst_path)["thetao"].isel(depth=0).sel(
        latitude=slice(*LAT_RANGE), longitude=slice(*LON_RANGE)
    )
    sst_daily = sst_raw.resample(time="1D").mean()  # 6-hourly -> daily, matches the others
    sss = xr.open_dataset(sss_path)["so"].isel(depth=0).sel(
        latitude=slice(*LAT_RANGE), longitude=slice(*LON_RANGE)
    )
    cur = xr.open_dataset(cur_path).isel(depth=0).sel(
        latitude=slice(*LAT_RANGE), longitude=slice(*LON_RANGE)
    )

    print("Loading subsetted arrays into memory (faster than repeated disk reads)...")
    thetao = thetao.load()
    sst_daily = sst_daily.load()
    sss = sss.load()
    cur = cur.load()

    # Common time window across all four real sources.
    common_days = sorted(
        set(pd.to_datetime(thetao.time.values).normalize())
        & set(pd.to_datetime(sst_daily.time.values).normalize())
        & set(pd.to_datetime(sss.time.values).normalize())
        & set(pd.to_datetime(cur.time.values).normalize())
    )
    thetao = thetao.sel(time=common_days)
    sst_daily = sst_daily.sel(time=common_days)
    sss = sss.sel(time=common_days)
    cur = cur.sel(time=common_days)

    # Nearest native depth index for each standard depth (documented approx).
    native_depths = thetao.depth.values
    depth_idx = [int(np.argmin(np.abs(native_depths - z))) for z in STANDARD_DEPTHS]
    matched_depths = native_depths[depth_idx]
    print("Standard depth -> nearest native level (m):")
    for z, nz in zip(STANDARD_DEPTHS, matched_depths):
        print(f"  {z:>5} -> {nz:.2f}")

    rng = np.random.default_rng(seed)
    lats, lons = thetao.latitude.values, thetao.longitude.values
    n_lat, n_lon, n_time = len(lats), len(lons), len(common_days)

    rows = []
    attempts = 0
    while len(rows) < n_samples and attempts < n_samples * 20:
        attempts += 1
        i, j, t = rng.integers(n_lat), rng.integers(n_lon), rng.integers(n_time)
        surf_t = float(thetao.isel(latitude=i, longitude=j, time=t, depth=depth_idx[0]).values)
        if np.isnan(surf_t):
            continue  # land
        sst_v = float(sst_daily.isel(latitude=i, longitude=j, time=t).values)
        sss_v = float(sss.isel(latitude=i, longitude=j, time=t).values)
        uo_v = float(cur["uo"].isel(latitude=i, longitude=j, time=t).values)
        vo_v = float(cur["vo"].isel(latitude=i, longitude=j, time=t).values)
        if any(np.isnan(v) for v in (sst_v, sss_v, uo_v, vo_v)):
            continue
        depth_vals = thetao.isel(latitude=i, longitude=j, time=t, depth=depth_idx).values
        if np.isnan(depth_vals).any():
            continue

        row = {
            "lat": float(lats[i]), "lon": float(lons[j]), "day": t,
            "sst": sst_v, "sss": sss_v, "uo": uo_v, "vo": vo_v,
        }
        for z, tv in zip(STANDARD_DEPTHS, depth_vals):
            row[f"temp_{z}m"] = float(tv)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nBuilt {len(df)} real training points from {attempts} sampled locations "
          f"({len(df)/attempts:.0%} were valid ocean).")

    X = df[REAL_FEATURE_COLS]
    Y = df[[f"temp_{z}m" for z in STANDARD_DEPTHS]]
    if return_grids:
        return X, Y, {"thetao": thetao, "sst_daily": sst_daily, "sss": sss, "cur": cur}
    return X, Y


PATCH_CHANNELS_REAL = ["sst", "sss", "uo", "vo"]  # all real -- no ssh/curl available for real data


def build_real_patches(X, thetao, sst_daily, sss, cur, half=2):
    """(N, 4, 5, 5) real satellite-grid patches [sst, sss, uo, vo] for each
    row of X, matching its (lat, lon, day). NaN (land) cells within a
    patch window are filled with that patch's own valid-cell mean per
    channel (simple, standard imputation for a mixed land/ocean window)."""
    lats, lons = thetao.latitude.values, thetao.longitude.values
    arrs = {
        "sst": sst_daily.values, "sss": sss.values,
        "uo": cur["uo"].values, "vo": cur["vo"].values,
    }
    patches = []
    for row in X.itertuples():
        i = int(np.argmin(np.abs(lats - row.lat)))
        j = int(np.argmin(np.abs(lons - row.lon)))
        t = int(row.day)
        ii = np.clip(np.arange(i - half, i + half + 1), 0, len(lats) - 1)
        jj = np.clip(np.arange(j - half, j + half + 1), 0, len(lons) - 1)
        channels = []
        for ch in PATCH_CHANNELS_REAL:
            block = arrs[ch][t][np.ix_(ii, jj)]
            if np.isnan(block).any():
                fill = np.nanmean(block) if not np.isnan(block).all() else 0.0
                block = np.where(np.isnan(block), fill, block)
            channels.append(block)
        patches.append(np.stack(channels))
    return np.stack(patches).astype(np.float32)


def _metrics_frame(Y_test, preds, Y_train):
    from sklearn.metrics import mean_squared_error
    baseline = pd.DataFrame(
        np.tile(Y_train.mean().values, (len(Y_test), 1)), columns=Y_test.columns, index=Y_test.index
    )
    rows = []
    for col in Y_test.columns:
        rmse_model = mean_squared_error(Y_test[col], preds[col]) ** 0.5
        rmse_base = mean_squared_error(Y_test[col], baseline[col]) ** 0.5
        corr = np.corrcoef(Y_test[col], preds[col])[0, 1]
        bias = float((preds[col] - Y_test[col]).mean())
        rows.append({"depth": col, "rmse_model": rmse_model, "rmse_baseline": rmse_base,
                      "correlation": corr, "bias": bias})
    return pd.DataFrame(rows)


def train_real_and_evaluate(X, Y, grids, frac=0.75):
    """Trains all 7 models (RF + 6 neural nets) on REAL data (this module's
    X, Y), same time-based split convention as the synthetic pipeline.
    Reuses the architecture classes from dl_pipeline.py unchanged -- only
    the feature set (7 real columns, not 10) and patch channels (4 real
    channels, not 4 synthetic ones) differ. Returns a dict with
    X_test/Y_test/preds/metrics per model."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import torch
    import torch.nn as nn
    import dl_pipeline as dlp

    order = X["day"].argsort()
    X, Y = X.iloc[order].reset_index(drop=True), Y.iloc[order].reset_index(drop=True)
    split = int(frac * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    Y_train, Y_test = Y.iloc[:split], Y.iloc[split:]
    n_train = len(X_train)

    results = {}

    def add(key, preds):
        results[key] = {"preds": preds, "metrics": _metrics_frame(Y_test, preds, Y_train)}

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train, Y_train)
    add("rf", pd.DataFrame(rf.predict(X_test), columns=Y_test.columns, index=Y_test.index))

    # --- Tabular neural nets: FFNN, GNN, Autoencoder, LSTM (7 real features) ---
    x_scaler = StandardScaler().fit(X_train[REAL_FEATURE_COLS])
    y_scaler = StandardScaler().fit(Y_train)
    x_train_s = x_scaler.transform(X_train[REAL_FEATURE_COLS]).astype(np.float32)
    x_test_s = x_scaler.transform(X_test[REAL_FEATURE_COLS]).astype(np.float32)
    y_train_s = y_scaler.transform(Y_train).astype(np.float32)

    def inv(raw):
        return pd.DataFrame(y_scaler.inverse_transform(raw), columns=Y_test.columns, index=Y_test.index)

    ffnn = dlp.train_ffnn(x_train_s, y_train_s)
    add("ffnn", inv(dlp.predict_ffnn(ffnn, x_test_s)))

    # GNN: transductive k-NN graph over train+test real (lat, lon)
    x_all_s = x_scaler.transform(X[REAL_FEATURE_COLS]).astype(np.float32)
    a_norm = dlp.build_knn_adjacency(X[["lat", "lon"]].values, k=6)
    torch.manual_seed(42)
    gnn = dlp.GNN(in_dim=len(REAL_FEATURE_COLS), out_dim=Y.shape[1]).to(dlp.DEVICE)
    opt = torch.optim.Adam(gnn.parameters(), lr=1e-2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.3)
    xt_all, at, yt = (torch.tensor(x_all_s, device=dlp.DEVICE), torch.tensor(a_norm, device=dlp.DEVICE),
                       torch.tensor(y_train_s, device=dlp.DEVICE))
    gnn.train()
    for _ in range(300):
        opt.zero_grad()
        out = gnn(xt_all, at)
        loss = nn.MSELoss()(out[:n_train], yt)
        loss.backward(); opt.step(); sched.step()
    gnn.eval()
    with torch.no_grad():
        out = gnn(xt_all, at)
    add("gnn", inv(out[n_train:].cpu().numpy()))

    # Autoencoder: unsupervised pretrain (no labels) + supervised probe
    torch.manual_seed(42)
    ae = dlp.AutoEncoder(in_dim=len(REAL_FEATURE_COLS)).to(dlp.DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.3)
    xt_train = torch.tensor(x_train_s, device=dlp.DEVICE)
    ae.train()
    for _ in range(300):
        opt.zero_grad()
        recon, _ = ae(xt_train)
        loss = nn.MSELoss()(recon, xt_train)
        loss.backward(); opt.step(); sched.step()
    ae.eval()
    with torch.no_grad():
        _, embed_train = ae(xt_train)
        _, embed_test = ae(torch.tensor(x_test_s, device=dlp.DEVICE))
    probe = dlp.EmbeddingRegressor(embed_dim=8, out_dim=Y.shape[1]).to(dlp.DEVICE)
    opt2 = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=100, gamma=0.3)
    probe.train()
    for _ in range(300):
        opt2.zero_grad()
        loss = nn.MSELoss()(probe(embed_train), yt)
        loss.backward(); opt2.step(); sched2.step()
    probe.eval()
    with torch.no_grad():
        raw = probe(embed_test).cpu().numpy()
    add("autoencoder", inv(raw))

    # LSTM depth decoder
    torch.manual_seed(42)
    depth_embed = torch.tensor((np.array(STANDARD_DEPTHS) / max(STANDARD_DEPTHS)).astype(np.float32), device=dlp.DEVICE)
    lstm = dlp.LSTMDecoder(in_dim=len(REAL_FEATURE_COLS), n_steps=len(STANDARD_DEPTHS)).to(dlp.DEVICE)
    opt = torch.optim.Adam(lstm.parameters(), lr=1e-2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.3)
    lstm.train()
    for _ in range(300):
        opt.zero_grad()
        loss = nn.MSELoss()(lstm(xt_train, depth_embed), yt)
        loss.backward(); opt.step(); sched.step()
    lstm.eval()
    with torch.no_grad():
        raw = lstm(torch.tensor(x_test_s, device=dlp.DEVICE), depth_embed).cpu().numpy()
    add("lstm", inv(raw))

    # --- Patch-based: CNN, ViT (real 5x5 patches: sst/sss/uo/vo) ---
    patch_train = build_real_patches(X_train, grids["thetao"], grids["sst_daily"], grids["sss"], grids["cur"])
    patch_test = build_real_patches(X_test, grids["thetao"], grids["sst_daily"], grids["sss"], grids["cur"])
    ch_mean = patch_train.mean(axis=(0, 2, 3), keepdims=True)
    ch_std = patch_train.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    patch_train = (patch_train - ch_mean) / ch_std
    patch_test = (patch_test - ch_mean) / ch_std
    day_scaler = StandardScaler().fit(X_train[["day"]])
    day_train = day_scaler.transform(X_train[["day"]]).astype(np.float32)
    day_test = day_scaler.transform(X_test[["day"]]).astype(np.float32)

    torch.manual_seed(42)
    cnn = dlp.PatchCNN(in_channels=len(PATCH_CHANNELS_REAL), out_dim=Y.shape[1]).to(dlp.DEVICE)
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.3)
    pt, dt = torch.tensor(patch_train, device=dlp.DEVICE), torch.tensor(day_train, device=dlp.DEVICE)
    cnn.train()
    for _ in range(300):
        opt.zero_grad()
        loss = nn.MSELoss()(cnn(pt, dt), yt)
        loss.backward(); opt.step(); sched.step()
    cnn.eval()
    with torch.no_grad():
        raw = cnn(torch.tensor(patch_test, device=dlp.DEVICE), torch.tensor(day_test, device=dlp.DEVICE)).cpu().numpy()
    add("cnn", inv(raw))

    torch.manual_seed(42)
    n_tokens = (2 * dlp.PATCH_HALF + 1) ** 2
    vit = dlp.PatchViT(in_channels=len(PATCH_CHANNELS_REAL), n_tokens=n_tokens, out_dim=Y.shape[1]).to(dlp.DEVICE)
    opt = torch.optim.Adam(vit.parameters(), lr=1e-2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.3)
    vit.train()
    for _ in range(300):
        opt.zero_grad()
        loss = nn.MSELoss()(vit(pt, dt), yt)
        loss.backward(); opt.step(); sched.step()
    vit.eval()
    with torch.no_grad():
        raw = vit(torch.tensor(patch_test, device=dlp.DEVICE), torch.tensor(day_test, device=dlp.DEVICE)).cpu().numpy()
    add("vit", inv(raw))

    return X_test, Y_test, results


if __name__ == "__main__":
    X, Y, grids = build_real_training_table(return_grids=True)
    print("\nX shape:", X.shape, "Y shape:", Y.shape)

    X_test, Y_test, results = train_real_and_evaluate(X, Y, grids)
    print(f"\nTrain/test: {len(X) - len(X_test)}/{len(X_test)} (real data, time-based split)")
    print(f"\n{'Model':15s} {'Mean RMSE':>10s} {'Baseline':>10s} {'Correlation':>12s}")
    for key, label in [("ffnn", "FFNN"), ("rf", "Random Forest"), ("cnn", "CNN"), ("vit", "ViT"),
                        ("gnn", "GNN"), ("lstm", "LSTM"), ("autoencoder", "Autoencoder")]:
        m = results[key]["metrics"]
        print(f"{label:15s} {m['rmse_model'].mean():>10.4f} {m['rmse_baseline'].mean():>10.4f} {m['correlation'].mean():>12.4f}")
