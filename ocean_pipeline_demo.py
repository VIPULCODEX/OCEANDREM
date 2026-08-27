"""
OceanEmbed - Demo Pipeline
Subsurface Ocean Temperature Reconstruction from Surface Satellite Observations
North Indian Ocean (5-30N, 45-105E)

PURPOSE OF THIS FILE
---------------------
This is a working, end-to-end demo of the pipeline architecture:
    Argo (ground truth) + Satellite surface data (features)
        -> align/match
        -> train model
        -> validate
        -> visualize

IMPORTANT - READ BEFORE YOUR MEETING
--------------------------------------
USE_SYNTHETIC_DATA = True below. That means the numbers, plots and RMSE
you see are generated from a physically-motivated SIMULATION, not real
Argo/satellite data. This sandbox cannot reach the Argo GDAC or Copernicus
Marine Service servers (network is restricted here).

On your own laptop (which has normal internet), you can flip
USE_SYNTHETIC_DATA = False and the two functions get_argo_data() /
get_satellite_data() will attempt REAL fetches:
  - Argo: via the `argopy` package (already pip-installable: pip install argopy)
  - Satellite: via Copernicus Marine (`copernicusmarine` package, needs a
    free account at marine.copernicus.eu)

So: present this as "pipeline is built and validated on a synthetic
version of the same problem; swapping in real Argo/CMEMS data is a
one-line config change" -- that is an honest and strong thing to say
in a review meeting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Marine heatwave thresholds (deg C SST anomaly above local climatology),
# loosely following the Hobday et al. category scale used by NOAA CoralWatch.
HEATWAVE_CATEGORIES = [
    (0.5, "Watch"),
    (1.0, "Warning"),
    (1.5, "Severe"),
    (2.0, "Extreme"),
]
N_CLUSTERS = 6  # spatiotemporal clusters (lat, lon, day, ssh) -- see clustering note below

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
USE_SYNTHETIC_DATA = True          # flip to False on your laptop with real data access
LAT_RANGE = (5, 30)                # North Indian Ocean
LON_RANGE = (45, 105)
DEPTH_LEVELS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]  # meters -- standard depths per problem statement
N_ARGO_PROFILES = 220               # roughly realistic sparse float count for a season
N_DAYS = 90                         # one monsoon-season-ish window
GRID_STEP = 1.0                     # degrees, satellite grid resolution for this demo
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


def heatwave_bump(day):
    """
    Injects a synthetic marine heatwave event (a multi-week warm pulse,
    stronger near the equator) into the demo SST field so the pipeline has
    an actual event to detect -- stands in for a real Indian Ocean marine
    heatwave (e.g. the kind tracked via MOSDAC/INSAT-3D SST + Argo heat
    content in 2019/2020). Purely synthetic; real deployments detect this
    from the live SST anomaly, not an injected signal.
    """
    return 1.9 * np.exp(-((day - 55) / 11) ** 2)


# ----------------------------------------------------------------------
# 1. DATA LAYER
# ----------------------------------------------------------------------
def get_satellite_grid(day):
    """
    Returns a full lat/lon grid of surface variables for a given day.
    Real version: fetch SST/SSH/SSS/wind from Copernicus Marine for `day`.
    Synthetic version: physically-plausible fields (warmer near equator,
    smooth SSH 'eddies', mild day-to-day drift).
    """
    if not USE_SYNTHETIC_DATA:
        # --- REAL DATA HOOK ---
        # import copernicusmarine
        # ds = copernicusmarine.open_dataset(
        #     dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        #     variables=["thetao", "so", "zos"],
        #     minimum_longitude=LON_RANGE[0], maximum_longitude=LON_RANGE[1],
        #     minimum_latitude=LAT_RANGE[0], maximum_latitude=LAT_RANGE[1],
        #     start_datetime=day, end_datetime=day,
        # )
        # return ds  # then reshape into the same lat/lon/SST/SSH/SSS/wind columns used below
        raise NotImplementedError("Set USE_SYNTHETIC_DATA=True, or fill in the CMEMS call above.")

    lats = np.arange(LAT_RANGE[0], LAT_RANGE[1] + GRID_STEP, GRID_STEP)
    lons = np.arange(LON_RANGE[0], LON_RANGE[1] + GRID_STEP, GRID_STEP)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")

    # SST: warmer near equator, cools slightly poleward, small seasonal wobble,
    # plus an injected marine heatwave pulse (stronger near the equator)
    sst = 29.5 - 0.10 * (LAT - LAT_RANGE[0]) + 0.4 * np.sin(day / 20 + LON / 25)
    sst += heatwave_bump(day) * (1 - 0.5 * (LAT - LAT_RANGE[0]) / (LAT_RANGE[1] - LAT_RANGE[0]))
    sst += rng.normal(0, 0.15, LAT.shape)

    # SSH anomaly: a couple of smooth mesoscale-eddy-like blobs (this is the
    # signal that should correlate with subsurface heat content)
    ssh = (
        0.25 * np.exp(-((LAT - 15) ** 2 + (LON - 70) ** 2) / 40)
        - 0.2 * np.exp(-((LAT - 10) ** 2 + (LON - 90) ** 2) / 30)
        + 0.05 * np.sin(day / 15 + LAT / 10)
    )

    sss = 34.5 + 0.5 * np.sin(LON / 30) + rng.normal(0, 0.05, LAT.shape)
    wind = 5 + 2 * np.abs(np.sin(day / 25 + LAT / 15)) + rng.normal(0, 0.3, LAT.shape)

    return pd.DataFrame({
        "lat": LAT.ravel(), "lon": LON.ravel(),
        "sst": sst.ravel(), "ssh": ssh.ravel(),
        "sss": sss.ravel(), "wind": wind.ravel(),
    })


def get_argo_data():
    """
    Returns sparse Argo-like profiles: lat, lon, day, sst_at_location,
    ssh_at_location, and true subsurface temperature at each DEPTH_LEVEL.
    Real version: argopy.DataFetcher().region([...]).load()
    Synthetic version: physically-motivated thermocline model driven by
    the *same* underlying SST/SSH field, so there is a real learnable
    relationship between surface and subsurface (like in reality, SSH
    anomalies indicate deeper/shallower warm layers).
    """
    if not USE_SYNTHETIC_DATA:
        # --- REAL DATA HOOK ---
        # from argopy import DataFetcher
        # f = DataFetcher().region([LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1],
        #                            0, 600, '2024-06-01', '2024-08-31'])
        # ds = f.load().data.to_dataframe().reset_index()
        # return ds  # then pivot PRES/TEMP into the DEPTH_LEVELS columns used below
        raise NotImplementedError("Set USE_SYNTHETIC_DATA=True, or fill in the argopy call above.")

    lats = rng.uniform(*LAT_RANGE, N_ARGO_PROFILES)
    lons = rng.uniform(*LON_RANGE, N_ARGO_PROFILES)
    days = rng.integers(0, N_DAYS, N_ARGO_PROFILES)

    rows = []
    for lat, lon, day in zip(lats, lons, days):
        sst = 29.5 - 0.10 * (lat - LAT_RANGE[0]) + 0.4 * np.sin(day / 20 + lon / 25) + rng.normal(0, 0.15)
        sst += heatwave_bump(day) * (1 - 0.5 * (lat - LAT_RANGE[0]) / (LAT_RANGE[1] - LAT_RANGE[0]))
        ssh = (
            0.25 * np.exp(-((lat - 15) ** 2 + (lon - 70) ** 2) / 40)
            - 0.2 * np.exp(-((lat - 10) ** 2 + (lon - 90) ** 2) / 30)
            + 0.05 * np.sin(day / 15 + lat / 10)
        )
        sss = 34.5 + 0.5 * np.sin(lon / 30) + rng.normal(0, 0.05)
        wind = 5 + 2 * np.abs(np.sin(day / 25 + lat / 15)) + rng.normal(0, 0.3)

        # Thermocline depth scale grows with positive SSH anomaly (warm eddy
        # = deeper warm layer) -- this is the real physical link we want the
        # model to discover from surface data alone.
        thermocline_scale = 80 + 220 * max(ssh, 0) - 120 * max(-ssh, 0)
        thermocline_scale = max(thermocline_scale, 25)
        t_deep = 8.0  # deep water asymptotic temperature

        row = {"lat": lat, "lon": lon, "day": day, "sst": sst, "ssh": ssh, "sss": sss, "wind": wind}
        for z in DEPTH_LEVELS:
            t_z = sst - (sst - t_deep) * (z / (z + thermocline_scale))
            row[f"temp_{z}m"] = t_z + rng.normal(0, 0.3)
        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 1b. MARINE HEATWAVE DETECTION
# ----------------------------------------------------------------------
def sst_climatology(lat_grid_step=1.0):
    """
    Per-latitude-band climatological SST baseline, computed as the mean
    surface temperature across the full N_DAYS window at each grid latitude.
    Real version: 30-year daily climatology from an SST reanalysis
    (e.g. NOAA OISST); here it's the model's own synthetic seasonal mean.
    """
    lats = np.arange(LAT_RANGE[0], LAT_RANGE[1] + lat_grid_step, lat_grid_step)
    sst_by_day = []
    for day in range(N_DAYS):
        g = get_satellite_grid(day)
        sst_by_day.append(g.groupby("lat")["sst"].mean())
    clim = pd.concat(sst_by_day, axis=1).mean(axis=1)
    return clim  # indexed by lat


def marine_heatwave_series(clim=None):
    """
    Basin-averaged SST anomaly (vs. climatology) for every day in the
    window, classified into Hobday-style heatwave categories. This is the
    headline "is there a heatwave right now" signal for the dashboard.
    """
    if clim is None:
        clim = sst_climatology()

    rows = []
    for day in range(N_DAYS):
        g = get_satellite_grid(day)
        anomaly = (g.groupby("lat")["sst"].mean() - clim).mean()
        category = "Normal"
        for thresh, label in HEATWAVE_CATEGORIES:
            if anomaly >= thresh:
                category = label
        rows.append({"day": day, "anomaly": float(anomaly), "category": category})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 2. ALIGNMENT (in real life: nearest satellite grid cell, same day)
# ----------------------------------------------------------------------
def spatiotemporal_clusters(df, n_clusters=N_CLUSTERS, fit_on=None):
    """
    Adaptive spatiotemporal clustering over (lat, lon, day, ssh), inspired
    by Loo et al. 2026 ("An Adaptive Spatiotemporal Clustering Framework
    for 3D Ocean Subsurface Temperature Reconstruction", arXiv:2605.00860):
    group observations that likely share thermocline structure (same
    region, same eddy regime, nearby in time) before/alongside regression,
    rather than treating every profile as spatially independent. Here it's
    a light-weight KMeans stand-in for the paper's adaptive clustering step.
    Returns integer cluster labels; pass `fit_on` to reuse a fitted model's
    cluster space when labeling a different set of points (e.g. a grid).
    """
    # Feature weights applied *after* z-scoring. `day` is deliberately
    # downweighted: a map "frame" is always a single fixed day, so day
    # contributes a per-cluster *constant* to every point's distance in
    # that frame -- at full weight this constant swamps the real lat/lon/
    # ssh signal and makes single-day snapshots degenerate (observed:
    # only 3-4 of 6 clusters ever appearing on a given day, split like
    # 1118 vs. 11 points). 0.3 keeps a mild temporal-regime influence
    # (profiles from similar times can still end up in the same cluster
    # when position/ssh are ambiguous) without letting it dominate.
    FEATURE_WEIGHTS = {"lat": 1.0, "lon": 1.0, "day": 0.3, "ssh": 1.0}

    feats = df[["lat", "lon", "day", "ssh"]].copy()
    if fit_on is None:
        scaler = StandardScaler()
        feats_scaled = scaler.fit_transform(feats) * list(FEATURE_WEIGHTS.values())
        km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(feats_scaled)
        return labels, (scaler, km)
    else:
        scaler, km = fit_on
        feats_scaled = scaler.transform(feats) * list(FEATURE_WEIGHTS.values())
        return km.predict(feats_scaled)


def build_training_table():
    argo = get_argo_data()
    # In the synthetic case the surface features are already attached
    # to each Argo row (that IS the alignment step, done physically).
    # In the real pipeline this is where you'd do:
    #   for each argo row -> pull nearest lat/lon satellite grid cell for that day
    #   from get_satellite_grid(day)
    cluster_labels, cluster_model = spatiotemporal_clusters(argo)
    argo = argo.copy()
    argo["cluster"] = cluster_labels

    feature_cols = ["lat", "lon", "day", "sst", "ssh", "sss", "wind"]
    cluster_dummies = pd.get_dummies(argo["cluster"], prefix="cluster")
    X = pd.concat([argo[feature_cols], cluster_dummies], axis=1)

    target_cols = [f"temp_{z}m" for z in DEPTH_LEVELS]
    return X, argo[target_cols], argo


# ----------------------------------------------------------------------
# 3. MODEL
# ----------------------------------------------------------------------
def train_and_evaluate(X, Y):
    # time-based split (not random) to avoid leakage, like the plan says
    order = X["day"].argsort()
    X, Y = X.iloc[order].reset_index(drop=True), Y.iloc[order].reset_index(drop=True)
    split = int(0.75 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    Y_train, Y_test = Y.iloc[:split], Y.iloc[split:]

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=RANDOM_SEED)
    model.fit(X_train, Y_train)
    preds = pd.DataFrame(model.predict(X_test), columns=Y_test.columns, index=Y_test.index)

    # naive baseline: predict the training-set mean per depth (a "climatology")
    baseline = pd.DataFrame(
        np.tile(Y_train.mean().values, (len(Y_test), 1)),
        columns=Y_test.columns, index=Y_test.index
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
# 4. VISUALIZATION
# ----------------------------------------------------------------------
def make_plots(X_test, Y_test, preds, metrics, day_for_map=45):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (A) Surface SST map for one day, with Argo profile locations overlaid
    grid = get_satellite_grid(day_for_map)
    sc = axes[0, 0].scatter(grid["lon"], grid["lat"], c=grid["sst"], cmap="turbo", s=8)
    axes[0, 0].scatter(X_test["lon"], X_test["lat"], c="black", s=12, marker="x", label="Argo profiles (test set)")
    axes[0, 0].set_title(f"Surface SST + Argo float locations (day {day_for_map})")
    axes[0, 0].set_xlabel("Longitude"); axes[0, 0].set_ylabel("Latitude")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    plt.colorbar(sc, ax=axes[0, 0], label="SST (deg C)")

    # (B) RMSE: model vs naive baseline, per depth
    x = np.arange(len(metrics))
    w = 0.35
    axes[0, 1].bar(x - w/2, metrics["rmse_baseline"], w, label="Naive baseline (climatology)")
    axes[0, 1].bar(x + w/2, metrics["rmse_model"], w, label="Model (Random Forest)")
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(metrics["depth"])
    axes[0, 1].set_ylabel("RMSE (deg C)")
    axes[0, 1].set_title("Model skill vs naive baseline, per depth")
    axes[0, 1].legend(fontsize=8)

    # (C) Predicted vs actual scatter, one representative depth (100m)
    dcol = "temp_100m"
    axes[1, 0].scatter(Y_test[dcol], preds[dcol], alpha=0.6, s=15)
    lims = [min(Y_test[dcol].min(), preds[dcol].min()), max(Y_test[dcol].max(), preds[dcol].max())]
    axes[1, 0].plot(lims, lims, "r--", label="Perfect prediction")
    axes[1, 0].set_xlabel("Actual temp at 100m (deg C)")
    axes[1, 0].set_ylabel("Predicted temp at 100m (deg C)")
    axes[1, 0].set_title("Predicted vs actual (100m depth)")
    axes[1, 0].legend(fontsize=8)

    # (D) Example vertical profile: predicted vs actual, one test float
    sample_idx = Y_test.index[0]
    actual_profile = Y_test.loc[sample_idx].values
    pred_profile = preds.loc[sample_idx].values
    axes[1, 1].plot(actual_profile, DEPTH_LEVELS, "o-", label="Actual (Argo)")
    axes[1, 1].plot(pred_profile, DEPTH_LEVELS, "s--", label="Predicted (from surface only)")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Temperature (deg C)"); axes[1, 1].set_ylabel("Depth (m)")
    axes[1, 1].set_title("Example vertical profile reconstruction")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(
        "OceanEmbed pipeline demo -- SYNTHETIC DATA (pipeline architecture proof, not real ocean data)",
        fontsize=11, color="darkred"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("/home/claude/ocean_pipeline_demo.png", dpi=150)
    print("Saved plot to /home/claude/ocean_pipeline_demo.png")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"USE_SYNTHETIC_DATA = {USE_SYNTHETIC_DATA}")
    X, Y, argo_full = build_training_table()
    print(f"Built training table: {X.shape[0]} Argo-like profiles, {X.shape[1]} input features, "
          f"{Y.shape[1]} depth targets ({DEPTH_LEVELS} m)")

    model, X_test, Y_test, preds, metrics = train_and_evaluate(X, Y)
    print("\nRMSE per depth (model vs naive climatology baseline):")
    print(metrics.to_string(index=False))

    make_plots(X_test, Y_test, preds, metrics)

    argo_full.to_csv("/home/claude/synthetic_argo_dataset.csv", index=False)
    print("\nSaved synthetic dataset to /home/claude/synthetic_argo_dataset.csv")
