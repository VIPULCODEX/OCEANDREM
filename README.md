# Sea Green — OceanEmbed

**Subsurface ocean temperature reconstruction & marine heatwave monitoring for the North Indian Ocean (5–30°N, 45–105°E)**, built from sparse Argo float profiles + dense surface satellite fields.

Argo floats measure subsurface temperature directly but are sparse in space and time. Satellites see the surface (SST, SSH, salinity, wind) continuously and everywhere. This pipeline learns the surface → subsurface relationship with a Random Forest, so subsurface structure can be estimated anywhere in the basin — and tracks basin-wide SST anomaly against climatology to flag marine heatwave events (Hobday-style categories: Watch / Warning / Severe / Extreme).

> ℹ **Mixed real + simulated.** The dashboard's **Live Data** section runs on real
> **MOSDAC (INSAT-3DR) satellite SST** and real **CMEMS `thetao` reanalysis** data
> (current as of 25–26 Aug 2026) — see [Real data](#real-data-mosdac--cmems) below.
> The **Pipeline Validation** section (Argo floats, subsurface reconstruction, RMSE)
> still runs on a physically-motivated simulation with an injected heatwave event,
> since a real multi-depth Argo/subsurface pull wasn't available for this build.
> Real-data hooks for that half are already wired in `ocean_pipeline_demo.py`.

## Two ways to run this

| | Static dashboard (`/public`) | Full app (`streamlit_app.py`) |
|---|---|---|
| Stack | Plain HTML/CSS/JS + Plotly.js (CDN) | Streamlit |
| Deploy target | **Vercel** (zero server, reads a pre-baked `data.json`) | **Hugging Face Spaces** (or any host that runs Python) — use this if the dashboard needs to run the model live / interactively, or if Vercel's function memory limits become an issue |
| Data | Snapshot exported by `export_data.py` | Computed live on each run (cached per session) |

Both read from the same source of truth: `ocean_pipeline_demo.py`.

### Static dashboard → Vercel

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python export_data.py        # (re)builds public/data.json from the pipeline
```

Then push this repo to GitHub and import it in Vercel — `vercel.json` already points `outputDirectory` at `public/`, so it's a zero-config static deploy. No serverless functions, no build step, no cold starts.

To refresh the dashboard with new results, re-run `export_data.py` and commit the updated `public/data.json`.

### Full interactive app → Hugging Face Spaces (or local)

```bash
streamlit run streamlit_app.py
```

For a Space: pick the **Streamlit** SDK, point it at this repo, and it picks up `requirements.txt` and `.streamlit/config.toml` automatically.

## Real data: MOSDAC + CMEMS

`real_data.py` reads two real datasets and `export_real_data.py` bakes them into
`public/data_real.json` (committed; the raw source files are not — they're
100s of MB and gitignored):

- **MOSDAC** — `MOSDAC/*.h5`, INSAT-3DR L2B SST (ISRO/SAC), half-hourly, 25 Aug 2026.
  8 evenly-spaced real passes, cropped to the study region, drive the "Live
  Satellite Pass" panel.
- **CMEMS**, all `GLOBAL_ANALYSISFORECAST_PHY_001_024` / `_BGC_001_028`
  (Mercator Ocean), 1–27 Aug 2026, single near-surface level (~0.49 m):
  - `thetao` (SST) and `so` (SSS) — daily-mean basin trend + live status chip.
  - `uo`/`vo` (surface currents) — a real vector-field snapshot (direction +
    speed), one of the spec's five required input variables.
  - `chl` (chlorophyll, 0.25° BGC product) — a real ecosystem-impact proxy,
    tying back to the problem statement's "marine ecosystems" motivation.

  That's 4 of the spec's 5 required surface inputs now backed by real data —
  only **SSH/SLA** and **surface winds** are still missing.

To refresh with new files, drop them in `MOSDAC/` / the repo root and re-run:

```bash
pip install h5py xarray netCDF4   # only needed for this step
python export_real_data.py
```

Note this file only has one depth level, so it can't feed the multi-depth
subsurface reconstruction — that's why the ML/heatwave-detection demo still
runs on the simulation described above. A real subsurface signal would come
from a multi-depth CMEMS `thetao` pull (via `copernicusmarine`) or real Argo
profiles (via `argopy`); both need supporting infra (an account + working
build toolchain for `argopy`'s dependencies) that wasn't available in this
sprint.

## Going live with real data (subsurface reconstruction)

Two integration points in `ocean_pipeline_demo.py`, both currently gated behind `USE_SYNTHETIC_DATA = True`:

- **`get_argo_data()`** — real Argo profiles via [`argopy`](https://argopy.readthedocs.io/): `DataFetcher().region([lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, start, end])`.
- **`get_satellite_grid()`** — real surface fields. For the Indian Ocean specifically, swap in **MOSDAC / ISRO** INSAT-3D/3DR SST and OSCAT wind products (or Copernicus Marine as a global fallback).

Flip the flag, fill in the two commented API calls, re-run `export_data.py` (or just run `streamlit_app.py` directly) — everything downstream (training, metrics, heatwave detection, clustering, all charts) is unchanged.

## Methodology notes

- **Marine heatwave detection**: basin-mean SST anomaly vs. a climatology computed across the observation window, classified with Hobday-scale thresholds (0.5 / 1.0 / 1.5 / 2.0 °C).
- **Adaptive spatiotemporal clustering**: `spatiotemporal_clusters()` groups observations by (lat, lon, day, SSH) before regression — a lightweight KMeans stand-in for the clustering framework in Loo et al., *"An Adaptive Spatiotemporal Clustering Framework for 3D Ocean Subsurface Temperature Reconstruction"* (arXiv:2605.00860, 2026), which groups profiles sharing thermocline structure to improve reconstruction. Cluster membership is fed to the model as a feature and can be toggled as a map overlay in the dashboard.
- **Satellite context**: the dashboard embeds a live NASA Worldview view scoped to the study region (loads client-side; needs internet in the viewer's browser, not this repo).

## Repo layout

```
ocean_pipeline_demo.py   # simulated data layer, heatwave detection, clustering, model, training/eval
export_data.py           # bakes simulated pipeline output into public/data.json
real_data.py             # loaders for real MOSDAC (.h5) + CMEMS (.nc) files
export_real_data.py      # bakes real data into public/data_real.json
streamlit_app.py                   # Streamlit UI (full interactive app)
public/                  # static dashboard (index.html / style.css / app.js / data*.json) — deploy target for Vercel
vercel.json              # points Vercel at public/
.streamlit/config.toml   # Sea Green theme for the Streamlit app
requirements.txt
MOSDAC/, cmems_*.nc       # real source files (gitignored — large, regenerate data_real.json locally)
```

Team **Sea Green**.
