# Sea Green — OceanEmbed

**Subsurface ocean temperature reconstruction & marine heatwave monitoring for the North Indian Ocean (5–30°N, 45–105°E)**, built from sparse Argo float profiles + dense surface satellite fields.

Argo floats measure subsurface temperature directly but are sparse in space and time. Satellites see the surface (SST, SSH, salinity, wind) continuously and everywhere. This pipeline learns the surface → subsurface relationship with a Random Forest, so subsurface structure can be estimated anywhere in the basin — and tracks basin-wide SST anomaly against climatology to flag marine heatwave events (Hobday-style categories: Watch / Warning / Severe / Extreme).

> ⚠ **Ships on synthetic data.** The pipeline, model, and every chart are real, but the numbers come from a physically-motivated simulation, not a live feed — this was built without outbound network access to ARGO/MOSDAC/Copernicus. Real-data hooks are already wired in `ocean_pipeline_demo.py` (see below).

## Two ways to run this

| | Static dashboard (`/public`) | Full app (`app.py`) |
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
streamlit run app.py
```

For a Space: pick the **Streamlit** SDK, point it at this repo, and it picks up `requirements.txt` and `.streamlit/config.toml` automatically.

## Going live with real data

Two integration points in `ocean_pipeline_demo.py`, both currently gated behind `USE_SYNTHETIC_DATA = True`:

- **`get_argo_data()`** — real Argo profiles via [`argopy`](https://argopy.readthedocs.io/): `DataFetcher().region([lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, start, end])`.
- **`get_satellite_grid()`** — real surface fields. For the Indian Ocean specifically, swap in **MOSDAC / ISRO** INSAT-3D/3DR SST and OSCAT wind products (or Copernicus Marine as a global fallback).

Flip the flag, fill in the two commented API calls, re-run `export_data.py` (or just run `app.py` directly) — everything downstream (training, metrics, heatwave detection, clustering, all charts) is unchanged.

## Methodology notes

- **Marine heatwave detection**: basin-mean SST anomaly vs. a climatology computed across the observation window, classified with Hobday-scale thresholds (0.5 / 1.0 / 1.5 / 2.0 °C).
- **Adaptive spatiotemporal clustering**: `spatiotemporal_clusters()` groups observations by (lat, lon, day, SSH) before regression — a lightweight KMeans stand-in for the clustering framework in Loo et al., *"An Adaptive Spatiotemporal Clustering Framework for 3D Ocean Subsurface Temperature Reconstruction"* (arXiv:2605.00860, 2026), which groups profiles sharing thermocline structure to improve reconstruction. Cluster membership is fed to the model as a feature and can be toggled as a map overlay in the dashboard.
- **Satellite context**: the dashboard embeds a live NASA Worldview view scoped to the study region (loads client-side; needs internet in the viewer's browser, not this repo).

## Repo layout

```
ocean_pipeline_demo.py   # data layer, heatwave detection, clustering, model, training/eval — the shared core
export_data.py           # bakes pipeline output into public/data.json for the static dashboard
app.py                   # Streamlit UI (full interactive app)
public/                  # static dashboard (index.html / style.css / app.js / data.json) — deploy target for Vercel
vercel.json              # points Vercel at public/
.streamlit/config.toml   # Sea Green theme for the Streamlit app
requirements.txt
```

Team **Sea Green**.
