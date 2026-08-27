# Sea Green — OceanEmbed

**Subsurface ocean temperature reconstruction & marine heatwave monitoring for the North Indian Ocean (5–30°N, 45–105°E)**, built from sparse Argo float profiles + dense surface satellite fields.

Argo floats measure subsurface temperature directly but are sparse in space and time. Satellites see the surface (SST, SSH, salinity, wind) continuously and everywhere. This pipeline trains a neural network (FFNN) to learn the surface → subsurface relationship, so subsurface structure can be estimated anywhere in the basin — and tracks basin-wide SST anomaly against climatology to flag marine heatwave events (Hobday-style categories: Watch / Warning / Severe / Extreme).

The dashboard (`public/`) is organized as four tabs: **Live Monitor** (real current data), **The Model** (the reconstruction demo), **Results** (metrics/claims), **How it Works** (methodology, for anyone who wants the detail — kept out of the way of the main flow).

> ℹ **Mixed real + simulated.** The **Live Monitor** tab runs on real
> **MOSDAC (INSAT-3DR) satellite SST** and real **CMEMS** SST/SSS/currents/chlorophyll
> (current as of Aug 2026) — see [Real data](#real-data-mosdac--cmems) below.
> **The Model** and **Results** tabs (Argo floats, subsurface reconstruction, RMSE)
> run on a physically-motivated simulation with an injected heatwave event, since a
> real multi-depth Argo/subsurface pull wasn't available for this build. Real-data
> hooks for that half are already wired in `ocean_pipeline_demo.py`.

## The models

`dl_pipeline.py` trains **six independent models** on the synthetic Argo dataset,
all on the *identical* time-based train/test split (`time_based_split()` in
`ocean_pipeline_demo.py`) for a fair comparison — all six (plus a naive baseline)
are shown side by side in the dashboard's Results tab:

| Model | Type | Input | Mean RMSE | vs. baseline |
|---|---|---|---|---|
| Naive guess | — | — | ~0.71°C | — |
| Random Forest | classical ML | flat features | ~0.33°C | -53% |
| **FFNN** (headline) | neural net | flat features | ~0.33°C | -54% |
| ViT | neural net | real 5×5 satellite-grid patch → attention → embedding | ~0.34°C | -52% |
| CNN | neural net | real 5×5 satellite-grid patch → conv → pooled embedding | ~0.35°C | -51% |
| LSTM | neural net | depth-sequence decoder | ~0.36-0.40°C | -46-49% |
| Autoencoder | neural net | unsupervised embedding + small supervised probe | ~0.41°C | -42% |

(Exact CNN/ViT/LSTM figures vary slightly run to run — a known PyTorch/cuDNN GPU
non-determinism quirk in Conv2d/LSTM kernels, not a bug; the ordering is stable.)

FFNN wins and is the model shown in the profile explorer / scatter panels; it beat
Random Forest once the synthetic profile count was raised from 220 to 600 (not
enough data was the reason the FFNN initially lost — verified empirically). The
CNN and ViT are the closest things in this repo to the "satellite embeddings" the
problem statement asks for: both pool a real spatial patch of the surface grid
into a compact latent vector before predicting, rather than using flattened point
features directly — ViT edges out CNN here, the one case where attention beat
convolution on identical input. The Autoencoder tests a different idea: an
embedding trained *without ever seeing the depth targets* (pure unsupervised
reconstruction), then a small supervised head on top — it came in last, which is
the textbook expected result at this data scale (self-supervised pretraining's
usual edge is generalizing to unseen data, which isn't what's being tested here).
See `PROJECT_REPORT.txt` §4 for full architecture detail on all five neural nets
and why each lands where it does on this synthetic data.

We also tried to replicate the *actual* headline method from Loo et al. 2026
(arXiv:2605.00860) — cluster by depth-band and time-phase, then train a separate
small network per cluster, instead of one pooled network. On our synthetic data
this made results *worse*, not better (see `dl_pipeline.py`'s module docstring and
`PROJECT_REPORT.txt` for the full writeup and why: too little data per cluster,
and our synthetic generator uses one smooth formula for the whole depth profile,
so there's no genuine per-depth heterogeneity for the clustering to exploit). This
is reported honestly rather than hidden — it's a real, useful finding about when
that method pays off, not a bug we're pretending isn't there.

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
- **Adaptive spatiotemporal clustering**: `spatiotemporal_clusters()` groups observations by (lat, lon, day, SSH) before regression — a lightweight KMeans stand-in for the clustering framework in Loo et al., *"An Adaptive Spatiotemporal Clustering Framework for 3D Ocean Subsurface Temperature Reconstruction"* (arXiv:2605.00860, 2026), which groups profiles sharing thermocline structure to improve reconstruction. Cluster membership is fed to the model as a feature and can be toggled as a map overlay in the dashboard. Features are z-scored and `day` is downweighted (0.3x) before clustering — at full/raw weight, `day` being constant within a single map frame dominated cluster assignment and produced degenerate, unbalanced clusters (e.g. only 3 of 6 ever appearing on some days).
- **Satellite context**: the dashboard embeds a live NASA Worldview view scoped to the study region (loads client-side; needs internet in the viewer's browser, not this repo).

## Repo layout

```
ocean_pipeline_demo.py   # simulated data layer, heatwave detection, clustering, RF baseline, training/eval
dl_pipeline.py           # all 5 neural nets (FFNN headline, CNN, ViT, Autoencoder, LSTM) + the depth/time clustering experiment
export_data.py           # bakes simulated pipeline output (RF + FFNN) into public/data.json
real_data.py             # loaders for real MOSDAC (.h5) + CMEMS (.nc) files
export_real_data.py      # bakes real data into public/data_real.json
streamlit_app.py         # Streamlit UI (full interactive app) -- NOTE: still RF-only, not yet updated to the FFNN
public/                  # static dashboard (index.html / style.css / app.js / data*.json) — deploy target for Vercel
vercel.json              # points Vercel at public/
.streamlit/config.toml   # Sea Green theme for the Streamlit app
requirements.txt
MOSDAC/, cmems_*.nc       # real source files (gitignored — large, regenerate data_real.json locally)
```

Team **Sea Green**.
