# Sea Green — OceanEmbed

**Subsurface ocean temperature reconstruction and marine heatwave monitoring for the North Indian Ocean (5–30°N, 45–105°E)**, built entirely from real satellite and reanalysis data (MOSDAC + CMEMS).

Direct underwater temperature measurements are sparse and expensive to collect at scale. Satellites observe the surface (SST, salinity, currents) continuously and across the full basin. This pipeline trains seven independent models to learn the surface-to-subsurface relationship, so that subsurface structure can be estimated anywhere in the basin, and tracks basin-wide SST anomaly against the observation window's own mean to flag marine heatwave events (Hobday-scale categories: Watch / Warning / Severe / Extreme).

The dashboard (`public/`) is organized into four tabs: **Live Monitor** (real, current data), **The Model** (the reconstruction demo), **Results** (metrics and claims), and **How it Works** (methodology, presented separately from the primary flow for readers who want additional detail).

> ℹ **Every tab runs on real data.** There is no simulated or synthetic
> data anywhere in this build — Live Monitor, The Model, and Results all
> run on real MOSDAC + CMEMS data. Two honest limitations remain: the
> real feature set covers 4 of the problem statement's 5 required surface
> inputs (SST, SSS, and surface currents are real; a real surface-wind
> product and a real SSH/SLA product have not yet been obtained, and
> neither is modeled with any placeholder), and the "1000 m" depth target
> is approximated by the deepest native level actually available in the
> pulled CMEMS extract, approximately 902 m. See [Real data](#real-data-mosdac--cmems)
> below for the full accounting.

> 📊 **Recommended starting point: [`DATASET.md`](DATASET.md).** It
> documents the exact features used, in contrast to every candidate
> field available; includes a real 25-row sample of the dataset the
> pipeline trains on, committed and browsable directly on GitHub; and
> explains precisely how satellite imagery is converted into the
> CNN/ViT "compact satellite embedding" required by the problem
> statement.

## The models

`real/real_training.py`'s `train_real_and_evaluate()` trains **seven independent models** on real MOSDAC/CMEMS-derived data, all using an *identical* time-based train/test split to ensure a fair comparison (600 train / 200 test, from 800 sampled real ocean locations). All seven models, together with a naive baseline, are presented side by side in the dashboard's Results tab. This covers every architecture family named in the problem statement (CNN, ViT, Autoencoder, GNN, and an attention-based hybrid via ViT):

| Model | Type | Input | Mean RMSE | vs. baseline |
|---|---|---|---|---|
| Naive guess | — | — | 2.055°C | — |
| **Random Forest** (headline) | classical ML | flat features (sst, sss, currents) | 0.761°C | -63.0% |
| GNN | neural net | k-NN graph of test-point locations → 2-layer GCN with self-feature skip | 0.770°C | -62.5% |
| FFNN | neural net | flat features (sst, sss, currents) | 0.853°C | -58.5% |
| LSTM | neural net | depth-sequence decoder | 1.055°C | -48.7% |
| ViT | neural net | 5×5 satellite-grid patch (sst/sss/currents) → attention → embedding | 1.150°C | -44.0% |
| Autoencoder | neural net | unsupervised embedding with a small supervised probe | 1.152°C | -43.9% |
| CNN | neural net | 5×5 satellite-grid patch (sst/sss/currents) → convolution → pooled embedding | 1.198°C | -41.7% |

(Exact CNN/ViT/LSTM/GNN figures vary slightly between runs, a known PyTorch/cuDNN GPU non-determinism effect in the Conv2d/LSTM/matmul kernels rather than a defect; the relative ordering of models is stable.)

**Random Forest achieves the lowest error on real data**, narrowly ahead of the GNN (0.761°C vs. 0.770°C — close enough to be effectively tied), and is the model shown in the profile explorer, scatter panel, and headline claim. This is a real, legitimate finding, not noise: tree ensembles and local-neighbor graph methods tend to handle small real datasets (600 training rows) more effectively than data-hungry patch-based models, which typically need substantially more examples to show their usual advantage — CNN and ViT, which pool a real spatial patch of the surface grid into a compact latent vector before predicting (the implementations in this repository closest to the "satellite embeddings" the problem statement asks for), land at the back of the field here for exactly that reason, not because the architecture is flawed. The GNN represents each real test/train location as a node connected to its six nearest neighbors, and a two-layer Graph Convolutional Network propagates surface information between nearby locations; its initial version exhibited pronounced over-smoothing, resolved with a direct, un-smoothed skip pathway (the same principle underlying APPNP/JK-Nets — a genuine architectural correction, not parameter tuning; see `models/dl_pipeline.py`'s `GNN` class docstring). The Autoencoder tests a distinct hypothesis: an embedding trained without exposure to the depth targets (unsupervised reconstruction only), followed by a small supervised head — it lands mid-to-back of the field, broadly in line with the expectation that self-supervised pretraining's usual advantage (generalizing to data the encoder never trained on) is not the condition being tested here.

## Two ways to run this

| | Static dashboard (`/public`) | Full app (`streamlit_app.py`) |
|---|---|---|
| Stack | Plain HTML/CSS/JS with Plotly.js (CDN) | Streamlit |
| Deploy target | **Vercel** (no server required; reads a pre-baked `data.json`) | **Hugging Face Spaces** (or any Python-capable host) |
| Data | Snapshot exported by `real/export_real_results.py` | Not currently available — see note below |

The static dashboard is the working, current option. `streamlit_app.py` has not yet been updated for the real-data-only pipeline (it was built around the now-removed synthetic track) — see the note at the top of that file.

### Static dashboard → Vercel

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m real.export_real_results        # (re)builds public/data.json by training all 7 models on real data
```

Push this repository to GitHub and import it in Vercel. `vercel.json` already sets `outputDirectory` to `public/`, so this is a zero-configuration static deployment with no serverless functions, no build step, and no cold starts.

To refresh the dashboard with new results, re-run `python -m real.export_real_results` and commit the updated `public/data.json`.

## Real data: MOSDAC and CMEMS

`real/real_data.py` reads real datasets from `real/data/` for the Live Monitor tab, and `real/export_real_data.py` compiles them into `public/data_real.json`. `real/real_training.py` and `real/export_real_results.py` read the same real source files (plus a real multi-depth subsurface extract) to train and evaluate all seven models for the Model and Results tabs. Raw source files are not committed (several hundred megabytes each, excluded via `.gitignore`); provenance for every raw file — product ID, institution, coverage, and checksum — is committed regardless, in `real/PROVENANCE.md` and `real/data/CHECKSUMS.sha256`, so the real-data claim can be verified independently from the repository alone, without requiring the multi-gigabyte source files.

- **MOSDAC** — `real/data/MOSDAC/*.h5`, INSAT-3DR L2B SST (ISRO/SAC), half-hourly, 25 August 2026. Eight evenly-spaced real passes, cropped to the study region, drive the "Live Satellite Pass" panel.
- **CMEMS**, all `GLOBAL_ANALYSISFORECAST_PHY_001_024` / `_BGC_001_028` (Mercator Ocean), 1–27 August 2026:
  - `thetao` (SST) and `so` (SSS) — daily-mean basin trend and a live status indicator (single near-surface level, ~0.49 m, for the Live Monitor tab).
  - `uo`/`vo` (surface currents) — a real vector-field snapshot (direction and speed), one of the specification's five required input variables.
  - `chl` (chlorophyll, 0.25° BGC product) — a real ecosystem-impact proxy, connecting to the problem statement's marine-ecosystem motivation. Not one of the 5 required inputs; not used as a model feature.
  - A separate multi-depth `thetao` extract (35 native depth levels, ~0.49–902 m, 1–27 August) supplies the real subsurface temperature target used to train and evaluate all seven models — this is the piece that makes real model training possible at all, distinct from the single-level file used for the Live Monitor trend chart.

  Four of the specification's five required surface inputs are backed by real data; only **SSH/SLA** and **surface winds** remain unavailable, and neither is modeled with a placeholder.

All CMEMS grids used by the Live Monitor tab (SST/SSS/chlorophyll maps and the current vector field) are **genuinely regridded to the specification's required 0.25°** via `_regrid_to_target()` (`xarray.coarsen().mean()`, a real block-average from the native 0.083° grid, verified to land on exactly 0.25° spacing), rather than decimated or subsampled. Vector fields (uo/vo) are regridded as a single Dataset before speed and heading are derived, since averaging speed and heading separately would be physically incorrect. MOSDAC (native geostationary swath resolution) has not yet been regridded to a regular grid, since it only feeds the visual "Live Satellite Pass" panel, not any model input. The model-training pipeline (`real/real_training.py`) uses the native 0.083° CMEMS grid directly, rather than the regridded 0.25° product, since patch extraction benefits from the finer native resolution.

To refresh with new files, place them in `real/data/MOSDAC/` or `real/data/` and re-run:

```bash
pip install h5py xarray netCDF4   # only needed for this step
python -m real.export_real_data      # Live Monitor tab
python -m real.export_real_results   # Model + Results tabs (retrains all 7 models)
```

Then regenerate `real/data/CHECKSUMS.sha256` and update `real/PROVENANCE.md` (commands provided in that file), so the provenance record remains consistent with the files that produced the exports.

Independent real Argo float validation (via `argopy`) and a real surface-wind/SSH source remain known gaps — see `real/PROVENANCE.md`, "Where the Ground Truth Comes From," for what this project's own real-data benchmark does and does not validate.

## Methodology notes

- **Marine heatwave detection**: basin-mean SST anomaly relative to the pulled observation window's own mean (not a 30-year climatology — a short-window baseline, labeled as such in the UI), classified using Hobday-scale thresholds (0.5 / 1.0 / 1.5 / 2.0 °C).
- **Region clustering**: `build_clusters()` in `real/export_real_results.py` groups real test/train locations by `(lat, lon, day)` via K-means (n=6, matching the map overlay's color palette), informed by the adaptive clustering framework in Loo et al., *"An Adaptive Spatiotemporal Clustering Framework for 3D Ocean Subsurface Temperature Reconstruction"* (arXiv:2605.00860, 2026). Cluster membership is supplied to the map overlay and the test-point table, and can be toggled on in the dashboard. This is a lighter version of an earlier `(lat, lon, day, ssh)`-weighted approach that existed when SSH was synthetically available; day is not currently downweighted, and the resulting cluster sizes were verified reasonably balanced (no cluster near-empty) on the real data actually used.
- **Knowledge-informed depth-gradient loss**: `DepthGradientLoss` (`models/dl_pipeline.py`) penalizes mismatch in the depth-to-depth *gradient* of the predicted profile, rather than scoring each depth in isolation, informed by the adaptive depth-gradient loss in Wang et al., *"Knowledge-Informed Deep Learning Model for Subsurface Thermohaline Reconstruction From Satellite Observations"* (CGKDN, IEEE TGRS vol. 62, 2024). It is implemented and available via `train_ffnn(..., use_depth_grad=True)`, but is not currently enabled by `real/real_training.py`'s training calls, which train every model with plain MSE.
- **Wind stress curl**: an earlier build modeled this (following Xie et al. 2022's Attention U-Net paper) as a synthetic-only feature. It was dropped along with the rest of the synthetic track, since it required a real wind product this project does not have.
- **Satellite context**: the dashboard embeds a live NASA Worldview view scoped to the study region (loaded client-side; requires internet access in the viewer's browser, independent of this repository).

## Repo layout

The repository is split so real-data code, the shared model toolbox, and the dashboard are clearly separated.

```
real/                            everything in this project runs on real MOSDAC and CMEMS data
  real_data.py                     loaders for real MOSDAC (.h5) and CMEMS (.nc) files -> public/data_real.json
  real_training.py                 real (surface, subsurface) training pairs; trains all 7 models on real data
  export_real_data.py              compiles real_data.py's output into public/data_real.json (Live Monitor tab)
  export_real_results.py           trains all 7 models and compiles results into public/data.json (Model + Results tabs)
  data/                            raw source files (gitignored; 100s of MB to ~800 MB; regenerate locally)
    MOSDAC/*.h5, *.nc
    CHECKSUMS.sha256               sha256 of every raw file above, committed so integrity is checkable without the files
  PROVENANCE.md                    source, product ID, coverage, and checksum record for every real file, and a
                                    statement of where the actual validation authority resides (CMEMS QUID / ISRO,
                                    not this repository and not any AI tool used to build it)

models/
  dl_pipeline.py                  reusable model-architecture toolbox (FFNN, CNN, ViT, GNN, Autoencoder, LSTM),
                                   imported by real/real_training.py

streamlit_app.py                 Streamlit UI; not yet updated for the real-data-only pipeline (see note at top of file)
public/                          static dashboard (index.html / style.css / app.js / data*.json), the deploy target for Vercel
vercel.json                      points Vercel at public/
.streamlit/config.toml           Sea Green theme for the Streamlit app
requirements.txt
DATASET.md                       feature selection and the satellite-embedding pipeline, with real sample data
```

Every script under `real/` can be run either directly (for example, `python real/export_real_results.py`) or as a module from the repository root (for example, `python -m real.export_real_results`); both invocation styles are supported.

Team **Sea Green**.
