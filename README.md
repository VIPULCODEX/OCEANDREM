# Sea Green — OceanEmbed

**Subsurface ocean temperature reconstruction and marine heatwave monitoring for the North Indian Ocean (5–30°N, 45–105°E)**, built from sparse Argo float profiles and dense surface satellite fields.

Argo floats measure subsurface temperature directly but are sparse in space and time. Satellites observe the surface (SST, SSH, salinity, wind) continuously and across the full basin. This pipeline trains a neural network (FFNN) to learn the surface-to-subsurface relationship, so that subsurface structure can be estimated anywhere in the basin, and tracks basin-wide SST anomaly against climatology to flag marine heatwave events (Hobday-scale categories: Watch / Warning / Severe / Extreme).

Surface wind is represented as true 2D u/v components together with a derived **wind stress curl** field — the specific additional input identified by Xie et al. (2022, "Attention U-Net" for South China Sea subsurface reconstruction, IEEE TGRS) as the single largest accuracy contributor below approximately 50 m depth, since it drives Ekman pumping, which in turn affects thermocline depth. This was verified on the project's own data: adding curl produced a measurable improvement for the spatially-aware models (CNN, ViT) and the sequential model (LSTM), while leaving the flat-feature FFNN largely unchanged, consistent with the source paper's finding that this input benefits only models capable of using spatial structure. An optional "adaptive depth-gradient" loss term (`DepthGradientLoss` in `models/dl_pipeline.py`, enabled via `use_depth_grad=True`) is also implemented, informed by the knowledge-informed CGKDN model of Wang et al. (2024). This term was tested and found not to improve performance on the current synthetic data; it is disabled by default, and the result is documented in `PROJECT_REPORT.txt`, Sections 1b and 2.

The dashboard (`public/`) is organized into four tabs: **Live Monitor** (real, current data), **The Model** (the reconstruction demo), **Results** (metrics and claims), and **How it Works** (methodology, presented separately from the primary flow for readers who want additional detail).

> ℹ **Real and simulated data are mixed in this build; the distinction
> is stated explicitly here to avoid ambiguity.** The **Live Monitor** tab
> is driven by real **MOSDAC (INSAT-3DR) satellite SST** and real
> **CMEMS** SST/SSS/currents/chlorophyll data (current as of August 2026)
> — see [Real data](#real-data-mosdac--cmems) below. The **Model** and
> **Results** tabs (Argo floats, subsurface reconstruction, RMSE) run on
> a physically-motivated simulation with an injected heatwave event,
> since a real multi-depth Argo/subsurface dataset was not available at
> the time of this build. The integration points for real data on that
> side are already implemented in `synthetic/ocean_pipeline_demo.py`.

> 📊 **Recommended starting point: [`DATASET.md`](DATASET.md).** It
> documents the exact features used, in contrast to every candidate
> field available; includes a real 25-row sample of the dataset each
> track trains on, committed and browsable directly on GitHub; and
> explains precisely how satellite imagery is converted into the
> CNN/ViT "compact satellite embedding" required by the problem
> statement. For the methodology behind that schema — how the data was
> acquired, how it was preprocessed, and why specific features were
> selected and compacted — see
> [`TECHNICAL_APPROACH.md`](TECHNICAL_APPROACH.md).

## The models

`models/dl_pipeline.py` trains **seven independent models** on the synthetic Argo dataset, all using an *identical* time-based train/test split (`time_based_split()` in `synthetic/ocean_pipeline_demo.py`) to ensure a fair comparison. All seven models, together with a naive baseline, are presented side by side in the dashboard's Results tab. This covers every architecture family named in the problem statement (CNN, ViT, Autoencoder, GNN, and an attention-based hybrid via ViT):

| Model | Type | Input | Mean RMSE | vs. baseline |
|---|---|---|---|---|
| Naive guess | — | — | ~0.71°C | — |
| Random Forest | classical ML | flat features (including u/v wind and curl) | ~0.34°C | -52% |
| **FFNN** (headline) | neural net | flat features (including u/v wind and curl) | ~0.33°C | -54% |
| ViT | neural net | 5×5 satellite-grid patch (sst/ssh/sss/curl) → attention → embedding | ~0.33°C | -53% |
| GNN | neural net | k-NN graph of profile locations → 2-layer GCN with self-feature skip | ~0.34°C | -53% |
| CNN | neural net | 5×5 satellite-grid patch (sst/ssh/sss/curl) → convolution → pooled embedding | ~0.34°C | -52% |
| LSTM | neural net | depth-sequence decoder | ~0.36°C | -49% |
| Autoencoder | neural net | unsupervised embedding with a small supervised probe | ~0.44°C | -38% |

(Exact CNN/ViT/LSTM/GNN figures vary slightly between runs, a known PyTorch/cuDNN GPU non-determinism effect in the Conv2d/LSTM/matmul kernels rather than a defect; the relative ordering of models is stable.)

The FFNN achieves the lowest error and is the model shown in the profile explorer and scatter panels. It surpassed Random Forest once the synthetic profile count was increased from 220 to 600; the earlier loss was attributable to insufficient training data, confirmed empirically. The CNN and ViT are the closest implementations in this repository to the "satellite embeddings" specified in the problem statement: both pool a real spatial patch of the surface grid into a compact latent vector prior to prediction, rather than operating on flattened point features directly. ViT outperforms CNN here, the one case in this project where attention outperformed convolution on identical input. The GNN represents the ocean as a graph, with each Argo profile as a node connected to its six nearest neighbors by location, and a two-layer Graph Convolutional Network propagating surface information between nearby profiles. Its initial version exhibited pronounced over-smoothing (0.526 RMSE, worse than every other model); this was diagnosed as classic GCN self-feature dilution and resolved with a direct, un-smoothed skip pathway, the same principle underlying APPNP/JK-Nets — a genuine architectural correction rather than parameter tuning. The Autoencoder tests a distinct hypothesis: an embedding trained without exposure to the depth targets (unsupervised reconstruction only), followed by a small supervised head. It produced the highest error, the expected result at this data scale, since self-supervised pretraining's typical advantage is generalization to unseen data, which is not the condition being tested here. Full architecture-level detail for all six neural networks, and the reasoning behind each result, is provided in `PROJECT_REPORT.txt`, Section 2, item 4.

A further experiment attempted to replicate the primary method of Loo et al. (2026, arXiv:2605.00860): clustering by depth band and time phase, then training a separate small network per cluster rather than one pooled network. On this project's synthetic data, the result was a *degradation* in performance rather than an improvement; the module docstring in `models/dl_pipeline.py` and `PROJECT_REPORT.txt` provide the full analysis, attributing this to insufficient data per cluster and to the synthetic generator's use of a single smooth formula for the entire depth profile, which leaves no genuine per-depth heterogeneity for the clustering to exploit. This negative result is reported directly rather than omitted, as it is a legitimate finding regarding the conditions under which the method is expected to help.

## Two ways to run this

| | Static dashboard (`/public`) | Full app (`streamlit_app.py`) |
|---|---|---|
| Stack | Plain HTML/CSS/JS with Plotly.js (CDN) | Streamlit |
| Deploy target | **Vercel** (no server required; reads a pre-baked `data.json`) | **Hugging Face Spaces** (or any Python-capable host) — recommended if the dashboard must run the model live and interactively, or if Vercel's function memory limits become constraining |
| Data | Snapshot exported by `synthetic/export_data.py` | Computed live on each run (cached per session) |

Both read from the same source of truth: `synthetic/ocean_pipeline_demo.py`.

### Static dashboard → Vercel

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m synthetic.export_data        # (re)builds public/data.json from the pipeline
```

Push this repository to GitHub and import it in Vercel. `vercel.json` already sets `outputDirectory` to `public/`, so this is a zero-configuration static deployment with no serverless functions, no build step, and no cold starts.

To refresh the dashboard with new results, re-run `python -m synthetic.export_data` and commit the updated `public/data.json`.

### Full interactive app → Hugging Face Spaces (or local)

```bash
streamlit run streamlit_app.py
```

For a Space: select the **Streamlit** SDK and point it at this repository; it will pick up `requirements.txt` and `.streamlit/config.toml` automatically.

## Real data: MOSDAC and CMEMS

`real/real_data.py` reads two real datasets from `real/data/`, and `real/export_real_data.py` compiles them into `public/data_real.json` (committed to the repository; the raw source files are not, as they are several hundred megabytes each and are excluded via `.gitignore`). Provenance for every raw file — product ID, institution, coverage, and checksum — is committed regardless, in `real/PROVENANCE.md` and `real/data/CHECKSUMS.sha256`, so the real-data claim can be verified independently from the repository alone, without requiring the multi-gigabyte source files.

- **MOSDAC** — `real/data/MOSDAC/*.h5`, INSAT-3DR L2B SST (ISRO/SAC), half-hourly, 25 August 2026. Eight evenly-spaced real passes, cropped to the study region, drive the "Live Satellite Pass" panel.
- **CMEMS**, all `GLOBAL_ANALYSISFORECAST_PHY_001_024` / `_BGC_001_028` (Mercator Ocean), 1–27 August 2026, single near-surface level (approximately 0.49 m):
  - `thetao` (SST) and `so` (SSS) — daily-mean basin trend and a live status indicator.
  - `uo`/`vo` (surface currents) — a real vector-field snapshot (direction and speed), one of the specification's five required input variables.
  - `chl` (chlorophyll, 0.25° BGC product) — a real ecosystem-impact proxy, connecting to the problem statement's marine-ecosystem motivation.

  Four of the specification's five required surface inputs are now backed by real data; only **SSH/SLA** and **surface winds** remain unavailable.

All CMEMS grids (SST/SSS/chlorophyll maps and the current vector field) are **genuinely regridded to the specification's required 0.25°** via `_regrid_to_target()` (`xarray.coarsen().mean()`, a real block-average from the native 0.083° grid, verified to land on exactly 0.25° spacing), rather than decimated or subsampled as in earlier builds. Vector fields (uo/vo) are regridded as a single Dataset before speed and heading are derived, since averaging speed and heading separately would be physically incorrect. MOSDAC (native geostationary swath resolution) and the synthetic pipeline's map grid (currently 1.0°, a deliberate size/speed tradeoff) have not yet been regridded; see `PROJECT_REPORT.txt`, Section 2, item 2, for the complete account.

To refresh with new files, place them in `real/data/MOSDAC/` or `real/data/` and re-run:

```bash
pip install h5py xarray netCDF4   # only needed for this step
python -m real.export_real_data
```

Then regenerate `real/data/CHECKSUMS.sha256` and update `real/PROVENANCE.md` (commands provided in that file), so the provenance record remains consistent with the files that produced `data_real.json`.

This dataset contains only one depth level and therefore cannot support multi-depth subsurface reconstruction; this is why the ML and heatwave-detection demonstration continues to run on the simulation described above. A real subsurface signal would require either a multi-depth CMEMS `thetao` pull (via `copernicusmarine`) or real Argo profiles (via `argopy`); both require supporting infrastructure — an account and a working build toolchain for `argopy`'s dependencies — that was not available during this development period.

## Going live with real data (subsurface reconstruction)

Two integration points exist in `synthetic/ocean_pipeline_demo.py`, both currently gated behind `USE_SYNTHETIC_DATA = True`:

- **`get_argo_data()`** — real Argo profiles via [`argopy`](https://argopy.readthedocs.io/): `DataFetcher().region([lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, start, end])`.
- **`get_satellite_grid()`** — real surface fields. For the Indian Ocean specifically, substitute **MOSDAC / ISRO** INSAT-3D/3DR SST and OSCAT wind products, or Copernicus Marine as a global fallback.

Setting the flag and implementing the two commented API calls, then re-running `python -m synthetic.export_data` (or running `streamlit_app.py` directly), leaves everything downstream — training, metrics, heatwave detection, clustering, and all charts — unchanged.

(Separately, `real/real_training.py` already trains all seven models on genuine real surface and real subsurface data; see `PROJECT_REPORT.txt`, Section 1d. Integrating its output into the dashboard, in place of `synthetic/export_data.py`'s synthetic-trained models, is the next planned step, tracked in Section 1c, part J.)

## Methodology notes

- **Marine heatwave detection**: basin-mean SST anomaly relative to a climatology computed across the observation window, classified using Hobday-scale thresholds (0.5 / 1.0 / 1.5 / 2.0 °C).
- **Adaptive spatiotemporal clustering**: `spatiotemporal_clusters()` groups observations by (lat, lon, day, SSH) prior to regression — a lightweight K-means analogue of the clustering framework in Loo et al., *"An Adaptive Spatiotemporal Clustering Framework for 3D Ocean Subsurface Temperature Reconstruction"* (arXiv:2605.00860, 2026), which groups profiles sharing thermocline structure to improve reconstruction. Cluster membership is supplied to the model as a feature and can be displayed as a map overlay in the dashboard. Features are z-scored and `day` is downweighted by a factor of 0.3 before clustering; at full weight, `day` — being constant within a single map frame — dominated cluster assignment and produced degenerate, unbalanced clusters (for example, only three of six clusters appearing on some days).
- **Wind stress curl**: `compute_wind_stress_curl()` derives an Ekman-pumping-relevant curl field from the synthetic 2D wind field, following Xie et al., *"Reconstruction of Subsurface Temperature Field in the South China Sea From Satellite Observations Based on an Attention U-Net Model"* (IEEE TGRS vol. 60, 2022), identified there as the single largest accuracy contributor below approximately 50 m. It feeds every model (`FEATURE_COLS`) and the CNN/ViT patches (`PATCH_CHANNELS`).
- **Knowledge-informed depth-gradient loss**: `DepthGradientLoss` (optional, enabled via `use_depth_grad=True`) penalizes mismatch in the depth-to-depth *gradient* of the predicted profile, rather than scoring each depth in isolation, informed by the adaptive depth-gradient loss in Wang et al., *"Knowledge-Informed Deep Learning Model for Subsurface Thermohaline Reconstruction From Satellite Observations"* (CGKDN, IEEE TGRS vol. 62, 2024). It is disabled by default: testing found it slightly degraded RMSE on the current single-formula synthetic profiles, which lack genuine vertical heterogeneity for the term to exploit; it is expected to be beneficial on real, heterogeneous ocean data.
- **Satellite context**: the dashboard embeds a live NASA Worldview view scoped to the study region (loaded client-side; requires internet access in the viewer's browser, independent of this repository).

## Repo layout

The repository is deliberately split into `synthetic/` and `real/` directories, so that which code path produces which numbers is unambiguous from the directory structure itself, rather than relying solely on documentation. `models/` is shared by both tracks (identical architecture code, with different data supplied to it).

```
synthetic/                       # [SIMULATED] everything trained/evaluated on the physically-motivated simulation
  ocean_pipeline_demo.py           simulated data layer, heatwave detection, clustering, RF baseline, training/eval
  export_data.py                   compiles simulated pipeline output (RF + FFNN + 5 more) into public/data.json
  synthetic_argo_dataset.csv       historical artifact, not read by current code (see PROJECT_REPORT.txt, Section 2)
  ocean_pipeline_demo.png          historical artifact, not read by current code

real/                            # [REAL] everything trained/evaluated on actual MOSDAC and CMEMS data
  real_data.py                     loaders for real MOSDAC (.h5) and CMEMS (.nc) files -> public/data_real.json
  real_training.py                 real (surface, subsurface) training pairs; trains all 7 models on real data
  export_real_data.py              compiles real_data.py's output into public/data_real.json
  data/                            raw source files (gitignored; 100s of MB to ~800 MB; regenerate locally)
    MOSDAC/*.h5, *.nc
    CHECKSUMS.sha256               sha256 of every raw file above, committed so integrity is checkable without the files
  PROVENANCE.md                    source, product ID, coverage, and checksum record for every real file, and a
                                    statement of where the actual validation authority resides (CMEMS QUID / ISRO,
                                    not this repository and not any AI tool used to build it)

models/
  dl_pipeline.py                  all 6 neural nets (FFNN headline, CNN, ViT, GNN, Autoencoder, LSTM) and the
                                   depth/time clustering experiment — shared architecture code, imported by
                                   both synthetic/export_data.py and real/real_training.py

streamlit_app.py                 Streamlit UI (full interactive app); note: still Random-Forest-only, not yet updated to the FFNN
public/                          static dashboard (index.html / style.css / app.js / data*.json), the deploy target for Vercel
vercel.json                      points Vercel at public/
.streamlit/config.toml           Sea Green theme for the Streamlit app
requirements.txt
DATASET.md                       feature selection and the satellite-embedding pipeline, with real sample data
TECHNICAL_APPROACH.md            methodology: data acquisition, preprocessing, and feature engineering
```

Every script under `synthetic/` or `real/` can be run either directly (for example, `python synthetic/export_data.py`) or as a module from the repository root (for example, `python -m synthetic.export_data`); both invocation styles are supported.

Team **Sea Green**.
