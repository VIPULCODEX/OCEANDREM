# Technical Approach: Data Acquisition, Preprocessing, and Feature Engineering

This document describes, in narrative form, how the data used by this
project was obtained, how it was transformed into a form suitable for
model training, and the reasoning behind which features were selected
and how they were compacted into model inputs. It is intended as a
companion to [`DATASET.md`](DATASET.md), which documents the resulting
feature schema and the satellite-embedding architecture in tabular and
diagrammatic form. This document instead explains the methodology and
reasoning that produced that schema — the *how* and *why*, not only the
*what*.

Two independent data tracks exist in this project, referred to
throughout as the **real track** and the **synthetic track**. Both are
described below, since the acquisition and preprocessing methodology
differs substantially between them.

## 1. Data Acquisition

### 1.1 Real-track acquisition

Real data was obtained from two independent sources, both official
providers rather than aggregators, so that provenance could be traced to
an authoritative institution:

- **Copernicus Marine Service (CMEMS)**, via the official
  `copernicusmarine` command-line toolbox, authenticated with a
  registered account. Surface fields (`thetao`/SST, `so`/SSS,
  `uo`/`vo`/currents, `chl`/`phyc`/chlorophyll) were obtained first, at
  the single near-surface level available in the standard product. A
  genuine subsurface target was obtained subsequently, using
  `copernicusmarine subset` with an explicit depth range request
  (0–1000 m) rather than the default surface-only extract. This
  produced a file with 35 real depth levels (approximately 0.49 m to
  902 m), daily, over the exact study region (5–30°N, 45–105°E),
  1–27 August. This file is the only genuine subsurface *measurement*
  source used anywhere in this project; every other subsurface value
  used for training is either a held-out portion of this same file or a
  synthetic estimate.
- **MOSDAC (ISRO / Space Applications Centre)**, via direct download of
  INSAT-3DR L2B SST granules (HDF5 format) for a single day, half-hourly,
  yielding 17 real satellite passes over the study region.

The exact product identifiers, institutions, coverage windows, and
per-file checksums for every real file are recorded in
[`real/PROVENANCE.md`](real/PROVENANCE.md); that document also states
explicitly where the accuracy/validation authority for this data
actually resides (the CMEMS Quality Information Documents and ISRO's own
calibration reports), which is outside the scope of this project.

**What real acquisition could not yet provide.** No real surface wind
product (for example, ASCAT or OSCAT) and no real sea-surface-height or
sea-level-anomaly (SSH/SLA) product were obtained, due to time and
account-access constraints. A real, independent Argo float pull (via the
`argopy` library) was also attempted but not completed, due to a build
failure in one of its dependencies in the available environment (no C
compiler present for a required package). These gaps are the direct
reason a second, synthetic data track exists: it is not a substitute for
real data acquisition, but a way to complete the full input specification
so that the modeling and evaluation pipeline could be built, run, and
validated end to end while real-data acquisition for the missing inputs
continues.

### 1.2 Synthetic-track generation

Where a real source was not available, values were not left blank or
filled with a placeholder constant; they were generated from explicit,
physically-motivated formulas designed to reproduce known oceanographic
relationships, so that the resulting dataset would contain a genuine,
learnable surface-to-subsurface relationship rather than noise. The
generation logic lives in `synthetic/ocean_pipeline_demo.py`
(`get_satellite_grid()` for the gridded surface fields,
`get_argo_data()` for the sparse profile locations) and follows this
reasoning:

- **Sea surface temperature** is modeled as warmer near the equator,
  cooling gradually poleward, with a smooth seasonal oscillation and an
  injected multi-week warm pulse (`heatwave_bump()`) that stands in for
  a genuine marine heatwave event, so that the heatwave-detection logic
  has an actual event to detect during development and demonstration.
- **Sea surface height (SSH) anomaly** is modeled as a small number of
  smooth Gaussian "blobs," representing mesoscale eddies, since eddy-driven
  SSH anomalies are the physical signal expected to correlate with
  subsurface heat content.
- **Surface wind** is generated as true eastward/northward (u, v)
  components (not a single scalar speed), following a smooth monsoon-like
  regime with superimposed rotational structure, from which wind stress
  curl is derived analytically (see Section 2.3 below).
- **The critical physical link — the one the models are meant to learn —
  is between SSH anomaly and subsurface temperature.** Each synthetic
  Argo profile's thermocline depth scale is set as an increasing function
  of positive SSH anomaly (a warm eddy implies a deeper warm layer) and a
  decreasing function of negative SSH anomaly, and the full 15-depth
  temperature profile is then generated from an exponential
  thermocline-decay formula anchored to that scale and to the profile's
  own surface temperature. This is what makes the synthetic dataset
  useful for validating the modeling pipeline: the subsurface target is
  not independent random noise, but a deterministic (plus small
  observational noise) function of the surface fields the models
  actually receive, mirroring the real relationship the project is
  ultimately trying to reconstruct.

Every synthetic field additionally has independent Gaussian noise added,
so that the resulting profiles are not perfectly recoverable from a
closed-form inverse of the generating formula; this keeps the
regression problem non-trivial. The generation is fully reproducible: a
fixed random seed (`RANDOM_SEED = 42`) is used throughout.

**This is a deliberate methodological choice, not an attempt to
represent the synthetic data as real.** Every dashboard view and every
document in this repository states explicitly which numbers come from
the real track and which come from the synthetic track; see the "Mixed
real and simulated" notice in `README.md` and the track-by-track
breakdown in `DATASET.md`, Section 1.

## 2. Preprocessing

Preprocessing differs by source, since each real product arrives in a
different native format and resolution, while the synthetic generator
produces already-aligned output directly. The steps below are applied
before any feature is exposed to a model.

### 2.1 Harmonization of real source files

`real/real_data.py` resolves each CMEMS file by the physical variable it
actually contains, rather than by filename, since files renamed by a
browser during download do not reliably follow a naming convention (for
example, `Sea Surface temp.nc` and a CMEMS-prefixed file were confirmed,
by content and checksum, to be the same download saved under two names;
see `real/PROVENANCE.md`). For MOSDAC's HDF5 granules, fill values are
masked and the stored scale/offset encoding is applied to recover
physical SST values, following the product's documented format.

### 2.2 Temporal resampling

All real time series are resampled to a consistent daily cadence: CMEMS
surface fields are read from their daily-mean product directly, while
higher-frequency sources (for example, the 6-hourly single-level CMEMS
extract, and MOSDAC's half-hourly passes) are aggregated to a daily
value before being used in any time-indexed comparison. This matches the
specification's requirement of daily temporal resolution.

### 2.3 Spatial regridding to 0.25°

The specification requires a uniform 0.25° × 0.25° spatial grid. Native
CMEMS resolution for the physical products used here is 0.083°
(approximately three times finer). Rather than subsampling or decimating
(discarding grid points), `_regrid_to_target()` in `real/real_data.py`
performs a genuine block-mean spatial average:

```
T_coarse(i, j) = (1 / (f_lat * f_lon)) *
                 sum_{a=0}^{f_lat-1} sum_{b=0}^{f_lon-1}
                 T_native(f_lat*i + a, f_lon*j + b)
```

where `f_lat = f_lon = round(0.25 / native_grid_step)`, equal to 3 for
the 0.083° CMEMS grid; the resulting output grid was verified to land
exactly on 0.25° spacing. This is implemented with
`xarray.coarsen(...).mean()`. Vector fields (surface current components
`uo`/`vo`) are regridded together as a single dataset *before* speed and
heading are derived from them, since averaging speed and heading
independently, after the fact, would not be physically equivalent to
averaging the underlying vector components first. The chlorophyll
product, already natively at 0.25°, is left unchanged by this step
(a no-op is triggered automatically when the native resolution already
meets the target). The synthetic pipeline's map-visualization grid
remains at a coarser 1.0° for size and rendering-speed reasons in the
current build; this is a known, stated limitation, not an oversight (see
`README.md`, "Repo layout," and the discussion of scope in
`real/PROVENANCE.md`).

### 2.4 Wind stress curl derivation

Wind stress curl is not a directly observed or downloaded quantity in
either track; it is derived from the u/v wind components via central
finite differences, with the east–west grid spacing weighted by the
cosine of latitude to account for meridian convergence:

```
curl = dv/dx - du/dy
dv/dx ≈ (v[:, j+1] - v[:, j-1]) / (2·dx),   dx = dlon · 111000 · cos(lat)
du/dy ≈ (u[i+1, :] - u[i-1, :]) / (2·dy),   dy = dlat · 111000
```

This field is the specific additional input identified by Xie et al.
(2022) as the single largest accuracy contributor below approximately
50 m depth, since it is directly related to Ekman pumping. It is treated
here as a derived preprocessing step rather than a raw input, since it
is computed identically from the same underlying wind field on both
tracks (synthetically generated wind for the synthetic track; not yet
available as a real product, so not present on the real track — see
`DATASET.md`, Section 1).

### 2.5 Standard-depth matching (real subsurface target only)

The specification requires reconstruction at 15 standard depths (0, 5,
10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m). The real
multi-depth CMEMS extract provides 35 native levels that do not align
exactly with these values, so each standard depth is matched to its
nearest available native level by absolute difference:

```
z_matched(z_std) = argmin_{z_native} | z_native - z_std |
```

The deepest standard level, 1000 m, has no native level within the
downloaded range and therefore resolves to the deepest level actually
present, approximately 902 m. This approximation is stated explicitly in
`real/real_training.py`'s depth-match output and in
`real/PROVENANCE.md`, rather than concealed.

### 2.6 Spatiotemporal clustering (engineered categorical feature)

`spatiotemporal_clusters()` groups observations into regions using a
weighted K-means procedure over `(lat, lon, day, ssh)`:

```
minimize_{C_1..C_k}  sum_k sum_{x in C_k} || w ⊙ (x - mu_k) ||^2
```

with feature weights `w = (1.0, 1.0, 0.3, 1.0)`. The `day` feature is
deliberately downweighted to 0.3, because at full weight it dominates
cluster assignment within a single map frame (where `day` is constant
across every point in that frame), producing degenerate, unbalanced
clusters; this was confirmed empirically before the downweighting was
introduced. The resulting cluster label is supplied to the tabular
models as a one-hot categorical feature (`cluster_0`…`cluster_5`), not
used by the satellite-patch models. This step is informed by the
adaptive clustering framework in Loo et al. (2026), though implemented
here as a lightweight K-means approximation of it rather than a full
reproduction.

### 2.7 Feature scaling, applied at model-training time

Every tabular model and every patch-based model scales its inputs and
targets using `sklearn.preprocessing.StandardScaler`, fit exclusively on
the training split and then applied to both the training and test
splits, so that no information about the test set's distribution leaks
into preprocessing. For the satellite-patch models (CNN, ViT), the
patches themselves are normalized per channel (zero mean, unit variance,
computed from the training patches only), and the `day` feature and the
regression targets are scaled the same way (see `_prepare_patch_inputs()`
in `models/dl_pipeline.py`). Predictions are inverse-transformed back to
physical temperature units before evaluation.

## 3. Feature Selection and Compaction

### 3.1 Selecting relevant features from all candidate fields

Not every field produced by the data layer is passed to every model.
The full candidate-versus-used mapping, field by field, is documented in
`DATASET.md`, Section 2; the reasoning behind the two specific decisions
that most affect the final feature set is summarized here:

- **Wind stress curl was chosen to replace scalar wind speed as the
  patch input**, rather than being added alongside it, because
  `sst`/`ssh`/`sss`/`curl` was found to carry more distinct spatial
  information for the patch-based models than `sst`/`ssh`/`sss`/`|wind|`
  did, once curl was available. This was verified directly on this
  project's own data, not assumed from the source paper: the
  spatially-aware models (CNN, ViT) and the sequential model (LSTM)
  showed a measurable accuracy improvement after the swap, while the
  flat-feature FFNN, which cannot exploit spatial structure, showed
  almost no change — consistent with curl being specifically a *spatial*
  forcing signal.
- **The real-track feature set (7 fields) is deliberately smaller than
  the synthetic-track set (16 fields, including one-hot cluster
  columns), rather than padded to match it.** Only SST, SSS, and surface
  currents currently have a real source. Wind, curl, SSH, and cluster
  membership are omitted entirely from the real track rather than
  substituted with a synthetic stand-in, since mixing real and synthetic
  values within a single "real-data" training run would undermine the
  purpose of having a real-data track at all.

### 3.2 Compaction via satellite-embedding architectures

The specification requires that satellite information be reduced to a
*compact embedding* using a CNN, Vision Transformer, Autoencoder, GNN, or
attention-based hybrid architecture, rather than consumed as raw,
high-dimensional imagery. This project implements that requirement
literally for the two spatial models:

- A 5×5-cell neighborhood of the surface grid is extracted around each
  profile's location (`extract_patch()` in `models/dl_pipeline.py`,
  `build_real_patches()` in `real/real_training.py` for the real track),
  giving a small tensor of shape `(channels, 5, 5)`.
- The **CNN** compacts this patch through two convolutional layers
  followed by global average pooling, producing a single 32-dimensional
  embedding vector per profile.
- The **ViT** compacts the same patch by treating each of the 25 cells as
  a token, projecting each to an embedding with a learned positional
  encoding, passing the sequence through one self-attention layer, and
  mean-pooling the result into a single embedding vector.

In both cases, a spatial input of 100 raw values (4 channels × 25 cells,
in the synthetic case) is reduced to a much smaller latent vector before
any prediction is made — this reduction is the literal "compact
embedding" the specification asks for, and it is the primary respect in
which the CNN and ViT differ methodologically from the remaining five
models, which consume the same location as a flat, unreduced feature
vector. The full architectural detail and an ASCII diagram of this
pipeline are provided in `DATASET.md`, Section 3.

## 4. Summary

| Stage | Real track | Synthetic track |
|---|---|---|
| Acquisition | `copernicusmarine` CLI (CMEMS) and direct MOSDAC download | Generated from physically-motivated formulas (`synthetic/ocean_pipeline_demo.py`) |
| Harmonization | Variable-based file resolution; HDF5 fill-value/scale-offset decoding | Not required; output is already aligned |
| Temporal resolution | Resampled to daily | Generated directly at daily resolution |
| Spatial resolution | Regridded to 0.25° via block-mean averaging | Generated at 1.0° (map visualization only; a stated limitation) |
| Derived features | Wind stress curl not available (no real wind product yet) | Wind stress curl computed via finite differences |
| Depth alignment | Nearest-neighbor matching to the 15 standard depths | Generated directly at the 15 standard depths |
| Feature scaling | `StandardScaler`, fit on the training split only | `StandardScaler`, fit on the training split only |
| Compaction (CNN/ViT) | 5×5 patch of `sst`/`sss`/`uo`/`vo` → embedding | 5×5 patch of `sst`/`ssh`/`sss`/`curl` → embedding |
