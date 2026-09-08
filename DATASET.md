# Dataset Composition, Feature Selection, and Satellite Embedding Methodology

This document describes, in a form that can be reviewed directly from the
repository without executing any code, which surface variables were
selected as model inputs, why they were selected, and precisely how
satellite imagery is converted into the "compact satellite embedding"
required by the problem statement. Two representative samples of the
dataset are committed alongside this document:

- **[`synthetic/sample_dataset.csv`](synthetic/sample_dataset.csv)** — 25 rows produced directly by
  `synthetic.ocean_pipeline_demo.build_training_table()`, rounded for
  readability and otherwise unmodified. This is the exact table every
  synthetic-track model is trained on.
- **[`real/sample_dataset.csv`](real/sample_dataset.csv)** — 25 rows produced directly by
  `real.real_training.build_real_training_table()`, with the same
  treatment. This is the exact table every real-data-track model is
  trained on. Source and provenance of the underlying MOSDAC/CMEMS files
  are documented in `real/PROVENANCE.md`.

Both files are intentionally small enough to be opened and inspected
directly on GitHub, so the dataset's structure can be verified without
relying on this document's description of it.

## 1. Required Surface Inputs vs. Inputs Actually Available

The problem statement specifies five required surface input variables.
The table below states, per data track, which of these are genuinely
available and which are not.

| Required input | Synthetic track | Real track |
|---|---|---|
| SST | `sst` — generated from a physically-motivated formula | `sst` — real, CMEMS `thetao` |
| SSS | `sss` — generated from a physically-motivated formula | `sss` — real, CMEMS `so` |
| SSH / SLA | `ssh` — synthetic Gaussian "eddy" field | Not available — no real altimetry file was obtained |
| Surface currents (U, V) | Not modeled — no synthetic current field exists | `uo`, `vo` — real, CMEMS |
| Surface winds (U, V) | `u_wind`, `v_wind` — synthetic, plus a derived `curl` field | Not available — no real wind product was obtained |

Two separate tracks exist because the real data currently obtained does
not cover all five required inputs (see `real/PROVENANCE.md` for an exact
accounting of which files exist and which are missing). The synthetic
track supplies every required input via a physically-motivated formula so
that the full seven-model comparison could be run and evaluated end to
end while real-data acquisition continues.

## 2. Feature Selection: Candidate Fields vs. Fields Consumed by Each Model Type

`get_satellite_grid()` (synthetic pipeline) produces nine candidate
fields; the real-data loaders produce seven. Not every candidate field is
passed to every model. Two distinct "relevant feature" subsets exist,
because the tabular models and the satellite-patch models consume the
data in fundamentally different forms.

| Field | Present in synthetic grid | Present in real data | Used by tabular models (`FEATURE_COLS` / `REAL_FEATURE_COLS`) | Used by CNN/ViT satellite patch (`PATCH_CHANNELS` / `PATCH_CHANNELS_REAL`) |
|---|:-:|:-:|:-:|:-:|
| `lat`, `lon` | Yes | Yes | Yes | Used to locate the patch; not included as a channel within it |
| `day` | Yes | Yes | Yes | Concatenated after the patch embedding; not spatial |
| `sst` | Yes | Yes | Yes | Yes |
| `ssh` | Yes | No | Yes (synthetic track only) | Yes (synthetic track only) |
| `sss` | Yes | Yes | Yes | Yes |
| `wind` (scalar speed) | Yes | No | Superseded by `u_wind`/`v_wind`/`curl` below | No |
| `u_wind`, `v_wind` | Yes | No | Yes (synthetic track only) | No — only the derived `curl` field enters the patch |
| `curl` (derived) | Yes | No | Yes (synthetic track only) | Yes (synthetic track only) |
| `uo`, `vo` (currents) | No | Yes | Yes (real track only) | Yes (real track only) |
| `cluster_0`–`cluster_5` (derived) | Yes | No | Yes, one-hot (synthetic track only) | No |

**Rationale for using `curl` rather than raw wind components in the
satellite patch.** `compute_wind_stress_curl()` derives an
Ekman-pumping-relevant curl field from `u_wind`/`v_wind` using central
finite differences with cos(lat) weighting. This is the specific
additional input identified by Xie et al. (2022, Attention U-Net model for
subsurface temperature reconstruction) as the single largest accuracy
contributor below approximately 50 m depth. It replaced the scalar `wind`
field previously used in `PATCH_CHANNELS`, since `sst/ssh/sss/curl`
provides more distinct spatial information than `sst/ssh/sss/|wind|`, a
result confirmed empirically on this project's own data (see README,
"The models" section).

**Rationale for the smaller real-track feature set (seven fields vs.
sixteen).** Only SST, SSS, and surface currents currently have a real data
source. No real wind product or SSH/SLA file has yet been obtained, so
`real/real_training.py`'s `REAL_FEATURE_COLS` and `PATCH_CHANNELS_REAL`
omit these variables entirely rather than substituting a synthetic
stand-in for them. The full reasoning is documented in
`PROJECT_REPORT.txt`, Section 1a, item 3.

The one-hot cluster columns (`cluster_0` through `cluster_5`) are produced
by `spatiotemporal_clusters()`, a K-means procedure applied to
`(lat, lon, day, ssh)`, informed by the adaptive clustering framework of
Loo et al. (2026). These columns are used only by the tabular models and
are not part of the satellite patch.

## 3. Satellite Imagery Usage: The Compact Embedding Requirement

The problem statement specifies that compact satellite embeddings be
generated using a CNN, Vision Transformer, Autoencoder, GNN, or
attention-based hybrid architecture. The CNN and ViT implementations in
this project satisfy this requirement directly: each is trained on a real
spatial crop of the satellite grid, rather than a single point-wise
feature vector.

```
Satellite grid for day D                Profile at (lat, lon), day D
(full basin, every 1.0 deg cell,   ---->  locate nearest grid cell (i, j)
 4 channels: sst/ssh/sss/curl)            extract a 5x5 neighborhood
                                           around (i, j); edge-replicated
                                           padding is applied at basin
                                           boundaries
                                                    |
                                                    v
                                   patch: shape (4 channels, 5, 5)
                                                    |
                          -------------------------------------------------
                          |                                               |
                          v                                               v
                   CNN (PatchCNN)                                  ViT (PatchViT)
             Conv2d(3x3) -> ReLU                          each of the 25 cells treated
             Conv2d(3x3) -> ReLU                          as one token; linear projection
             GlobalAvgPool                                + learned positional embedding
                          |                          1 TransformerEncoder layer (self-attention)
                          v                                mean-pool over tokens (no CLS token)
             32-dim embedding vector                                       |
                          |                                                v
                          |                                    embedding vector
                          -------------------------------------------------
                                                    |
                                    concatenated with `day` (non-spatial)
                                                    |
                                                    v
                                    small regression head
                                                    |
                                                    v
                              predicted temperature at 15 standard depths
```

`extract_patch()` (`models/dl_pipeline.py`) is the single function
responsible for constructing this crop, and it is shared identically by
both the CNN and the ViT. As a result, the CNN-vs-ViT comparison in the
Results tab isolates convolution versus attention as the only variable,
rather than conflating it with a difference in input data. The real-data
track uses the equivalent mechanism, `build_real_patches()` in
`real/real_training.py`, with real `sst/sss/uo/vo` channels in place of
the synthetic `sst/ssh/sss/curl` channels.

This constitutes the most direct implementation of "satellite embedding"
in this project: a compact latent vector derived from a spatial image
patch, as distinct from the tabular models (FFNN, Random Forest, GNN,
Autoencoder, LSTM), which represent the same location as a flat feature
vector with no spatial context. Full architecture-level detail — exact
layer dimensions, the reason ViT outperformed CNN, and the reason the GNN
required a skip connection — is documented in `PROJECT_REPORT.txt`,
Section 1c (Section E) and Section 2, item 4.

## 4. Reference: Where Each Component Resides in the Repository

| Component | Location |
|---|---|
| Synthetic feature/target table builder | `synthetic/ocean_pipeline_demo.py` → `build_training_table()` |
| Real feature/target table builder | `real/real_training.py` → `build_real_training_table()` |
| Tabular feature list (synthetic) | `models/dl_pipeline.py` → `FEATURE_COLS` |
| Tabular feature list (real) | `real/real_training.py` → `REAL_FEATURE_COLS` |
| Satellite-patch channel list (synthetic) | `models/dl_pipeline.py` → `PATCH_CHANNELS` |
| Satellite-patch channel list (real) | `real/real_training.py` → `PATCH_CHANNELS_REAL` |
| Patch extraction | `models/dl_pipeline.py` → `extract_patch()` (synthetic); `real/real_training.py` → `build_real_patches()` (real) |
| Standard depth levels (prediction target) | `synthetic/ocean_pipeline_demo.py` → `DEPTH_LEVELS`; `real/real_training.py` → `STANDARD_DEPTHS` |
| Raw real-data source files and provenance record | `real/PROVENANCE.md`, `real/data/CHECKSUMS.sha256` |
