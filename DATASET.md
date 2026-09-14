# Dataset Composition, Feature Selection, and Satellite Embedding Methodology

This document describes, in a form that can be reviewed directly from the
repository without executing any code, which surface variables were
selected as model inputs, why they were selected, and precisely how
satellite imagery is converted into the "compact satellite embedding"
required by the problem statement. A representative sample of the
dataset is committed alongside this document:

- **[`real/sample_dataset.csv`](real/sample_dataset.csv)** — 25 rows produced directly by
  `real.real_training.build_real_training_table()`, rounded for
  readability and otherwise unmodified. This is the exact table every
  model in this project is trained on. Source and provenance of the
  underlying MOSDAC/CMEMS files are documented in `real/PROVENANCE.md`.

This file is intentionally small enough to be opened and inspected
directly on GitHub, so the dataset's structure can be verified without
relying on this document's description of it.

There is a single data track in this project: every number produced by
this codebase — the Live Monitor tab, the model/reconstruction demo, and
the Results tab — is trained and evaluated on real MOSDAC/CMEMS data.
An earlier synthetic data track existed to validate the modeling pipeline
end to end while real-data acquisition for a fuller feature set was in
progress; it has since been removed, since no sourcing exists for it
going forward. Any input listed as unavailable below is genuinely
unmodeled, not filled with a synthetic stand-in.

## 1. Required Surface Inputs vs. Inputs Actually Available

The problem statement specifies five required surface input variables.
The table below states which are genuinely available and which are not.

| Required input | Status |
|---|---|
| SST | Available — real, CMEMS `thetao` |
| SSS | Available — real, CMEMS `so` |
| SSH / SLA | Not available — no real altimetry file has been obtained |
| Surface currents (U, V) | Available — real, CMEMS `uo`/`vo` |
| Surface winds (U, V) | Not available — no real wind product has been obtained |

Four of the five required inputs are real and modeled; SSH/SLA and
surface winds remain unmodeled entirely, pending a real altimetry and
wind product respectively (see `real/PROVENANCE.md` for an exact
accounting of which files exist and which are missing, and README.md,
"Real data: MOSDAC and CMEMS," for the acquisition path for each).

## 2. Feature Selection: Candidate Fields vs. Fields Consumed by Each Model Type

The real-data loaders produce seven candidate fields. Not every candidate
field is passed to every model — two distinct "relevant feature" subsets
exist, because the tabular models and the satellite-patch models consume
the data in fundamentally different forms.

| Field | Used by tabular models (`REAL_FEATURE_COLS`) | Used by CNN/ViT satellite patch (`PATCH_CHANNELS_REAL`) |
|---|:-:|:-:|
| `lat`, `lon` | Yes | Used to locate the patch; not included as a channel within it |
| `day` | Yes | Concatenated after the patch embedding; not spatial |
| `sst` | Yes | Yes |
| `sss` | Yes | Yes |
| `uo`, `vo` (currents) | Yes | Yes |

**Why the feature set has only seven fields, not the full sixteen the
specification's variable list would imply.** Only SST, SSS, and surface
currents currently have a real data source. No real wind product or
SSH/SLA file has yet been obtained, so `real/real_training.py`'s
`REAL_FEATURE_COLS` and `PATCH_CHANNELS_REAL` omit these variables
entirely rather than substituting a placeholder for them. An earlier
build of this project modeled wind stress curl (`compute_wind_stress_curl()`,
following Xie et al. 2022's Attention U-Net paper) as a synthetic-only
feature; it was dropped along with the rest of the synthetic track, since
it required a real wind product this project does not have. It remains a
documented option for a future build once such a product is sourced.

Region-based clustering (`build_clusters()` in `real/export_real_results.py`)
groups observations by `(lat, lon, day)` via K-means, informed by the
adaptive clustering framework of Loo et al. (2026), and feeds a `cluster`
label to the map overlay and the test-point table. This is a lighter
replacement for an earlier `(lat, lon, day, ssh)`-weighted version that
existed when SSH was synthetically available; it is not part of either
`REAL_FEATURE_COLS` or the satellite patch.

## 3. Satellite Imagery Usage: The Compact Embedding Requirement

The problem statement specifies that compact satellite embeddings be
generated using a CNN, Vision Transformer, Autoencoder, GNN, or
attention-based hybrid architecture. The CNN and ViT implementations in
this project satisfy this requirement directly: each is trained on a real
spatial crop of the satellite grid, rather than a single point-wise
feature vector.

```
Satellite grid for day D                Test point at (lat, lon), day D
(real CMEMS grid, native 0.083 deg,---->  locate nearest grid cell (i, j)
 4 channels: sst/sss/uo/vo)               extract a 5x5 neighborhood
                                           around (i, j); NaN (land) cells
                                           filled with the patch's own
                                           valid-cell mean per channel
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

`build_real_patches()` (`real/real_training.py`) is the single function
responsible for constructing this crop, and it is shared identically by
both the CNN and the ViT. As a result, the CNN-vs-ViT comparison in the
Results tab isolates convolution versus attention as the only variable,
rather than conflating it with a difference in input data.

This constitutes the most direct implementation of "satellite embedding"
in this project: a compact latent vector derived from a spatial image
patch, as distinct from the tabular models (FFNN, Random Forest, GNN,
Autoencoder, LSTM), which represent the same location as a flat feature
vector with no spatial context. On real data, the tabular/graph-based
models (Random Forest, GNN) outperform the patch-based models (CNN, ViT)
— see README.md, "The models," for the full real-data results table and
the reasoning behind this reversal from the earlier synthetic-track
finding.

## 4. Reference: Where Each Component Resides in the Repository

| Component | Location |
|---|---|
| Feature/target table builder | `real/real_training.py` → `build_real_training_table()` |
| Tabular feature list | `real/real_training.py` → `REAL_FEATURE_COLS` |
| Satellite-patch channel list | `real/real_training.py` → `PATCH_CHANNELS_REAL` |
| Patch extraction | `real/real_training.py` → `build_real_patches()` |
| Model architecture toolbox (FFNN, CNN, ViT, GNN, Autoencoder, LSTM) | `models/dl_pipeline.py` |
| Training orchestration (all 7 models, one call) | `real/real_training.py` → `train_real_and_evaluate()` |
| Standard depth levels (prediction target) | `real/real_training.py` → `STANDARD_DEPTHS` |
| Dashboard export (metrics, map animation, region clusters) | `real/export_real_results.py` |
| Raw real-data source files and provenance record | `real/PROVENANCE.md`, `real/data/CHECKSUMS.sha256` |
