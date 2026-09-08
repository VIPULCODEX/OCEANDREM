# Data provenance — `real/data/`

This file exists so a reviewer can verify what the "real data" claim in this
repo actually rests on, **without needing the raw files themselves** (they're
gitignored — 100s of MB to ~800 MB each, not meant for git). What's
committed instead:

- This document (source, product ID, coverage, per-file checksum).
- `CHECKSUMS.sha256` — machine-verifiable hashes, re-checkable with
  `sha256sum -c real/data/CHECKSUMS.sha256` if the files are ever restored
  locally (e.g. after re-running the download).
- The compact derived JSON these files produce: `public/data_real.json`
  (via `real/export_real_data.py`) and the real 7-model benchmark results
  quoted in `PROJECT_REPORT.txt` §1d (via `real/real_training.py`).

**The validation authority for these numbers is not this repo and not any
AI tool that touched this code — it's the producing institution's own
published QC process.** See "Where the ground truth comes from" below.

## Files

| File (in `real/data/`) | Variable(s) | Product | Institution | Spatial coverage | Time range | Native resolution | Size | Downloaded |
|---|---|---|---|---|---|---|---|---|
| `cmems_mod_glo_phy-thetao_..._45.00E-105.00E_5.00N-30.00N_0.49-902.34m_2026-08-01-2026-08-27.nc` | `thetao` (35 depth levels, 0.49–902 m) | `GLOBAL_ANALYSISFORECAST_PHY_001_024` | Mercator Ocean Int'l / Copernicus Marine Service | 5–30°N, 45–105°E (exact, full spec box) | 2026-08-01 to 2026-08-27, daily | 0.083° | 783 MB | 2026-08-27 |
| `cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_....nc` | `uo`, `vo` (surface currents) | `GLOBAL_ANALYSISFORECAST_PHY_001_024` | Mercator Ocean Int'l / CMEMS | lat −45–30°N, lon 30–~100°E (superset, cropped to region in code) | 2026-08-01 to 2026-08-27, daily | 0.083° | 156 MB | 2026-08-27 |
| `cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m_....nc` | `chl`, `phyc` | `GLOBAL_ANALYSISFORECAST_BGC_001_028` | Mercator Ocean Int'l / CMEMS | same superset box | 2026-08-01 to 2026-08-27, daily | 0.25° (already at spec) | 18 MB | 2026-08-27 |
| `cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i_....nc` **== `Sea Surface temp.nc`** (identical byte content, verified by matching sha256 below — same download, saved under two names) | `thetao` (single level, ~0.49 m) | `GLOBAL_ANALYSISFORECAST_PHY_001_024` | Mercator Ocean Int'l / CMEMS | same superset box | 2026-08-01 to 2026-08-26, 6-hourly | 0.083° | 301 MB | 2026-08-26 |
| `Sea surface salinity.nc` | `so` | `GLOBAL_ANALYSISFORECAST_PHY_001_024` | Mercator Ocean Int'l / CMEMS | same superset box | 2026-08-01 to 2026-08-26 | 0.083° | 309 MB | 2026-08-26 |
| `MOSDAC/3RIMG_*_L2B_SST_V02R00.h5` (17 files) | SST | INSAT-3DR L2B SST V02R00 | ISRO / Space Applications Centre (MOSDAC) | Full-disk geostationary swath, cropped in code to 5–30°N/45–105°E | 25 Aug 2026, half-hourly (17 evenly-spaced passes) | native swath (~4 km at nadir) | ~15–17 MB each, 369 MB total | 2026-08-26 |

**Known gap, stated plainly:** the currents/salinity/single-level-SST files
only extend to ~99.9–100°E, short of the spec's full 105°E — see
`real/real_training.py`'s `LON_RANGE = (45, 99.9)`. Only the dedicated
multi-depth subsurface extract covers the full 45–105°E, 5–30°N box exactly
(confirmed from the file's own coordinate arrays: `longitude` runs 45.0 to
105.0 exactly, 721 points at 0.083°). This is a real, specific data
completeness gap, not a wrong-ocean-basin error — every file here is
Indian Ocean coverage (lon 30–105°E), nowhere near the Pacific.

Per-file exact hashes are in `CHECKSUMS.sha256` in this directory.

## Where the ground truth comes from

Neither this codebase nor any AI tool used while building it is the source
of validation for these numbers. That authority sits with the producing
institution, documented externally and citably:

- **CMEMS products** (`GLOBAL_ANALYSISFORECAST_PHY_001_024`,
  `GLOBAL_ANALYSISFORECAST_BGC_001_028`) each ship an official **Quality
  Information Document (QUID)**, published by Copernicus Marine Service,
  reporting how Mercator Ocean validated that exact product against real
  Argo floats, moorings, and altimeter cal/val sites — with quantified
  skill scores. Cite the QUID, not this repo, as the accuracy source for
  the real SST/SSS/currents/chlorophyll fields.
- **MOSDAC INSAT-3DR L2B SST** is calibrated/validated by ISRO/SAC against
  buoy and ship measurements, documented in ISRO's own product validation
  reports.
- **What this repo's own analysis validates, and what it doesn't:** the
  `real/real_training.py` benchmark (7 models trained/tested on a
  chronological split of this real data) validates that the *reconstruction
  method* generalizes to held-out real observations *drawn from the same
  reanalysis product*. It is **not** an independent-observation validation
  in the stricter sense the original problem spec asks for (evaluating
  against real Argo profiles that never fed into CMEMS's own reanalysis) —
  that step (via `argopy` or INCOIS LAS) is a known, explicitly tracked gap;
  see `PROJECT_REPORT.txt` §2, item 7 and §1c, section J.

## Regenerating this manifest

If the files in `real/data/` are refreshed:

```bash
cd real/data
find . -type f \( -name "*.nc" -o -name "*.h5" \) -print0 | sort -z \
  | while IFS= read -r -d '' f; do sha256sum "$f" >> CHECKSUMS.sha256; done
```

Then update the table above with the new coverage/date-range/size values
and re-verify duplicate content with `sha256sum -c`.
