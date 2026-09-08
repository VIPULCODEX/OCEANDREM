# Data Provenance — `real/data/`

This document allows a reviewer to verify what the "real data" claim in
this repository rests on, without requiring the raw files themselves,
which are excluded from version control (gitignored; 100s of MB to
approximately 800 MB each). The following is committed in their place:

- This document — source, product ID, coverage, and per-file checksum for
  every real data file.
- `CHECKSUMS.sha256` — machine-verifiable hashes, re-checkable with
  `sha256sum -c real/data/CHECKSUMS.sha256` if the files are restored
  locally (for example, after re-running the download).
- The compact derived data these files produce: `public/data_real.json`
  (via `real/export_real_data.py`) and the real seven-model benchmark
  results quoted in `PROJECT_REPORT.txt`, Section 1d (via
  `real/real_training.py`).

**The validation authority for these figures is neither this repository
nor any AI tool used in its construction; it is the producing
institution's own published quality-control process.** See "Where the
ground truth comes from" below.

## Files

| File (in `real/data/`) | Variable(s) | Product | Institution | Spatial coverage | Time range | Native resolution | Size | Downloaded |
|---|---|---|---|---|---|---|---|---|
| `cmems_mod_glo_phy-thetao_..._45.00E-105.00E_5.00N-30.00N_0.49-902.34m_2026-08-01-2026-08-27.nc` | `thetao` (35 depth levels, 0.49–902 m) | `GLOBAL_ANALYSISFORECAST_PHY_001_024` | Mercator Ocean International / Copernicus Marine Service | 5–30°N, 45–105°E (exact match to the specification's required extent) | 2026-08-01 to 2026-08-27, daily | 0.083° | 783 MB | 2026-08-27 |
| `cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_....nc` | `uo`, `vo` (surface currents) | `GLOBAL_ANALYSISFORECAST_PHY_001_024` | Mercator Ocean International / CMEMS | lat −45–30°N, lon 30–~100°E (a superset of the study region, cropped in code) | 2026-08-01 to 2026-08-27, daily | 0.083° | 156 MB | 2026-08-27 |
| `cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m_....nc` | `chl`, `phyc` | `GLOBAL_ANALYSISFORECAST_BGC_001_028` | Mercator Ocean International / CMEMS | same superset extent | 2026-08-01 to 2026-08-27, daily | 0.25° (already at specification resolution) | 18 MB | 2026-08-27 |
| `cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i_....nc` (byte-identical to `Sea Surface temp.nc`; confirmed by matching sha256 values below — the same download, saved under two filenames) | `thetao` (single level, ~0.49 m) | `GLOBAL_ANALYSISFORECAST_PHY_001_024` | Mercator Ocean International / CMEMS | same superset extent | 2026-08-01 to 2026-08-26, 6-hourly | 0.083° | 301 MB | 2026-08-26 |
| `Sea surface salinity.nc` | `so` | `GLOBAL_ANALYSISFORECAST_PHY_001_024` | Mercator Ocean International / CMEMS | same superset extent | 2026-08-01 to 2026-08-26 | 0.083° | 309 MB | 2026-08-26 |
| `MOSDAC/3RIMG_*_L2B_SST_V02R00.h5` (17 files) | SST | INSAT-3DR L2B SST V02R00 | ISRO / Space Applications Centre (MOSDAC) | Full-disk geostationary swath, cropped in code to 5–30°N/45–105°E | 25 August 2026, half-hourly (17 evenly-spaced passes) | native swath (~4 km at nadir) | approximately 15–17 MB each, 369 MB total | 2026-08-26 |

**Known limitation.** The currents, salinity, and single-level-SST files
extend only to approximately 99.9–100°E, short of the specification's
full 105°E boundary; see `LON_RANGE = (45, 99.9)` in
`real/real_training.py`. Only the dedicated multi-depth subsurface
extract covers the full 45–105°E, 5–30°N region exactly, confirmed from
the file's own coordinate arrays (`longitude` runs from 45.0 to 105.0
exactly, 721 points at 0.083° spacing). This is a specific, bounded data
completeness gap rather than an error of ocean basin: every file listed
here covers the Indian Ocean (30–105°E), and none overlaps the Pacific.

Exact per-file hashes are recorded in `CHECKSUMS.sha256` in this
directory.

## Where the Ground Truth Comes From

Neither this codebase nor any AI tool used in its development is the
source of validation for these figures. That authority resides with the
producing institution and is documented externally and citably:

- **CMEMS products** (`GLOBAL_ANALYSISFORECAST_PHY_001_024`,
  `GLOBAL_ANALYSISFORECAST_BGC_001_028`) each publish an official
  **Quality Information Document (QUID)**, issued by the Copernicus
  Marine Service, describing how Mercator Ocean validated that specific
  product against real Argo floats, moorings, and altimeter calibration
  and validation sites, with quantified skill scores. The QUID, not this
  repository, should be cited as the accuracy source for the real
  SST/SSS/currents/chlorophyll fields.
- **MOSDAC INSAT-3DR L2B SST** is calibrated and validated by ISRO/SAC
  against buoy and ship measurements, as documented in ISRO's own product
  validation reports.
- **What this repository's own analysis validates, and what it does
  not.** The `real/real_training.py` benchmark — seven models trained and
  tested on a chronological split of this real data — demonstrates that
  the reconstruction method generalizes to held-out real observations
  drawn from the same reanalysis product. This is **not** equivalent to
  an independent-observation validation in the stricter sense specified
  by the original problem statement, which would require evaluation
  against real Argo profiles that were never assimilated into the CMEMS
  reanalysis. That evaluation step — via `argopy` or the INCOIS Live
  Access Server — remains a known, explicitly tracked gap; see
  `PROJECT_REPORT.txt`, Section 2, item 7, and Section 1c, part J.

## Regenerating This Manifest

If the files in `real/data/` are refreshed, regenerate the checksum file
with:

```bash
cd real/data
find . -type f \( -name "*.nc" -o -name "*.h5" \) -print0 | sort -z \
  | while IFS= read -r -d '' f; do sha256sum "$f" >> CHECKSUMS.sha256; done
```

Then update the table above with the new coverage, date range, and size
values, and re-verify any duplicate content with `sha256sum -c`.
