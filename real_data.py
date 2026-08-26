"""
Real-data loaders for Sea Green -- reads the actual downloaded files:

  - MOSDAC/*.h5            INSAT-3DR L2B SST (ISRO/SAC), half-hourly, 25 Aug 2026
  - cmems_mod_glo_phy-thetao_..._.nc   CMEMS GLOBAL_ANALYSISFORECAST_PHY_001_024
                            "thetao" (sea water potential temperature), single
                            near-surface level (~0.49 m), 6-hourly, 1-26 Aug 2026

These are real satellite / reanalysis products, not synthetic -- unlike
ocean_pipeline_demo.py, which stays a physically-motivated simulation used
to validate the ML + heatwave-detection logic end to end (see README for why).

Run `python export_real_data.py` after adding/refreshing files here to
regenerate public/data_real.json.
"""
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
import h5py

LAT_RANGE = (5, 30)
LON_RANGE = (45, 105)

HEATWAVE_CATEGORIES = [(0.5, "Watch"), (1.0, "Warning"), (1.5, "Severe"), (2.0, "Extreme")]

MOSDAC_DIR = "MOSDAC"

# Resolve CMEMS files by the variable they actually contain, not by filename --
# the user's downloads get renamed by the browser ("Sea Surface temp.nc",
# "Sea surface salinity.nc") and don't reliably match a naming convention.
_VAR_LABEL = {"thetao": "sst", "so": "sss"}


def _classify(anomaly):
    category = "Normal"
    for thresh, label in HEATWAVE_CATEGORIES:
        if anomaly >= thresh:
            category = label
    return category


def _find_cmems_dataset(varname):
    for path in sorted(glob.glob("*.nc")):
        try:
            ds = xr.open_dataset(path)
        except Exception:
            continue
        if varname in ds.data_vars:
            return ds, path
        ds.close()
    raise FileNotFoundError(f"No .nc file in {os.getcwd()} contains variable '{varname}'")


def load_cmems_series(varname="thetao", stride=6):
    """
    Real basin surface monitor (SST if varname='thetao', SSS if varname='so'):
    daily-mean series over the pulled window, plus one gridded "current
    conditions" snapshot (most recent timestep). Anomaly is against the
    window's own mean (short-window baseline, not a 30-yr climatology --
    labelled as such in the UI).
    """
    ds, path = _find_cmems_dataset(varname)
    label = _VAR_LABEL.get(varname, varname)

    lon_hi = min(LON_RANGE[1], float(ds.longitude.max()))
    sub = ds[varname].isel(depth=0).sel(
        latitude=slice(*LAT_RANGE), longitude=slice(LON_RANGE[0], lon_hi)
    )

    basin_mean = sub.mean(dim=["latitude", "longitude"], skipna=True).to_series()
    daily = basin_mean.resample("1D").mean()
    window_mean = float(daily.mean())

    series = [
        {"date": str(d.date()), label: round(float(v), 3), "anomaly": round(float(v - window_mean), 3)}
        for d, v in daily.items()
    ]
    today_anomaly = series[-1]["anomaly"]

    latest = sub.isel(time=-1)
    lat2d, lon2d = np.meshgrid(latest.latitude.values, latest.longitude.values, indexing="ij")
    vals = latest.values
    mask = ~np.isnan(vals)
    lat_s, lon_s, val_s = lat2d[mask][::stride], lon2d[mask][::stride], vals[mask][::stride]

    return {
        "source_file": os.path.basename(path),
        "variable": varname,
        "window_start": series[0]["date"],
        "window_end": series[-1]["date"],
        f"window_mean_{label}": round(window_mean, 3),
        "series": series,
        "today_anomaly": round(today_anomaly, 3),
        "today_category": _classify(today_anomaly),
        "snapshot": {
            "time": str(latest.time.values)[:19],
            "lat": [round(float(v), 2) for v in lat_s],
            "lon": [round(float(v), 2) for v in lon_s],
            label: [round(float(v), 2) for v in val_s],
        },
    }


def _read_mosdac_frame(path, stride=8):
    with h5py.File(path, "r") as f:
        lat = f["Latitude"][::stride, ::stride].astype(np.float32) * f["Latitude"].attrs["scale_factor"][0]
        lon = f["Longitude"][::stride, ::stride].astype(np.float32) * f["Longitude"].attrs["scale_factor"][0]
        sst_k = f["SST"][0, ::stride, ::stride]
        fill = f["SST"].attrs["_FillValue"][0]
        acq_time = f.attrs["Acquisition_Time_in_GMT"]
        acq_time = acq_time.decode() if isinstance(acq_time, bytes) else str(acq_time)

    valid = (sst_k != fill) & (lat < 1000) & (lon < 1000)
    valid &= (lat >= LAT_RANGE[0]) & (lat <= LAT_RANGE[1]) & (lon >= LON_RANGE[0]) & (lon <= LON_RANGE[1])
    sst_c = sst_k[valid] - 273.15

    return {
        "time": acq_time,
        "lat": [round(float(v), 2) for v in lat[valid]],
        "lon": [round(float(v), 2) for v in lon[valid]],
        "sst": [round(float(v), 2) for v in sst_c],
    }


def load_mosdac_series(n_frames=8, stride=8):
    """Real INSAT-3DR satellite SST passes, cropped to the study region."""
    files = sorted(glob.glob(os.path.join(MOSDAC_DIR, "*_L2B_SST_*.h5")))
    if not files:
        raise FileNotFoundError(f"No MOSDAC L2B SST files found in {MOSDAC_DIR}/")

    step = max(1, len(files) // n_frames)
    chosen = files[::step][:n_frames]
    return [_read_mosdac_frame(p, stride=stride) for p in chosen]


if __name__ == "__main__":
    sst = load_cmems_series("thetao")
    print("CMEMS SST window:", sst["window_start"], "->", sst["window_end"],
          "| today anomaly:", sst["today_anomaly"], sst["today_category"],
          "| file:", sst["source_file"])
    sss = load_cmems_series("so")
    print("CMEMS SSS window:", sss["window_start"], "->", sss["window_end"],
          "| today anomaly:", sss["today_anomaly"], "| file:", sss["source_file"])
    frames = load_mosdac_series()
    print(f"MOSDAC frames: {len(frames)}, first={frames[0]['time']}, "
          f"points/frame~{len(frames[0]['sst'])}")
