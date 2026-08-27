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

TARGET_RESOLUTION_DEG = 0.25  # the problem statement's required spatial resolution

# Resolve CMEMS files by the variable they actually contain, not by filename --
# the user's downloads get renamed by the browser ("Sea Surface temp.nc",
# "Sea surface salinity.nc") and don't reliably match a naming convention.
_VAR_LABEL = {"thetao": "sst", "so": "sss", "chl": "chl", "phyc": "phyc"}


def _regrid_to_target(obj):
    """
    Block-mean regrid from CMEMS's native ~0.083 deg grid to the spec's
    required 0.25 deg, via xarray.coarsen() (genuine spatial averaging,
    not decimation/subsampling) -- the problem statement explicitly allows
    "spatial and temporal interpolation/regridding" when a native product
    isn't already at the required resolution, which is exactly this case.
    Works on either a DataArray or a Dataset (so it can regrid uo+vo
    together, keeping their block-averages co-located for a correct
    vector-field regrid rather than resampling each component separately).
    No-ops (returns unchanged) if the grid is already coarser than target
    (e.g. the 0.25 deg chlorophyll product).
    """
    lat_step = float(abs(obj.latitude[1] - obj.latitude[0]))
    factor = round(TARGET_RESOLUTION_DEG / lat_step)
    if factor <= 1:
        return obj
    return obj.coarsen(latitude=factor, longitude=factor, boundary="trim").mean()


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


def load_cmems_series(varname="thetao", stride=2):
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

    latest = _regrid_to_target(sub.isel(time=-1))
    lat2d, lon2d = np.meshgrid(latest.latitude.values, latest.longitude.values, indexing="ij")
    vals = latest.values
    mask = ~np.isnan(vals)
    # `stride` here is now purely a plotting-density choice on an ALREADY
    # 0.25-deg-regridded field (not a substitute for regridding).
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


def load_cmems_currents_series(stride=10):
    """
    Real surface current field (uo, vo) -- one of the spec's five required
    input variables. Returns a basin-mean speed series over the pulled
    window plus a subsampled vector-field snapshot (lat, lon, uo, vo, speed,
    heading) for the most recent day, for drawing an actual arrow/quiver map.
    """
    ds, path = _find_cmems_dataset("uo")
    lon_hi = min(LON_RANGE[1], float(ds.longitude.max()))
    sub = ds.isel(depth=0).sel(latitude=slice(*LAT_RANGE), longitude=slice(LON_RANGE[0], lon_hi))

    speed = np.sqrt(sub["uo"] ** 2 + sub["vo"] ** 2)
    basin_mean = speed.mean(dim=["latitude", "longitude"], skipna=True).to_series()
    daily = basin_mean.resample("1D").mean()
    window_mean = float(daily.mean())

    series = [
        {"date": str(d.date()), "speed": round(float(v), 4), "anomaly": round(float(v - window_mean), 4)}
        for d, v in daily.items()
    ]

    # Regrid uo/vo TOGETHER (as one Dataset) before computing speed/heading --
    # averaging the vector components first, then deriving speed/direction
    # from the averaged vector, is the physically correct way to regrid a
    # vector field (averaging speed and heading separately would be wrong).
    latest = _regrid_to_target(sub[["uo", "vo"]].isel(time=-1))
    lat2d, lon2d = np.meshgrid(latest.latitude.values, latest.longitude.values, indexing="ij")
    uo, vo = latest["uo"].values, latest["vo"].values
    mask = ~np.isnan(uo) & ~np.isnan(vo)
    # `stride` is now a plotting-density choice (arrows overlap if too
    # dense) applied on top of the already-0.25-deg-regridded field.
    lat_s, lon_s = lat2d[mask][::stride], lon2d[mask][::stride]
    uo_s, vo_s = uo[mask][::stride], vo[mask][::stride]
    speed_s = np.sqrt(uo_s ** 2 + vo_s ** 2)
    # Compass heading (0=N, 90=E, clockwise) from eastward/northward components,
    # matching Plotly's marker.angle convention (0=up, clockwise).
    heading = (90 - np.degrees(np.arctan2(vo_s, uo_s))) % 360

    return {
        "source_file": os.path.basename(path),
        "window_start": series[0]["date"],
        "window_end": series[-1]["date"],
        "window_mean_speed": round(window_mean, 4),
        "series": series,
        "today_anomaly": series[-1]["anomaly"],
        "snapshot": {
            "time": str(latest.time.values)[:19],
            "lat": [round(float(v), 2) for v in lat_s],
            "lon": [round(float(v), 2) for v in lon_s],
            "speed": [round(float(v), 3) for v in speed_s],
            "heading": [round(float(v), 1) for v in heading],
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
    cur = load_cmems_currents_series()
    print("CMEMS currents window:", cur["window_start"], "->", cur["window_end"],
          "| today anomaly:", cur["today_anomaly"], "| file:", cur["source_file"],
          "| snapshot points:", len(cur["snapshot"]["speed"]))

    chl = load_cmems_series("chl")
    print("CMEMS chlorophyll window:", chl["window_start"], "->", chl["window_end"],
          "| today anomaly:", chl["today_anomaly"], "| file:", chl["source_file"])

    frames = load_mosdac_series()
    print(f"MOSDAC frames: {len(frames)}, first={frames[0]['time']}, "
          f"points/frame~{len(frames[0]['sst'])}")
