"""
Bake real MOSDAC + CMEMS data into public/data_real.json for the static
dashboard. Run after refreshing files in MOSDAC/ or the CMEMS .nc file:

    python export_real_data.py

The raw source files (MOSDAC/*.h5, cmems_*.nc) are large (100s of MB) and
gitignored -- only this compact derived JSON is committed.
"""
import json
import os
from real_data import load_cmems_series, load_mosdac_series


def main():
    print("Loading real CMEMS basin SST series + current snapshot...")
    cmems_sst = load_cmems_series("thetao")

    print("Loading real CMEMS basin SSS series + current snapshot...")
    cmems_sss = load_cmems_series("so")

    print("Loading real MOSDAC (INSAT-3DR) satellite passes...")
    mosdac_frames = load_mosdac_series()

    bundle = {
        "source": {
            "cmems_sst": f"CMEMS GLOBAL_ANALYSISFORECAST_PHY_001_024 (thetao, ~0.49m), {cmems_sst['source_file']}",
            "cmems_sss": f"CMEMS GLOBAL_ANALYSISFORECAST_PHY_001_024 (so, ~0.49m), {cmems_sss['source_file']}",
            "mosdac": "INSAT-3DR L2B SST V02R00, ISRO/SAC, 25 Aug 2026",
        },
        "cmems": cmems_sst,
        "cmems_sss": cmems_sss,
        "mosdac_frames": mosdac_frames,
    }

    out_path = "public/data_real.json"
    with open(out_path, "w") as f:
        json.dump(bundle, f)

    print(f"Wrote {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
