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
    cmems = load_cmems_series()

    print("Loading real MOSDAC (INSAT-3DR) satellite passes...")
    mosdac_frames = load_mosdac_series()

    bundle = {
        "source": {
            "cmems": "CMEMS GLOBAL_ANALYSISFORECAST_PHY_001_024 (thetao, ~0.49m), Mercator Ocean",
            "mosdac": "INSAT-3DR L2B SST V02R00, ISRO/SAC, 25 Aug 2026",
        },
        "cmems": cmems,
        "mosdac_frames": mosdac_frames,
    }

    out_path = "public/data_real.json"
    with open(out_path, "w") as f:
        json.dump(bundle, f)

    print(f"Wrote {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
