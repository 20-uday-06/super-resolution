# Download ASTER scenes in parallel (month‑by‑month) via earthaccess
# ------------------------------------------------------------------
import os, time, calendar
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser

import earthaccess           #  pip install earthaccess
from geopy.geocoders import Nominatim   # pip install geopy
from geopy.exc import GeocoderTimedOut


# ── utils ─────────────────────────────────────────────────────────
def bbox_from_place(place, buffer_deg=1.0):
    geolocator = Nominatim(user_agent="aster_downloader")
    try:
        loc = geolocator.geocode(place)
    except GeocoderTimedOut:
        raise RuntimeError("Geocoding timed‑out; try again.")
    if not loc:
        raise ValueError(f"Could not resolve place name '{place}'")
    lat, lon = loc.latitude, loc.longitude
    # (west,south,east,north)
    return (lon-buffer_deg, lat-buffer_deg, lon+buffer_deg, lat+buffer_deg)


def daterange_for_month(year, month):
    start = f"{year}-{month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    end   = f"{year}-{month:02d}-{last_day:02d}"
    return start, end


# ── main downloader ───────────────────────────────────────────────
def download_month(product, year, month, bbox, dest, session):
    start, end = daterange_for_month(year, month)

    print(f"🔍  Searching {product}  {start} → {end}")
    try:
        granules = earthaccess.search_data(
            short_name = product,
            temporal   = (start, end),
            bounding_box = bbox,
            # ASTER is not fully cloud‑hosted yet; search both pools
            cloud_hosted = False
        )
    except Exception as e:
        print(f"⚠️  Search failed for {start}‑{end}: {e}")
        return 0

    print(f"📦  {len(granules)} granules found  → downloading …")

    count = 0
    for g in granules:
        try:
            earthaccess.download(g, dest)
            count += 1
        except Exception as e:
            print(f"  – failed {g.title}: {e}")
    return count


def aster_parallel_downloader(
        years, product, bbox, threads, out_root):

    os.makedirs(out_root, exist_ok=True)
    earthaccess.login()        # interactive prompt (token cached)

    total_dl = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = []
        for yr in years:
            for m in range(1, 13):
                month_dir = os.path.join(out_root, f"{yr}", f"{m:02d}")
                os.makedirs(month_dir, exist_ok=True)
                futs.append(
                    ex.submit(download_month,
                              product, yr, m, bbox, month_dir, None)
                )

        for f in as_completed(futs):
            total_dl += f.result()

    print(f"\n✅  Finished – downloaded {total_dl} ASTER granules.")


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = ArgumentParser(
        description="Parallel ASTER downloader (year/month grid)")
    p.add_argument('--year_begin', type=int, default=2021)
    p.add_argument('--year_end',   type=int, default=2022,
                   help="exclusive upper bound (like range)")
    p.add_argument('--product',    type=str, default="AST_L1T",
                   help="ASTER short‑name (e.g. AST_L1T, AST_07XT)")
    p.add_argument('--bbox',       type=str,
                   help="west,south,east,north (comma‑sep)  –OR– use --place")
    p.add_argument('--place',      type=str,
                   help="Resolve a place name to a bbox (e.g. 'Delhi')")
    p.add_argument('--threads',    type=int, default=6,
                   help="parallel worker threads")
    p.add_argument('--outdir',     type=str, default="ASTER",
                   help="root folder for downloads")
    args = p.parse_args()

    # years list like range(year_begin, year_end)
    year_list = list(np.arange(args.year_begin, args.year_end))

    # bounding‑box logic ------------------------------------------------------
    if args.place:
        bbox = bbox_from_place(args.place)
    elif args.bbox:
        bbox = tuple(map(float, args.bbox.split(',')))
        if len(bbox) != 4:
            raise ValueError("--bbox must be four comma‑sep numbers")
    else:
        raise SystemExit("❌  Provide either --bbox or --place")

    print(f"🗺  Bounding box  {bbox}")

    aster_parallel_downloader(
        years   = year_list,
        product = args.product,
        bbox    = bbox,
        threads = args.threads,
        out_root= args.outdir
    )
