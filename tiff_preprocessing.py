# Resample ASTER GeoTIFFs to 250 m and 1 km for super‑res work
# -----------------------------------------------------------
import os, time, glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser


# ── core resample helper ────────────────────────────────────
def resample_and_save(src_path, out_dir, native_res, targets):
    """
    Parameters
    ----------
    src_path   : str – path to one ASTER GeoTIFF
    out_dir    : folder where resampled files go
    native_res : 15 or 30  (metres/pixel of the input)
    targets    : dict(label -> metreResolution)  e.g. {"250m":250, "1km":1000}
    """
    fname = os.path.basename(src_path)
    with rasterio.open(src_path) as src:
        band    = src.read(1)            # single‑band; change if multi‑band
        prof_in = src.profile

        for lab, tgt_res in targets.items():
            scale       = native_res / tgt_res
            new_h       = int(src.height / scale)
            new_w       = int(src.width  / scale)
            if new_h < 1 or new_w < 1:
                print(f"skip {lab} for {fname}: too small")
                continue

            # resample
            data = src.read(
                out_shape=(1, new_h, new_w),
                resampling=Resampling.bilinear
            )

            # corrected GeoTransform
            transform = (
                src.transform *
                Affine.scale(src.width / new_w, src.height / new_h)
            )

            prof_out = prof_in.copy()
            prof_out.update({
                "height": new_h,
                "width" : new_w,
                "transform": transform
            })

            out_path = os.path.join(out_dir, f"{lab}_{fname}")
            with rasterio.open(out_path, 'w', **prof_out) as dst:
                dst.write(data)
            print(f"✓ {lab:<4} ➜ {out_path}")



# ── main wrapper ────────────────────────────────────────────
def batch_resample(in_path, out_root, native_res, threads):
    if os.path.isfile(in_path):
        tiffs = [in_path]
    else:
        tiffs = glob.glob(os.path.join(in_path, "*.tif"))

    if not tiffs:
        raise RuntimeError("No .tif files found.")

    os.makedirs(out_root, exist_ok=True)
    targets = {"250m": 250, "1km": 1000}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=threads) as exe:
        futs = [
            exe.submit(resample_and_save, tif, out_root, native_res, targets)
            for tif in tiffs
        ]
        for _ in as_completed(futs):
            pass
    print(f"\nDone: processed {len(tiffs)} file(s) in {(time.time()-t0)/60:.1f} min.")



if __name__ == "__main__":
    p = ArgumentParser(
        description="Resample ASTER GeoTIFF(s) to 250 m and 1 km versions")
    p.add_argument("--input",  required=True,
                   help="GeoTIFF file OR folder of .tif files")
    p.add_argument("--outdir", default="output",
                   help="folder where resampled TIFFs are written")
    p.add_argument("--native_res", type=int, choices=[15,30], default=15,
                   help="native resolution of the ASTER product (m)")
    p.add_argument("--threads", type=int, default=4,
                   help="parallel threads")
    args = p.parse_args()

    batch_resample(args.input, args.outdir, args.native_res, args.threads)
