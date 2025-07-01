# Convert ASTER native (15 m / 30 m) scenes into
#   250 m HR TIFFs   and
#   1 km LR TIFFs
# ----------------------------------------------------------
import os, time, glob
import numpy as np
from argparse import ArgumentParser
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

# ────────────────────────────────────────────────────────────
def resample_array(arr, src_res, tgt_res):
    """Resample a numpy image from src_res (m) to tgt_res (m) with bilinear."""
    scale = src_res / tgt_res
    new_h = int(arr.shape[0] / scale)
    new_w = int(arr.shape[1] / scale)
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def process_one_aster(src_path, band_id, native_res, hr_dir, lr_dir):
    fname = os.path.basename(src_path).replace('.hdf', '.tif')

    # ---------- read one band ----------
    with rasterio.open(src_path) as src:
        band     = src.read(band_id).astype(np.float32)        # 15 m or 30 m
        meta_in  = src.meta.copy()

    # ---------- HR 250 m ----------
    hr_arr   = resample_array(band, native_res, 250)
    hr_meta  = meta_in.copy()
    hr_meta.update({
        "height": hr_arr.shape[0],
        "width" : hr_arr.shape[1],
        "transform": meta_in["transform"] * Affine.scale(native_res/250,
                                                         native_res/250)
    })
    hr_path = os.path.join(hr_dir, "250m_" + fname)
    with rasterio.open(hr_path, 'w', **hr_meta) as dst:
        dst.write(hr_arr, 1)

    # ---------- LR 1 km ----------
    lr_arr   = resample_array(hr_arr, 250, 1000)
    lr_meta  = hr_meta.copy()
    lr_meta.update({
        "height": lr_arr.shape[0],
        "width" : lr_arr.shape[1],
        "transform": hr_meta["transform"] * Affine.scale(250/1000,
                                                         250/1000)
    })
    lr_path = os.path.join(lr_dir, "1km_" + fname)
    with rasterio.open(lr_path, 'w', **lr_meta) as dst:
        dst.write(lr_arr, 1)

    return hr_path, lr_path


# ────────────────────────────────────────────────────────────
def ASTER_Data_Preprocessing(year, product, band_id, native_res):
    root_dir   = f'ASTER/AST_{year}_{product}'
    hdfs_path  = os.path.join(root_dir, 'hdfs_files')     # original .hdf
    tifs_hr    = os.path.join(root_dir, 'tifs_files/250m')
    tifs_lr    = os.path.join(root_dir, 'tifs_files/1km')

    os.makedirs(tifs_hr, exist_ok=True)
    os.makedirs(tifs_lr, exist_ok=True)

    hdfs = sorted(glob.glob(os.path.join(hdfs_path, '*.hdf')))
    if not hdfs:
        print(f"❌ No .hdf files found in {hdfs_path}")
        return

    print(f"🔧 Pre‑processing {len(hdfs)} ASTER files (band {band_id}) …")
    t0 = time.time()

    count = 0
    for hdf in hdfs:
        try:
            process_one_aster(hdf, band_id, native_res, tifs_hr, tifs_lr)
            count += 1
        except Exception as e:
            print(f"⚠️  Failed {os.path.basename(hdf)} → {e}")

    dt = time.time() - t0
    print(f"✅ Finished {count}/{len(hdfs)} files in {dt/60:.2f} min\n")


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = ArgumentParser(
        description="ASTER 1 km / 250 m TIFF generator")
    p.add_argument('--year_begin',  type=int, default=2021)
    p.add_argument('--year_end',    type=int, default=2022,
                   help="exclusive upper bound (like range)")
    p.add_argument('--product',     type=str, default="AST_L1T",
                   help="ASTER product short‑name (AST_L1T, AST_07XT, AST_08)")
    p.add_argument('--band',        type=int, default=2,
                   help="ASTER band number to extract (1‑14)")
    p.add_argument('--native_res',  type=int, choices=[15,30], default=15,
                   help="native ground sampling distance (m)")
    args = p.parse_args()

    for yr in range(args.year_begin, args.year_end):
        ASTER_Data_Preprocessing(
            year       = yr,
            product    = args.product,
            band_id    = args.band,
            native_res = args.native_res
        )
