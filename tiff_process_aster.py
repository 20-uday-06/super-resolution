# Build LR‑(1 km) / HR‑(250 m) stacks from ASTER GeoTIFFs
# ----------------------------------------------------------
import os, time
import numpy as np
from multiprocessing import Pool, cpu_count
import rasterio
from rasterio.enums import Resampling

# ── helper to read a single band and create LR / HR pairs ────────────────────
def _process_one_tif(args):
    idx, tif_path, band_id, native_res = args
    with rasterio.open(tif_path) as src:
        full = src.read(band_id).astype(np.float32)          # native 15 m or 30 m

        # ↓↓↓  HR: resample to 250 m  ↓↓↓
        scale_hr = native_res / 250
        new_h_hr = int(full.shape[0] / scale_hr)
        new_w_hr = int(full.shape[1] / scale_hr)
        hr_250 = src.read(
            band_id,
            out_shape=(new_h_hr, new_w_hr),
            resampling=Resampling.bilinear
        ).astype(np.float32)

        # ↓↓↓  LR: resample HR to 1 km  ↓↓↓
        scale_lr = 250 / 1000
        new_h_lr = int(hr_250.shape[0] * scale_lr)
        new_w_lr = int(hr_250.shape[1] * scale_lr)
        lr_1k = rasterio.warp.reproject(
            source=hr_250,
            destination=np.empty((new_h_lr, new_w_lr), dtype=np.float32),
            src_transform=src.transform * src.transform.scale(scale_hr, scale_hr),
            src_crs=src.crs,
            dst_transform=src.transform * src.transform.scale(scale_hr/scale_lr,
                                                              scale_hr/scale_lr),
            dst_crs=src.crs,
            resampling=Resampling.bilinear
        )[0]
    return idx, lr_1k, hr_250   # <‑‑ both have the same *pixel* count ratio (1:4)

# ── main API ─────────────────────────────────────────────────────────────────
def tiff_process(data_path, band=2, native_res=15):
    """
    Build numpy stacks from all .tif files under `data_path`.
    Parameters
    ----------
    data_path  : folder containing ASTER GeoTIFFs.
    band       : ASTER band # to read (1‑14). Default = 2 (NIR).
    native_res : 15 or 30 (m). Use 30 for SWIR products.

    Returns
    -------
    LR_stack : np.ndarray [N, H_lr, W_lr]   -> simulated 1 km
    HR_stack : np.ndarray [N, H_hr, W_hr]   -> resampled 250 m
    """
    start = time.time()

    tifs = sorted([f for f in os.listdir(data_path) if f.lower().endswith('.tif')])
    tasks = [(idx, os.path.join(data_path, tif), band, native_res)
             for idx, tif in enumerate(tifs)]

    # allocate ragged list first (height/width differ per scene)
    LR_list, HR_list = [None]*len(tifs), [None]*len(tifs)

    with Pool(processes=min(cpu_count(), 4)) as pool:
        for idx, lr, hr in pool.map(_process_one_tif, tasks):
            LR_list[idx] = lr
            HR_list[idx] = hr

    # drop any scenes that came back empty
    LR_stack = np.array([lr for lr in LR_list if lr is not None])
    HR_stack = np.array([hr for hr in HR_list if hr is not None])

    end = time.time()
    print(f"Finished processing {len(LR_stack)} ASTER files "
          f"in {(end-start)/60:.2f} min.\n")

    return LR_stack, HR_stack
