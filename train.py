# ===== ASTER 1 km ➜ 250 m MRUNet training
import os
import time
import argparse
import numpy as np
from glob import glob
import rasterio
from rasterio.enums import Resampling

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from model import MRUNet
from dataset import LOADDataset
from utils import get_loss, psnr, ssim, downsampling, upsampling, normalization   # unchanged

# ────────────────────────────────────────────────────────────────────────────────
# 1) ─── Arguments ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MRUNet on ASTER: 1 km ➜ 250 m",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', default='ASTER/ASTER_2021_AST_07XT',  # <‑‑ your ASTER folder
                        help='folder with ASTER GeoTIFFs')
    parser.add_argument('--band', default=2, type=int,
                        help='ASTER band to use (1‑14). 2=NIR, 1=Green')
    parser.add_argument('--lr', default=1e‑3, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--model_name', default='MRUNet_ASTER_1k_250m.pt')
    parser.add_argument('--continue_train', choices=['True','False'], default='False')
    return parser.parse_args()

# ────────────────────────────────────────────────────────────────────────────────
# 2) ─── ASTER preprocessing helpers ────────────────────────────────────────────
def read_aster_band(path, band_id):
    """Read one band (1‑indexed) from an ASTER GeoTIFF, return as np.float32"""
    with rasterio.open(path) as src:
        band = src.read(band_id).astype(np.float32)
    return band

def resample_array(arr, src_res, tgt_res):
    """Quick resampling using rasterio in‑memory dataset"""
    scale = src_res / tgt_res
    height, width = arr.shape
    new_h, new_w = int(height / scale), int(width / scale)
    with rasterio.MemoryFile() as mem:
        profile = {
            "driver":"GTiff","dtype":arr.dtype,"count":1,
            "height":height,"width":width,
            "transform":rasterio.transform.from_origin(0,0,src_res,src_res),
            "crs":None
        }
        with mem.open(**profile) as tmp:
            tmp.write(arr,1)
            data = tmp.read(
                out_shape=(1, new_h, new_w),
                resampling=Resampling.bilinear
            )[0]
    return data

def aster_to_lr_hr(tif_path, band_id, native_res):
    """Return (lr_1km, hr_250m) numpy arrays from an ASTER scene."""
    full = read_aster_band(tif_path, band_id)          # e.g. 15 m native
    hr_250 = resample_array(full, native_res, 250)     # ⬇ to 250 m
    lr_1k  = resample_array(hr_250, 250, 1000)         # ⬇ to 1 km
    return lr_1k, hr_250

# ────────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = MRUNet(res_down=True, n_resblocks=1, bilinear=0).to(device)
    scale  = 4                    # 1 km ➜ 250 m factor

    # 3) ─── Gather all ASTER GeoTIFFs ───────────────────────────────────────────
    tifs  = sorted(glob(os.path.join(args.datapath, '*.tif')))
    if not tifs:
        raise RuntimeError(f"No .tif found in {args.datapath}")

    native_res = 15               # set 15 or 30 depending on product
    band_id    = args.band

    print(f"➜ Building LR/HR pairs from {len(tifs)} ASTER files …")
    lr_imgs, hr_imgs = [], []
    for tif in tqdm(tifs, desc="pre‑proc"):
        lr, hr = aster_to_lr_hr(tif, band_id, native_res)
        if np.count_nonzero(hr) == 0:   # skip empty
            continue
        lr_imgs.append(lr)
        hr_imgs.append(hr)

    # Convert to numpy arrays
    LR = np.array(lr_imgs, dtype=np.float32)
    HR = np.array(hr_imgs, dtype=np.float32)

    # Shuffle / train‑val split
    np.random.seed(1)
    idx = np.random.permutation(len(LR))
    LR, HR = LR[idx], HR[idx]
    split = int(0.75 * len(LR))
    x_train, x_val = LR[:split], LR[split:]
    y_train, y_val = HR[:split], HR[split:]

    # ── Normalisation & bicubic upsample (to 250 m size) just like MODIS code ──
    max_val = np.max(y_train)
    print(f"Normalization factor (max) = {max_val}")

    def prep_pair(lr, hr):
        """Match your original MODIS preprocessing logic."""
        lr_up = normalization(upsampling(lr, scale), max_val)     # bicubic back to 250 m grid
        return lr_up, hr

    # Apply preprocessing
    x_train = np.stack([prep_pair(lr, hr)[0] for lr,hr in zip(x_train,y_train)])
    x_val   = np.stack([prep_pair(lr, hr)[0] for lr,hr in zip(x_val,  y_val)])
    y_train = np.stack([hr for hr in y_train])
    y_val   = np.stack([hr for hr in y_val])

    # Add channel dimension
    x_train = x_train[:,None,:,:]
    x_val   = x_val[:,None,:,:]
    y_train = y_train[:,None,:,:]
    y_val   = y_val[:,None,:,:]

    # 4) ─── Dataloaders ────────────────────────────────────────────────────────
    train_set = LOADDataset(x_train, y_train)
    val_set   = LOADDataset(x_val,   y_val)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size)

    # 5) ─── Optimiser etc. (unchanged) ────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # resume or fresh start ----------------------------------------------------
    start_epoch, best_vloss = 0, np.inf
    if args.continue_train == 'True' and os.path.exists(args.model_name):
        ckpt = torch.load(args.model_name, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch  = ckpt['epoch'] + 1
        best_vloss   = ckpt['losses'][3]
        print(f"→ Resumed from epoch {start_epoch}")

    # 6) ─── Training loop (exactly your old logic) ────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        tr_loss, tr_psnr, tr_ssim = train(model, train_loader, optimizer, train_set, max_val)
        vl_loss, vl_psnr, vl_ssim = validate(model,  val_loader, epoch, val_set, max_val)

        print(f"Train loss {tr_loss:.4f} | Val loss {vl_loss:.4f}")

        # save best model
        if vl_loss < best_vloss:
            best_vloss = vl_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses':[tr_loss,tr_psnr,tr_ssim,vl_loss,vl_psnr,vl_ssim]},
                args.model_name)
            print("✓ checkpoint saved")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
