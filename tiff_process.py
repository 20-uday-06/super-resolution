import time
import os
import numpy as np
from multiprocessing import Pool, cpu_count
from utils import read_tif

def _process_one_tif(args):
    idx, tif_path = args
    LST_K_day, LST_K_night, _, _, _, _ = read_tif(tif_path)
    return idx, LST_K_day, LST_K_night

def tiff_process(data_path):
    start = time.time()

    tifs = [f for f in os.listdir(data_path) if f.endswith('tif')]
    tifs = sorted(tifs)  # optional: ensure consistent order
    tif_paths = [(idx, os.path.join(data_path, tif)) for idx, tif in enumerate(tifs)]

    # Allocate space
    Y_day = np.zeros((len(tifs), 64, 64), dtype='float64')
    Y_night = np.zeros((len(tifs), 64, 64), dtype='float64')

    with Pool(processes=min(cpu_count(), 4)) as pool:
        for idx, day, night in pool.map(_process_one_tif, tif_paths):
            Y_day[idx, :, :] = day
            Y_night[idx, :, :] = night

    end = time.time()
    print(f"Finished processing tif files in {((end-start)/60):.3f} minutes\n")

    return Y_day, Y_night



# python train.py --datapath MODIS\MOD_2021_MOD11A1\tifs_files\1km --model_name MRUNet --lr 0.0001 --epochs 10 --batch_size 32 --continue_train False