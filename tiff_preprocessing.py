import os
import numpy as np
import time
import rasterio  # Library for handling TIFF images
from rasterio.enums import Resampling
from utils import *
from argparse import ArgumentParser

def process_tiff(tiff_path, output_dir):
    """
    Function to process a TIFF image: resample and save in different resolutions.
    """
    with rasterio.open(tiff_path) as src:
        # Define the scales you want to generate
        scales = {
            "1km": 1,   # original resolution
            "2km": 2,   # half-size
            "4km": 4    # quarter-size
        }

        for res_label, scale in scales.items():
            new_height = src.height // scale
            new_width  = src.width  // scale

            # Skip if we’d end up with a zero-sized image
            if new_height < 1 or new_width < 1:
                print(f"  → Skipping {res_label}, output size would be {new_width}×{new_height}")
                continue

            # Read & resample
            resampled = src.read(
                out_shape=(1, new_height, new_width),
                resampling=Resampling.bilinear
            )

            # Write out the new TIFF
            output_path = os.path.join(output_dir, f"{res_label}_{os.path.basename(tiff_path)}")
            with rasterio.open(
                output_path, 'w',
                driver='GTiff',
                height=new_height,
                width=new_width,
                count=1,
                dtype=resampled.dtype,
                crs=src.crs,
                transform=src.transform
            ) as dst:
                dst.write(resampled)

            print(f"  ✔ Saved {res_label} TIFF: {output_path}")

def MODIS_Data_Preprocessing(year, product, num_threads, tiff_path):
    # Force everything into a local "output/" folder
    output_dir = os.path.abspath("output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing TIFF file: {tiff_path}")
    process_tiff(tiff_path, output_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--year',        type=int,   default=2024)
    parser.add_argument('--product',     type=str,   default="MOD11A2.061")
    parser.add_argument('--tiff_path',   type=str,   required=True)
    parser.add_argument('--num_threads', type=int,   default=4)
    args = parser.parse_args()

    MODIS_Data_Preprocessing(
        args.year,
        args.product,
        args.num_threads,
        args.tiff_path
    )
