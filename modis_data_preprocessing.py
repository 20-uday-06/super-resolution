import os
import numpy as np
import time
from utils import *
from argparse import ArgumentParser

def MODIS_Data_Preprocessing(year, product, num_threads):
    sensor = product.split(".")[0]
    root_dir = f'MODIS/MOD_{year}_{sensor}'
    hdfs_path = os.path.join(root_dir, 'hdfs_files')
    tifs_1km_path = os.path.join(root_dir, 'tifs_files/1km')
    tifs_2km_path = os.path.join(root_dir, 'tifs_files/2km')
    tifs_4km_path = os.path.join(root_dir, 'tifs_files/4km')

    os.makedirs(hdfs_path, exist_ok=True)
    os.makedirs(tifs_1km_path, exist_ok=True)
    os.makedirs(tifs_2km_path, exist_ok=True)
    os.makedirs(tifs_4km_path, exist_ok=True)

    print(f"🔧 Starting preprocessing in {hdfs_path}")
    hdfs = sorted([f for f in os.listdir(hdfs_path) if f.endswith('.hdf')])
    start_time = time.time()

    for hdf in hdfs:
        hdf_path = os.path.join(hdfs_path, hdf)
        try:
            if sensor == 'MOD11A1':
                print(f"🛠️  Processing {hdf} (LST)")
                crop_modis(hdf_path, hdf, tifs_1km_path, tifs_2km_path, tifs_4km_path, 64, (64, 64))
            # elif sensor == 'MOD13A2':
            #     print(f"🛠️  Processing {hdf} (NDVI)")
            #     crop_modis_MOD13A2(hdf_path, hdf, tifs_1km_path, tifs_2km_path, tifs_4km_path, 64, (64, 64))
            print(f"✅ Done with {hdf}")
        except Exception as e:
            print(f"❌ Failed to process {hdf} — {type(e).__name__}: {e}")

    print(f"✅ Finished processing {product} — Time taken: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--year_begin', type=int, default=2021, help="Start year (inclusive)")
    parser.add_argument('--year_end', type=int, default=2022, help="End year (exclusive)")
    args = parser.parse_args()

    years = list(np.arange(args.year_begin, args.year_end))
    products = ["MOD11A1.061", "MOD13A2.061"]  # Matches downloader structure
    num_threads = 4  # Parallel threads if needed later

    for year in years:
        for product in products:
            MODIS_Data_Preprocessing(year, product, num_threads)
