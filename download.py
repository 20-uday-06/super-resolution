import os
import time
import numpy as np
from pymodis import downmodis
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

def MODIS_Parallel_Downloader(startdate, enddate, product, num_threads, tiles, user="projet3a", password="Projet3AIMT"):
    year = startdate.split("-")[0]
    sensor = product.split(".")[0]

    hdfs_path     = f'MODIS/MOD_{year}_{sensor}/hdfs_files'
    tifs_1km_path = f'MODIS/MOD_{year}_{sensor}/tifs_files/1km'
    tifs_2km_path = f'MODIS/MOD_{year}_{sensor}/tifs_files/2km'
    tifs_4km_path = f'MODIS/MOD_{year}_{sensor}/tifs_files/4km'

    os.makedirs(hdfs_path, exist_ok=True)
    os.makedirs(tifs_1km_path, exist_ok=True)
    os.makedirs(tifs_2km_path, exist_ok=True)
    os.makedirs(tifs_4km_path, exist_ok=True)

    def download_task():
        print(f"🛰️  Start downloading {product} from {startdate} to {enddate}")
        start_time = time.time()
        try:
            modisDown = downmodis.downModis(
                user=user,
                password=password,
                product=product,
                destinationFolder=hdfs_path,
                tiles=tiles,
                today=startdate,
                enddate=enddate
            )
            modisDown.connect()
            modisDown.downloadsAllDay()
        except Exception as e:
            print(f"❌ Download error for {product} ({startdate} to {enddate}): {e}")
        print(f"✅ Finished downloading {product} from {startdate} to {enddate} — time: {time.time() - start_time:.2f}s")

    # Only one task here, so threads are optional — future-proofed for parallel batches
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.submit(download_task).result()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--startdate', type=str, default="2021-01-01", help="Start date in YYYY-MM-DD format")
    parser.add_argument('--enddate', type=str, default="2021-02-02", help="End date in YYYY-MM-DD format")
    parser.add_argument('--username', type=str, default="uday2006")
    parser.add_argument('--password', type=str, default="Udaytyagi2006@")
    args = parser.parse_args()

    products = ["MOD11A1.061", "MOD13A2.061"]  # LST, NDVI 1km
    tiles = "h18v04"  # France = h17v04, h18v04
    num_threads = 4   # Number of threads to use

    for product in products:
        MODIS_Parallel_Downloader(args.startdate, args.enddate, product, num_threads, tiles, args.username, args.password)
