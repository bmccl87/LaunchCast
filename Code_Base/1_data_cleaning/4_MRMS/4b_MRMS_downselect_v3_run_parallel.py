import sys
print(sys.executable)

import xarray as xr
import numpy as np
import os
import shutil
import pygrib
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xesmf as xe
import argparse
import time
import pandas as pd
import multiprocessing as mp

EFM_MASK_VALUE=-999999.0
ZDR_MASK_VALUE=-99
ZDR_FILL_VALUE=-5

_GLOBALS = {}

def create_parser():
    parser = argparse.ArgumentParser(description='Download', fromfile_prefix_chars='@')
    parser.add_argument('--exp', type=int, default=0,help='0-998')
    args = parser.parse_args()
    exp = args.exp
    start_idx = exp*123
    end_idx = start_idx+123
    return start_idx, end_idx, exp

def init_worker():
    """
    Runs once per worker process.
    Loads shared metadata and builds/reuses the regridder once per process.
    """
    idx_dict = pickle.load(open("idx_dict.pkl", "rb"))

    mrms_cropped_dict = pickle.load(open("mrms_cropped_grid.pkl", "rb"))
    hrrr_cropped_dict = pickle.load(open("hrrr_grid_cropped.pkl", "rb"))

    mrms_cropped_lon = mrms_cropped_dict["cropped_lon"]
    mrms_cropped_lat = mrms_cropped_dict["cropped_lat"]

    hrrr_cropped_lon = hrrr_cropped_dict["cropped_lon"]
    hrrr_cropped_lat = hrrr_cropped_dict["cropped_lat"]

    source_grid_mrms = {
        "lon": np.ascontiguousarray(mrms_cropped_lon),
        "lat": np.ascontiguousarray(mrms_cropped_lat),
    }

    target_grid_hrrr = {
        "lon": np.ascontiguousarray(hrrr_cropped_lon),
        "lat": np.ascontiguousarray(hrrr_cropped_lat),
    }

    regridder = xe.Regridder(
        source_grid_mrms,
        target_grid_hrrr,
        method="bilinear",
        reuse_weights=True,
        filename="regridder.nc",
    )

    _GLOBALS.update(
        y_min=idx_dict["y_min"],
        y_max=idx_dict["y_max"],
        x_min=idx_dict["x_min"],
        x_max=idx_dict["x_max"],
        mrms_cropped_lon=mrms_cropped_lon,
        mrms_cropped_lat=mrms_cropped_lat,
        hrrr_cropped_lon=hrrr_cropped_lon,
        hrrr_cropped_lat=hrrr_cropped_lat,
        source_grid_mrms=source_grid_mrms,
        target_grid_hrrr=target_grid_hrrr,
        regridder=regridder,
    )

def process_one(row_dict):
    """
    Runs once per file.
    row_dict should contain at least:
        file
        variable
    """

    g = _GLOBALS

    try:
        execute_task(
            file=row_dict,
            mrms_cropped_lon=g["mrms_cropped_lon"],
            mrms_cropped_lat=g["mrms_cropped_lat"],
            hrrr_cropped_lon=g["hrrr_cropped_lon"],
            hrrr_cropped_lat=g["hrrr_cropped_lat"],
            y_min=g["y_min"],
            y_max=g["y_max"],
            x_min=g["x_min"],
            x_max=g["x_max"],
            source_grid_mrms=g["source_grid_mrms"],
            target_grid_hrrr=g["target_grid_hrrr"],
            regridder=g["regridder"],
        )

        return {
            "status": "success",
            "file": row_dict["fname"],
            "variable": row_dict["variable"],
            "error": None,
        }

    except Exception as e:
        return {
            "status": "failed",
            "file": row_dict.get("fname", None),
            "variable": row_dict.get("variable", None),
            "error": repr(e),
        }

def execute_task(file={},
                    mrms_cropped_lon=[],
                    mrms_cropped_lat=[],
                    hrrr_cropped_lon=[],
                    hrrr_cropped_lat=[],
                    source_grid_mrms={},
                    target_grid_hrrr={},
                    regridder=(),
                    y_min=0,
                    y_max=1,
                    x_min=0,
                    x_max=1):
    try:
        load_file = file['fname_dir']
        save_dir = file['save_dir']
        file_str = file['fsave'][0:-2]+'pkl'
        radar_time = file['datetime']
        print(save_dir)
        print(file_str)
        num_pixels_outside_LC = 128
        mrms_radar_data = pygrib.open(load_file)[1].values[y_min:y_max, x_min:x_max]
        hrrr_radar_data = regridder(np.ascontiguousarray(mrms_radar_data))
        hrrr_radar_data_LC = hrrr_radar_data[num_pixels_outside_LC:-num_pixels_outside_LC,num_pixels_outside_LC:-num_pixels_outside_LC]
        pickle.dump({'data':hrrr_radar_data_LC,'time':radar_time},open(save_dir+file_str,'wb'))
        del hrrr_radar_data_LC, hrrr_radar_data, mrms_radar_data, save_dir, file_str
        print('pickle file saved successfully')

    except Exception as e:
        print(e)


def run_files(files_df=pd.DataFrame(),n_workers=4,chunksize=5):

    if files_df.empty:
        print("files_df is empty. Nothing to process.")
        return pd.DataFrame()

    rows = files_df.to_dict("records")

    print(f"Processing {len(rows)} files with {n_workers} workers")

    results = []

    with mp.Pool(processes=n_workers,initializer=init_worker,) as pool:
        for n, result in enumerate(pool.imap_unordered(process_one, rows, chunksize=chunksize),start=1,):
            results.append(result)
            if result["status"] == "failed":
                print('FAILED')
                print(result)
                # print(
                #     f"FAILED: {result['fname']} "
                #     f"{result['variable']} "
                #     f"{result['error']}"
                # )

            if n % 100 == 0 or n == len(rows):
                print(f"Finished {n}/{len(rows)}")

    results_df = pd.DataFrame(results)

    n_failed = (results_df["status"] == "failed").sum()
    n_success = (results_df["status"] == "success").sum()

    print(f"Done. Success: {n_success}, Failed: {n_failed}")

    return results_df

def main():
    
    files_df = pickle.load(open('/scratch/bmac87/MRMS_target_mask_files_needed.pkl','rb'))
    n_files = len(files_df)
    print(n_files)
    print(files_df.columns)
    print(files_df['save_dir'])
    print(files_df['fsave'])

    start_idx, end_idx,exp = create_parser()
    batch_df = files_df.iloc[start_idx:end_idx].copy()
    nw = int(os.environ.get("SLURM_CPUS_PER_TASK"))
    print(nw, 'num_cpus')
    results_df = run_files(batch_df,
                        n_workers=nw,
                        chunksize=5)
    

if __name__=='__main__':
    start_clock = time.time()
    main()
    end_clock = time.time()
    run_time_mins = (end_clock-start_clock)/60
    print('run_time_mins:',run_time_mins)