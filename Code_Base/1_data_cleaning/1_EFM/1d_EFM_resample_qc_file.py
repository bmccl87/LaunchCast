import xarray as xr
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import metpy.calc as mpcalc
import argparse 
import sys
from helper import *
import time

from multiprocessing import Pool

def process_file(file):
    minute = file[-6:-4]
    hour = file[-8:-6]
    day = file[-10:-8]
    month = file[-12:-10]
    year = file[-16:-12]
    print(year,month,day,hour,minute)
    try:
        open_efm(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            file=file,
            resample_key='30s'
        )
    except Exception as e:
        print('bad file', file)
        print(e)

def extract_slurm_env():
    slurm_vars = {key: os.environ[key] for key in os.environ if key.startswith("SLURM_")}
    if slurm_vars:
        print("SLURM Environment Variables:")
        for key, value in slurm_vars.items():
            print(f"{key}: {value}")
        return slurm_vars
    else:
        print("No SLURM environment variables found.")
        return {}

def check_files(all_files=[]):
    print('all_files',len(all_files))
    rerun_files = []
    for f,file in enumerate(all_files):
        minute = file[-6:-4]
        hour = file[-8:-6]
        day = file[-10:-8]
        month = file[-12:-10]
        year = file[-16:-12]
        fsave = '%s%s%s%s%s.parquet'%(year,month,day,hour,minute)
        print(fsave)
        save_dir = '/scratch/bmac87/efm_v2/%s/%s%s/'%(year,year,month)
        if os.path.isfile(save_dir+fsave)==False:
            rerun_files.append(file)
    rerun_files = np.unique(rerun_files)
    return rerun_files

def get_all_efm_files(base_dir, extensions=(".pkl")):
    from pathlib import Path
    base_dir = Path(base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    files = sorted(
        str(f) for f in base_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    )

    return files

def parse_args():
    print('parsing args')
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',type=int,default=36,help='1-84 int for year month')
    args=parser.parse_args()
    exp = args.exp
    return exp

def open_efm(year='2022',month='06',day='30',hour='22',minute='30',resample_key='30s',file=''):
    
    save_dir = '/scratch/bmac87/efm_v2/pkl/%s/%s%s/'%(year,year,month)
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    fbase='%s%s%s%s%s'%(year,month,day,hour,minute)

    if os.path.isfile('%s%s.parquet'%(save_dir,fbase))==True:
        return

    efm_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/1b_EFM_50HZ_single_file_pkl/%s%s/'%(year,month)
    print('loading the pickle file:',efm_dir+fbase+'.pkl')
    efm_dict = pickle.load(open(efm_dir+'%s.pkl'%fbase,'rb'))

    print('calculating the statistics')
    efm_df = compute_efm_resampled_stats(efm_dict,
                resample_key="30s",
                native_freq="20ms",
                min_count=500,
                min_frac=0.3,
                threshold_gt=1000.0,
                round_to_native=True)
    print('statistics calculated successfully')
    print('saving the parquet file')
    efm_df.columns.names = ["station", "stat"]
    efm_df.to_pickle('%s%s.pkl'%(save_dir,fbase))
    print('pickle file saved')

def compute_efm_resampled_stats(efm_dict,
    resample_key="30s",
    native_freq="20ms",
    min_count=1000,
    min_frac=0.7,
    threshold_gt=1000.0,
    round_to_native=True):

    """
    Build a common time grid, reindex each EFM station to that grid,
    resample, and compute summary statistics.

    Parameters
    ----------
    efm_dict : dict
        Dictionary of per-station pandas DataFrames.
        Expected structure:
            efm_dict[key] -> DataFrame with:
                - DatetimeIndex
                - column named 'ez'
    resample_key : str
        Pandas resample frequency, e.g. '30S', '1min', '10S'
    native_freq : str
        Native sampling interval, e.g. '20ms' for 0.02 s
    min_count : int
        Minimum number of valid samples required in each resampled bin
    min_frac : float
        Minimum fraction of valid samples required in each resampled bin
    threshold_gt : float
        Threshold for count of values greater than threshold_gt
    round_to_native : bool
        If True, round timestamps to native_freq before reindexing

    Returns
    -------
    results : dict
        Dictionary containing:
            - all_mean
            - all_median
            - all_min
            - all_max
            - all_std
            - all_var
            - all_rms
            - all_max_abs
            - all_gt_thresh
            - count
            - size
            - frac_valid
            - mask
            - efm_wide
            - stats
    """

    def rms(x):
        arr = np.asarray(x, dtype=float)
        return np.sqrt(np.nanmean(arr ** 2))

    def max_abs(x):
        arr = np.asarray(x, dtype=float)
        if np.all(np.isnan(arr)):
            return np.nan
        return np.nanmax(np.abs(arr))

    def count_gt_thresh(x):
        arr = np.asarray(x, dtype=float)
        return np.sum(np.abs(arr) > threshold_gt)

    # -----------------------------
    # 1. Collect cleaned series
    # -----------------------------
    series_dict = {}

    for key, df in efm_dict.items():
        if df is None or df.empty:
            continue
        if "ez" not in df.columns:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"Station {key} does not have a DatetimeIndex.")

        fm_str = f"FM{int(key):02}"

        s = df["ez"].copy()
        s = s.sort_index()

        if round_to_native:
            s.index = s.index.round(native_freq)

        # Drop duplicate timestamps after rounding
        s = s[~s.index.duplicated(keep="first")]

        s.name = fm_str
        series_dict[fm_str] = s

    if len(series_dict) == 0:
        raise ValueError("No valid EFM station data found in efm_dict.")

    # -----------------------------
    # 2. Build common native grid
    # -----------------------------
    start_time = min(s.index.min() for s in series_dict.values())
    end_time = max(s.index.max() for s in series_dict.values())

    common_index = pd.date_range(start=start_time, end=end_time, freq=native_freq)

    # -----------------------------
    # 3. Reindex each station
    # -----------------------------
    reindexed_series = []
    for fm_str, s in series_dict.items():
        s_reindexed = s.reindex(common_index)
        s_reindexed.name = fm_str
        reindexed_series.append(s_reindexed)

    # Wide dataframe: rows=time, cols=stations
    efm_wide = pd.concat(reindexed_series, axis=1)
    efm_wide = efm_wide.sort_index()

    # -----------------------------
    # 4. Resample once
    # -----------------------------
    resampled = efm_wide.resample(resample_key)

    # Valid counts per bin/per station
    count = resampled.count()          # non-NaN count, per station
    size = resampled.size()            # total rows in each bin
    frac_valid = count.div(size, axis=0)

    # Mask per bin/per station
    mask = (count >= min_count) & (frac_valid >= min_frac)

    # -----------------------------
    # 5. Compute stats
    # -----------------------------
    stats = resampled.agg([
        "mean",
        "median",
        "min",
        "max",
        "std",
        "var",
        rms,
        max_abs,
        count_gt_thresh,
    ])
    
    # Convert mask columns to match MultiIndex structure
    mask_df = mask.astype(np.float16).copy()

    # Create MultiIndex columns: (station, "mask")
    mask_df.columns = pd.MultiIndex.from_product(
        [mask_df.columns, ["mask"]]
    )

    # Combine with stats
    stats_with_mask = pd.concat([stats, mask_df], axis=1)

    # Optional: sort columns nicely
    stats_with_mask = stats_with_mask.sort_index(axis=1)
    return stats_with_mask

def visualize(year='2022',month='06',stat='median'):
    efm_dir='/scratch/bmac87/efm_v2/%s/%s%s/'%(year,year,month)
    files = os.listdir(efm_dir)
    print(files)
    print(len(files))
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(efm_dir + files[0])
    table = pf.read()
    df = table.to_pandas()
    print(df.head())

def execute_exp():

    exp=parse_args()
    print(exp)

    # base_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/1b_EFM_50HZ_single_file_pkl/'
    # all_files = pickle.load(open('all_efm_files_w_dir.pkl','rb'))
    # print(len(all_files))
    
    files_2_run = pickle.load(open('/scratch/bmac87/efm_file_lists/exp_%s.pkl'%(exp),'rb'))

    start_clock = time.time()
    slurm_vars = extract_slurm_env()
    num_cpus = int(slurm_vars['SLURM_CPUS_PER_TASK'])
    print('NUM_CPUS')
    print(num_cpus)
    with Pool(processes=num_cpus) as p:
        p.map(process_file, files_2_run)
    end_clock = time.time()
    print('time to run')
    print((end_clock-start_clock)/60.)
    
    # #check if the files saved
    # for file in all_files:
    #     minute = file[-6:-4]
    #     hour = file[-8:-6]
    #     day = file[-10:-8]
    #     month = file[-12:-10]
    #     year = file[-16:-12]
    #     save_dir = '/scratch/bmac87/efm_v2/%s/%s%s/'%(year,year,month)
    #     fbase='%s%s%s%s%s'%(year,month,day,hour,minute)
    #     print(fbase)
    #     if os.path.isfile('%s%s.parquet'%(save_dir,fbase))==False:
    #         missing_files.append(file)

    # pickle.dump(missing_files,open('./missing_efm_files.pkl','wb'))
    print('END OF 1d_EFM_RESAMPLE_QC_FILE.PY')
    print('CONGRATULATIONS')

if __name__=='__main__':
    execute_exp()
