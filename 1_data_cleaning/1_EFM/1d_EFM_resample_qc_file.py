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

def get_files_in_all_subdirectories(start_directory):
    """
    Retrieves a list of full paths to all files in the specified directory
    and its subdirectories.

    Args:
        start_directory (str): The path to the top-level directory to start searching from.

    Returns:
        list: A list of strings, where each string is the full path to a file.
    """
    print(start_directory)
    all_files = []
    for root,_,files in os.walk(start_directory):
        for file_name in files:
            full_file_path = os.path.join(root,file_name)
            all_files.append(full_file_path)
    return sorted(all_files)

def check_files(all_files=[]):
    print('all_files',len(all_files))
    rerun_files = []
    for stat in ['mean','median','min','max','std','var']:
        for file in all_files:
            minute = file[-6:-4]
            hour = file[-8:-6]
            day = file[-10:-8]
            month = file[-12:-10]
            year = file[-16:-12]
            fsave = '%s%s%s%s%s_%s.pkl'%(year,month,day,hour,minute,stat)
            save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/EFM_stats_5s/%s/%s%s/'%(stat,year,month)
            if os.path.isfile(save_dir+fsave)==False:
                rerun_files.append(file)
    rerun_files = np.unique(rerun_files)
    print('rerun_files',len(rerun_files))
    return rerun_files

def parse_args():
    print('parsing args')
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',type=int,default=36,help='1-84 int for year month')
    args = parser.parse_args()
    exp = args.exp
    if exp>=1 and exp<=12:
        year='2018'
        month=f"{exp:02}"
    elif exp>=13 and exp<=24:
        year='2019'
        month=f"{exp-12:02}"
    elif exp>=25 and exp<=36:
        year='2020'
        month=f"{exp-24:02}"
    elif exp>=37 and exp<=48:
        year='2021'
        month=f"{exp-36:02}"
    elif exp>=49 and exp<=60:
        year='2022'
        month=f"{exp-48:02}"
    elif exp>=61 and exp<=72:
        year='2023'
        month=f"{exp-60:02}"
    else:
        year='2024'
        month=f"{exp-72:02}"
    return year,month

def open_efm(year='2022',month='06',day='30',hour='22',minute='30',resample_key='5s'):
    day_str = '%s%s%s%s%s'%(year,month,day,hour,minute)
    efm_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/EFM_50HZ_single_file_pkl/%s%s/'%(year,month)
    file = '%s.pkl'%(day_str)
    print('loading the pickle file:',efm_dir+file)
    efm_data = pickle.load(open(efm_dir+file,'rb'))

    print('calculating the statistics')
    efm_ds = make_efm_ds_stats(efm_dict=efm_data, year=year, month=month, day=day,resample_key=resample_key)#has nans, returns [] if no data
    print('statistics calculated successfully')
    return efm_ds

def make_efm_ds_stats(efm_dict={},year='2024',month='06',day='30',hour='22',min='30',resample_key='5s'):
    first_efm = True
    cols = []
    for key in efm_dict:
        FM_str = 'FM%s'%(f"{int(key):02}")
        if efm_dict[key].empty==False:#there is data there
            if first_efm==True:
                temp = pd.DataFrame(efm_dict[key]['ez'])
                temp_resample = temp.resample(resample_key)

                all_mean = temp_resample.mean()
                all_mean = all_mean.rename(columns={"ez":FM_str})

                all_median = temp_resample.median()
                all_median = all_median.rename(columns={'ez':FM_str})

                all_min = temp_resample.min()
                all_min = all_min.rename(columns={'ez':FM_str})

                all_max = temp_resample.max()
                all_max = all_max.rename(columns={'ez':FM_str})

                all_std = temp_resample.std()
                all_std = all_std.rename(columns={'ez':FM_str})

                all_var = temp_resample.var()
                all_var = all_var.rename(columns={'ez':FM_str})
                first_efm=False
            else:
                temp = pd.DataFrame(efm_dict[key]['ez'])
                temp = temp.rename(columns={'ez':FM_str})
                temp_resample = temp.resample(resample_key)
                all_median = all_median.join(temp_resample.median(),how='outer',rsuffix=FM_str)
                all_mean = all_mean.join(temp_resample.mean(), how='outer',rsuffix=FM_str)
                all_min = all_min.join(temp_resample.min(), how='outer',rsuffix=FM_str)
                all_max = all_max.join(temp_resample.max(),how='outer',rsuffix=FM_str)
                all_std = all_std.join(temp_resample.std(),how='outer',rsuffix=FM_str)
                all_var = all_var.join(temp_resample.var(),how='outer',rsuffix=FM_str)
            del temp, temp_resample
    
    ds = xr.Dataset(data_vars = {
                                    "median":(("time","efms"),all_median.values),
                                    "mean":(("time","efms"),all_mean.values),
                                    "min":(("time","efms"),all_min.values),
                                    "max":(("time","efms"),all_max.values),
                                    "std":(("time","efms"),all_std.values),
                                    "var":(("time","efms"),all_var.values)
                                    },
                    coords={
                            "time": ("time",all_var.index),
                            "efm_sites": ("efms",all_var.columns)
                            })
    return ds

def calc_efm_distances():
    loc_df = pd.read_excel('EFM_Locations.xlsx')
    loc_df = loc_df.loc[loc_df['IsActive']==True]

    site_names = loc_df['SiteName'].values
    site_lats = loc_df['Latitude'].values
    site_lons = loc_df['Longitude'].values

    distances = np.zeros((len(site_names),len(site_names)))
    for i in range(len(site_names)):
        efm_lat1 = site_lats[i]
        efm_lon1 = site_lons[i]
        for j in range(len(site_names)):
            efm_lat2 = site_lats[j]
            efm_lon2 = site_lons[j]
            lons = np.asarray([efm_lon1,efm_lon2])
            lats = np.asarray([efm_lat1,efm_lat2])
            dx,dy = mpcalc.lat_lon_grid_deltas(longitude=lons,latitude=lats)
            dist = np.sqrt(dx[0]**2 + dy[0]**2)
            distances[i,j]=dist[0].magnitude
    df3 = pd.DataFrame(distances,index=site_names,columns=site_names)
    return df3

def qc_dist_efm(efm_ds=xr.Dataset(),efm_distances=pd.DataFrame(),year='2022',month='06',day='30',hour='22',minute='30',resample_key='5s'):

    print('in the qc_dist_efm function')
    efm_site_names = efm_ds['efm_sites'].values
    valid_times = efm_ds['time'].values
    for key in efm_ds:#these are the specific statistics
        print(key)
        save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/EFM_stats_%s/%s/%s%s/'%(resample_key,key,year,month)
        if os.path.isdir(save_dir)==False:
            os.makedirs(save_dir)

        stat_np = efm_ds[key].values
        stat_df = pd.DataFrame(stat_np,index=valid_times,columns=efm_site_names)
        del stat_np

        for site_name in efm_site_names:
            print('distance qc-ing ',key,site_name)

            if site_name in efm_distances.columns:
                #get the specific site statistics
                single_efm_stat_np = stat_df[site_name].values
                sorted_efms = efm_distances[site_name].sort_values(ascending=True).index.values
                if np.sum(np.isnan(single_efm_stat_np))>0:
                    print('replacing:',np.sum(np.isnan(single_efm_stat_np)),' for efm site ',site_name)
                    for close_efm in sorted_efms:
                        if close_efm in efm_site_names:
                            close_efm_np = stat_df[close_efm].values
                            if np.sum(np.isnan(single_efm_stat_np))>0:
                                single_efm_stat_np[np.isnan(single_efm_stat_np)] = close_efm_np[np.isnan(single_efm_stat_np)]
                            del close_efm_np

                #store back into the dataframe 
                stat_df[site_name] = single_efm_stat_np
                del single_efm_stat_np
            else:
                stat_df = stat_df.drop(columns=[site_name])
                print(site_name)
                
        fsave = '%s%s%s%s%s_%s.pkl'%(year,month,day,hour,minute,key)
        print('saving %s%s'%(save_dir,fsave))
        stat_df.to_pickle(save_dir+fsave)
        print('saved successfully')
        del stat_df 

def visualize(year='2022',month='06',stat='median'):

    efm_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/EFM_stats_5s/%s/%s%s/'%(stat,year,month)
    save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/EFM_stats_5s/images/%s/%s%s/'%(stat,year,month)
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)

    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    if stat=='median':
        vmin = -5000
        vmax = 5000
        cmap = 'viridis'

    if stat=='mean':
        vmin = -5000
        vmax = 5000
        cmap='viridis'

    if stat=='min':
        vmin = -5000
        vmax = 500
        cmap = 'viridis'

    if stat=='max':
        vmin = -5000
        vmax=5000
        cmap = 'viridis'

    if stat=='std':
        vmin = 0
        vmax = 50
        cmap = 'viridis'

    if stat=='var':
        vmin = 0
        vmax = 4000
        cmap = 'viridis'

    files = sorted(os.listdir(efm_dir))
    for f,file in enumerate(files):
        if f>=0:
            print(file)
            df = pickle.load(open(efm_dir+file,'rb'))
            year = file[0:4]
            month = file[4:6]
            day = file[6:8]
            hour = file[8:10]
            minute = file[10:12]
            mo_title = months[int(month)-1]
            title_str = '%s%s UTC %s %s %s'%(hour,minute,day,mo_title,year)
            fsave = '%s%s%s%s%s_%s.png'%(year,month,day,hour,minute,stat)
            print(title_str,fsave)

            columns = df.columns
            data_np = []
            for col in columns:
                data_np.append(df[col].values)
            data_np = np.stack(data_np,axis=0)
            x_tick_labels = []
            x_ticks = np.arange(0,data_np.shape[1])[::25]
            for x_tick in x_ticks:
                x_tick_labels.append(f"{x_tick:02}")
            fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(10,10))
            im = axes.pcolormesh(data_np,cmap=cmap,vmin=vmin,vmax=vmax)
            cb = plt.colorbar(im,ax=axes,label = "Potential Gradient (V/m)")
            axes.set_xlabel('Time (5 sec intervals) since (%s)'%(title_str),fontsize=18)
            axes.set_ylabel('EFM Sites',fontsize=18)
            axes.set_yticks(np.arange(0,len(columns)),columns,fontsize=18)
            axes.set_xticks(x_ticks,x_tick_labels,fontsize=18,rotation=45)
            axes.grid(axis='y')
            axes.set_title(stat,fontsize=24)
            plt.savefig('%s%s'%(save_dir,fsave))
            plt.close()

def run_qc():
    
    start_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/EFM_50HZ_single_file_pkl/'
    all_files = get_files_in_all_subdirectories(start_directory=start_dir)
    start_idx, end_idx = parse_args()
    rerun_files = check_files(all_files=all_files)
    files2qc = rerun_files
    print(rerun_files)
    efm_distances = calc_efm_distances()
    resample_key = '5s'

    for f,file in enumerate(files2qc):
        if f>=0:
            print(file)
            minute = file[-6:-4]
            hour = file[-8:-6]
            day = file[-10:-8]
            month = file[-12:-10]
            year = file[-16:-12]

            try:
                efm_ds = open_efm(year=year,month=month,day=day,hour=hour,minute=minute,resample_key=resample_key)
                qc_dist_efm(efm_ds=efm_ds,efm_distances=efm_distances,year=year,month=month,day=day,hour=hour,minute=minute,resample_key=resample_key)
            except Exception as e:
                print('bad file')

if __name__=='__main__':

    start_time = time.time()
    slurm_vars = extract_slurm_env()
    year,month = parse_args()
    stats = ['median','min','max','std']
    for stat in stats:
        visualize(year=year,month=month,stat=stat)
    end_time = time.time()
    print('the number of minutes it took to run:')
    print((end_time-start_time)/60)