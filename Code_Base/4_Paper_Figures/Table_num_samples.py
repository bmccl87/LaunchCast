import tensorflow as tf
import numpy as np
import xarray as xr
from Figure_1_MERLIN_Climo import grid_flashes
import pygrib
import cartopy.crs as ccrs
import pickle
import pandas as pd
def number_of_samples():

    print('generating the numbers for the table')
    for year in ['2021','2022','2023','2024']:
        print(year)
        ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3_data_processing/LC_Forecast_Final_log1p_GLM_VII_%s.nc'%(year),engine='netcdf4')
        sample_time = ds['sample_time'].values
        cc = ds['cc'].values[:,:,24:40,24:40]
        num_cc = np.sum(cc)
        cg = ds['cg'].values[:,:,24:40,24:40]
        num_cg = np.sum(cg)
        print('cc',num_cc)
        print('cg',num_cg)
        print('#',len(sample_time))

def onset_decay_count():

    pkl_dir = '/home/bmac87/LaunchCast/1_data_cleaning/5_Merge/'
    for year in ['2021','2022','2023','2024']:
        ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3_data_processing/LC_Forecast_Final_log1p_GLM_VII_%s.nc'%(year),engine='netcdf4')
        sample_time = ds['sample_time'].values

        pkl_file = '%s_samples_w_ltg_counts_w_efm_flag.pkl'%(year)
        df = pickle.load(open(pkl_dir+pkl_file,'rb'))
        df_5nm = df.loc[df['withinEFM_5nm']==1.0].copy()
        print('total samples within 5nm')
        print(len(df_5nm))
        glm_columns = ['G60245_num','G45230_num','G30215_num','G15200_num']
        first_flash_5nm = df_5nm.loc[(df_5nm[glm_columns] == 0.0).all(axis=1)]
        first_flash_times = pd.Index(first_flash_5nm['R60245']).intersection(sample_time).tolist()
        
        target_columns = ['M00215_num_total','M15230_num_total','M30245_num_total','M45260_num_total']
        last_flash_5nm = df_5nm.loc[(df_5nm[target_columns]==0.0).any(axis=1)]
        
        print(year)
        print('first 5nm',len(first_flash_times))

        last_flash_5nm1 = df_5nm.loc[(df_5nm[target_columns[-1]]==0.0)]
        last1 = last_flash_5nm1['R60245'].values
        last_flash_5nm2 = df_5nm.loc[(df_5nm[target_columns[2:4]]==0.0).all(axis=1)]
        last2 = last_flash_5nm2['R60245'].values
        last_flash_5nm3 = df_5nm.loc[(df_5nm[target_columns[1:4]]==0.0).all(axis=1)]
        last3 = last_flash_5nm3['R60245'].values
        last_single=pd.Index(sample_time).intersection(last1).intersection(last2).intersection(last3).tolist()
        print('last 5nm',len(last_single))

if __name__=='__main__':
    # number_of_samples()
    onset_decay_count()