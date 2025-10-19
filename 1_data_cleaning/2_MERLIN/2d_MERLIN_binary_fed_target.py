import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import pandas as pd
import copy
import time
import glob
import argparse

def extract_slurm_env():
    slurm_vars = {key: os.environ[key] for key in os.environ if key.startswith("SLURM_")}
    if slurm_vars:
        print("SLURM Environment Variables:")
        for key, value in slurm_vars.items():
            print(f"{key}: {value}")
    else:
        print("No SLURM environment variables found.")
    return slurm_vars

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

def binary_fed_target(year = '2022',month = '06'):
    print('generating the final targets')

    #target grid post 64x64 downselection
    #x__target_idxs for slicein: 23:39
    #y__target_idxs for slicin:26:42
    x_target_idxs = [23,39]
    y_target_idxs = [26,42]

    

    data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2c_MERLIN_grid_nc/%s%s/'%(year,month)
    save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2d_MERLIN_binary_fed_target/%s%s/'%(year,month)
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    files = os.listdir(data_dir)
    
    for f,file in enumerate(files):
        print(file)
        ds = xr.open_dataset(data_dir+file,engine='netcdf4')
        ds = ds.swap_dims({"t": "time"})
        if "t" in ds.coords:
            ds = ds.drop_vars("t")
        lon = ds['lon'].values[1:65,1:65]
        lat = ds['lat'].values[1:65,1:65]
        times = ds['time']
        target_lon = lon[y_target_idxs[0]:y_target_idxs[1],x_target_idxs[0]:x_target_idxs[1]]
        target_lat = lat[y_target_idxs[0]:y_target_idxs[1],x_target_idxs[0]:x_target_idxs[1]]

        cc_data = ds['cc'].values.astype(float)[:,1:65,1:65]
        cc_binary = copy.copy(cc_data)
        cc_binary[cc_data>=1.0] = 1.0
        cc_binary[cc_data<1.0] = 0.0
        cc_fed = cc_data/9.0
        
        cc_target = cc_data[:,y_target_idxs[0]:y_target_idxs[1],x_target_idxs[0]:x_target_idxs[1]]
        cc_binary_target = cc_binary[:,y_target_idxs[0]:y_target_idxs[1],x_target_idxs[0]:x_target_idxs[1]]
        cc_fed_target = cc_fed[:,y_target_idxs[0]:y_target_idxs[1],x_target_idxs[0]:x_target_idxs[1]]

        cg_data = ds['cg'].values.astype(float)[:,1:65,1:65]
        cg_binary = copy.copy(cg_data)
        cg_binary[cg_data>=1.0] = 1.0
        cg_binary[cg_data<1.0] = 0.0
        cg_fed = cg_data/9.0

        cg_target = cg_data[:,y_target_idxs[0]:y_target_idxs[1],x_target_idxs[0]:x_target_idxs[1]]
        cg_binary_target = cg_binary[:,y_target_idxs[0]:y_target_idxs[1],x_target_idxs[0]:x_target_idxs[1]]
        cg_fed_target = cg_fed[:,y_target_idxs[0]:y_target_idxs[1],x_target_idxs[0]:x_target_idxs[1]]

        ds2 = xr.Dataset(data_vars = dict(input_cc=(['tt','y','x'],cc_data),
                                            target_cc = (['tt','yy','xx'],cc_target),
                                            target_fed_cc = (['tt','yy','xx'],cc_fed_target),
                                            target_binary_cc = (['tt','yy','xx'],cc_binary_target),
                                            input_cg = (['tt','y','x'],cg_data),
                                            target_cg = (['tt','yy','xx'],cg_target),
                                            target_fed_cg = (['tt','yy','xx'],cg_fed_target),
                                            target_binary_cg = (['tt','yy','xx'],cg_binary_target)),
                            coords=dict(merlin_times=(['tt'],times.data),
                                        input_lon=(['y','x'],lon),
                                        input_lat=(['y','x'],lat),
                                        target_lon=(['yy','xx'],target_lon),
                                        target_lat=(['yy','xx'],target_lat)))

        #saved with the start time of the MERLIN roll out. MERLIN1 in the LC_slices_df
        ds2.to_netcdf(save_dir+file,engine='netcdf4')
        del ds2

def visualize(month='06',year='2022'):
    data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2d_MERLIN_binary_fed_target/%s%s/'%(year,month)
    image_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2d_MERLIN_binary_fed_target/images/%s%s/'%(year,month)
    if os.path.isdir(image_dir)==False:
        os.makedirs(image_dir)
    files = os.listdir(data_dir)
    for f,file in enumerate(files):
        if f>=0:
            ds = xr.open_dataset(data_dir+file,engine='netcdf4')
            ds = ds.swap_dims({"tt": "merlin_times"})
            if "t" in ds.coords:
                ds = ds.drop_vars("t")
            im_count=0
            fig = plt.figure(figsize=(20,40))
            for t,time in enumerate(ds['merlin_times'].values):
                if t==0:
                    ts = pd.Timestamp(time)
                data = ds.sel(merlin_times=time)
                cc_binary = data['target_binary_cc'].values
                cg_binary = data['target_binary_cg'].values
                
                im_count+=1
                ax = fig.add_subplot(4,2,im_count,projection=ccrs.PlateCarree())
                im = ax.pcolormesh(ds['target_lon'].values,ds['target_lat'].values,cc_binary,cmap='binary')
                ax.coastlines()
                ax.set_title(time,fontsize=24)
                
                im_count+=1    
                ax = fig.add_subplot(4,2,im_count,projection=ccrs.PlateCarree())
                im = ax.pcolormesh(ds['target_lon'].values,ds['target_lat'].values,cg_binary,cmap='binary')
                ax.coastlines()
                ax.set_title(time,fontsize=24)
                
            fminute = f"{ts.minute:02}"
            fhour = f"{ts.hour:02}"
            fday = f"{ts.day:02}"
            fmonth = f"{ts.month:02}"
            fyear = f"{ts.year:02}"
            fsave = '%s%s%s%s%s.png'%(fyear,fmonth,fday,fhour,fminute)
            plt.savefig(image_dir+fsave)
            plt.close()

def check_files():
    years = ['2018','2019','2020','2021','2022','2023','2024']
    months = ['03','04','05','06',
            '07','08','09','10','11']

    for year in years:
        for month in months:
            
            load_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2c_MERLIN_grid_nc/%s%s/'%(year,month)
            target_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2d_MERLIN_binary_fed_target/%s%s/'%(year,month)
            files = os.listdir(load_dir)
            bad_files = []
            for file in files:
                if os.path.isfile(load_dir+file)==False:
                    bad_files.append(load_dir+file)
                    print(file)

if __name__=='__main__':

    # slurm_vars = extract_slurm_env()
    # year,month=parse_args()
    # binary_fed_target(year=year,month=month)
    # visualize(year=year,month=month)
    check_files()