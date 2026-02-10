import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import os
import numpy as np
import shutil
from LC_parser import *

def merge_linear_zdr_to_final_xr():

    for year in ['2021','2022','2023','2024']:
        ds_list = []
        for zdr_type in ['low','mid','high']:
            print(year)
            zdr_fname='%s_linear_zdr_%s.nc'%(zdr_type,year)
            print(zdr_fname)
            ds = xr.open_dataset('/scratch/bmac87/%s'%zdr_fname,engine='netcdf4')
            ds_list.append(ds)
            del ds

        data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/6_final_xr/'
        ds = xr.open_dataset(data_dir+'LC_%s.nc'%year,engine='netcdf4')
        ds_list.append(ds)
        del ds

        aligned_ds = xr.align(ds_list[0],ds_list[1],ds_list[2],ds_list[3],join='inner')
        merged_ds = xr.merge(aligned_ds,compat='override')
        merged_ds.to_netcdf(data_dir+'LC_linear_zdr_%s.nc'%year,engine='netcdf4')
        del aligned_ds, merged_ds, ds_list

def aggressive_data_test(args, year='2021'):
    print('aggressive data test')
    hrrr_features = args.hrrr_features
    data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/6_final_xr/'
    ds = xr.open_dataset(data_dir+'LC_linear_zdr_%s.nc'%year,engine='netcdf4')
    lat = ds['lat'].values
    lon = ds['lon'].values
    sample_times = ds['sample_time'].values
    hrrr_data = ds['hrrr_data'].values
    glm_ds = ds[['group_energy','group_area']]
    zdr_ds = ds[['low_zdr','mid_zdr','high_zdr']]
    z_ds = ds[args.Z_keys]
    efm_ds = ds[args.efm_ts_keys]

    for img in range(hrrr_data.shape[0]):
        if img>2:
            plot_data = hrrr_data[img,:,:,:,:]
            fig,axes = plt.subplots(nrows=9,ncols=4,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8.5,11))
            for f in range(9):
                for t in range(4):
                    axes[f,t].pcolormesh(lon,lat,plot_data[t,:,:,f])
                    axes[f,t].coastlines()
                    if t==0:
                        axes[f,t].set_ylabel(hrrr_features[f])
            plt.tight_layout()
            plt.suptitle(sample_times[img])

            save_dir = '/scratch/bmac87/aggressive_data_test/%s/hrrr/'%year
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)
            plt.savefig('%shrrr_%s_%s.png'%(save_dir,year,img))
            plt.close()
            del plot_data

            temp_ds = glm_ds.sel(sample_time=sample_times[img])
            area_data = temp_ds['group_area'].values
            area_data[area_data==0] = np.nan
            energy_data = temp_ds['group_energy'].values
            energy_data[energy_data==0] = np.nan

            fig,axes = plt.subplots(nrows=2,ncols=4,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8.5,5))
            for t in range(4):
                axes[0,t].pcolormesh(lon,lat,area_data[t,:,:])
                axes[0,t].coastlines()
                axes[1,t].pcolormesh(lon,lat,energy_data[t,:,:])
                axes[1,t].coastlines()
            plt.tight_layout()
            plt.suptitle(sample_times[img])
            save_dir = '/scratch/bmac87/aggressive_data_test/%s/glm/'%year
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)
            plt.savefig('%sglm_%s_%s.png'%(save_dir,year,img))
            plt.close()
            del temp_ds, area_data, energy_data

            temp_ds = zdr_ds.sel(sample_time=sample_times[img])
            low_data = temp_ds['low_zdr'].values
            low_data[low_data<=0] = np.nan

            mid_data = temp_ds['mid_zdr'].values
            mid_data[mid_data<=0] = np.nan

            high_data = temp_ds['high_zdr'].values
            high_data[high_data<=0] = np.nan

            fig,axes = plt.subplots(nrows=3,ncols=4,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8.5,7))
            for t in range(4):
                axes[0,t].pcolormesh(lon,lat,low_data[t,:,:])
                axes[0,t].coastlines()
                axes[1,t].pcolormesh(lon,lat,mid_data[t,:,:])
                axes[1,t].coastlines()
                axes[2,t].pcolormesh(lon,lat,high_data[t,:,:])
                axes[2,t].coastlines()
            plt.tight_layout()
            plt.suptitle(sample_times[img])
            save_dir = '/scratch/bmac87/aggressive_data_test/%s/zdr/'%year
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)
            plt.savefig('%szdr_%s_%s.png'%(save_dir,year,img))
            plt.close()
            del low_data, mid_data, high_data, temp_ds

            temp_ds = z_ds.sel(sample_time=sample_times[img])
            fig,axes = plt.subplots(nrows=len(args.Z_keys),ncols=4,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8.5,11))
            for k,key in enumerate(args.Z_keys):
                plot_data = temp_ds[key].values
                plot_data[plot_data<=0] = np.nan
                for t in range(4):
                    axes[k,t].pcolormesh(lon,lat,plot_data[t,:,:])
                    axes[k,t].coastlines()
            plt.suptitle(sample_times[img])
            save_dir = '/scratch/bmac87/aggressive_data_test/%s/isothermal_z/'%year
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)
            plt.savefig('%sisothermal_z_%s_%s.png'%(save_dir,year,img))
            plt.close()
    
            temp_ds = efm_ds.sel(sample_time=sample_times[img])
            fig,axes = plt.subplots(nrows=len(args.efm_ts_keys),ncols=4,figsize=(8.5,44))
            for k,key in enumerate(args.efm_ts_keys):
                plot_data = temp_ds[key].values
                for t in range(4):
                    axes[k,t].plot(plot_data[t,:,0])
            plt.tight_layout()
            plt.suptitle(sample_times[img])
            save_dir = '/scratch/bmac87/aggressive_data_test/%s/efm_median/'%year
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)
            plt.savefig('%smedian_efm_%s_%s.png'%(save_dir,year,img))
            plt.close()
            del plot_data

            fig,axes = plt.subplots(nrows=len(args.efm_ts_keys),ncols=4,figsize=(8.5,44))
            for k,key in enumerate(args.efm_ts_keys):
                plot_data = temp_ds[key].values
                for t in range(4):
                    axes[k,t].plot(plot_data[t,:,3])
            plt.tight_layout()
            plt.suptitle(sample_times[img])
            save_dir = '/scratch/bmac87/aggressive_data_test/%s/efm_std/'%year
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)
            plt.savefig('%sstd_efm_%s_%s.png'%(save_dir,year,img))
            plt.close()
            del temp_ds, plot_data

if __name__=='__main__':
    parser = create_parser()
    args = parser.parse_args()
    exp = args.exp-1
    years = ['2021','2022','2023','2024']
    aggressive_data_test(args=args,year=years[exp])