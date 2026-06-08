import os
import shutil
import xarray as xr
import matplotlib.pyplot as plt
import sys
import argparse
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def main():

    data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2a_MERLIN_HRRR_grid_Jun302022/'
    files = sorted(os.listdir(data_dir))

    for file in files:
        print(file)

        ds = xr.open_dataset(data_dir+file,engine='netcdf4')
        cc_fed = ds['cc'].values.astype(float)
        cg_fed = ds['cg'].values.astype(float) 
        lon = ds['lon'].values
        lat = ds['lat'].values

        x_idxs = [1422,1486]
        y_idxs = [176,240]

        ksc_idxs = [y_idxs[0],y_idxs[1],x_idxs[0],x_idxs[1]]
        cc_fed = cc_fed[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        cg_fed = cg_fed[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        lon = lon[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
        lat = lat[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]

        fig = plt.figure()
        ax = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
        cc_fed[cc_fed<1]=np.nan
        cc_cb = ax.pcolormesh(lon,lat,cc_fed,cmap='Reds')
        ax.coastlines()
        ax.set_title('In/Intra/Inter-Cloud Flashes')

        ax1 = fig.add_subplot(1,2,2,projection=ccrs.PlateCarree())
        cg_fed[cg_fed<1]=np.nan
        cg_cb = ax1.pcolormesh(lon,lat,cg_fed,cmap='Reds')
        ax1.coastlines()
        ax1.set_title('Cloud-to-Ground Flashes')

        fsave = file[-13:-3]+'.png'
        print(fsave)
        plt.savefig('./MERLIN_test/%s'%(fsave))
        plt.close()

if __name__=='__main__':
    main()