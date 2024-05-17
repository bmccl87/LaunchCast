import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import shutil
import pygrib
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=18
plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
#################################################################



print('hello')

hrrr_dir = '/ourdisk/hpc/ai2es/bmac87/HRRR/202206/'
files = os.listdir(hrrr_dir)
for i, file in enumerate(files):
    if i==0:
        f = hrrr_dir+file
        grbs = pygrib.open(f)
        
        for i,grb in enumerate(grbs):
            if i==615:#2m temperature 
                print(i,grb)
                # print(grb.keys())
                # print(grb.units)
                hrrr_lats, hrrr_lons = grb.latlons()

                # print(hrrr_lats)
                # print(hrrr_lons)

                west_extent = -82
                east_extent = -79.5
                south_extent = 27
                north_extent = 29

                hrrr_lats_1d = np.squeeze(hrrr_lats[:,0])
                print(hrrr_lats_1d)
                hrrr_lats_south_idx = np.where(hrrr_lats_1d>=south_extent)[0][0]
                hrrr_lats_north_idx = np.where(hrrr_lats_1d>=north_extent)[0][0]
                print(hrrr_lats_south_idx)
                print(hrrr_lats_north_idx)

                hrrr_lons_1d = hrrr_lons[0,:]

                print(hrrr_lats_1d.shape)
                print(hrrr_lons_1d.shape)

                # temp_2m = grb.values

                

                # print(hrrr_lats.shape)
                
                # mask1 = (hrrr_lats>=south_extent)
                # print(mask1.shape)
                # mask2 = (hrrr_lats<=north_extent)
                # print(mask2.shape)
                # mask3 = np.multiply(mask1,mask2)
                # print(mask3.shape)
                # print(mask3.sum())

                # LC_hrrr_lats = hrrr_lats[[mask3]]
                # print(LC_hrrr_lats.shape)

                # LC_hrrr_lats = hrrr_lats[hrrr_lats>=south_extent]
                # LC_hrrr_lats = LC_hrrr_lats[LC_hrrr_lats<=north_extent]
                # print(LC_hrrr_lats.shape)

                # fig = plt.figure(figsize=(10,8))
                # ax = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree())
                # ax.set_extent([-82,-79.5,27,29],crs=ccrs.PlateCarree())
                # ax.contourf(hrrr_lons,hrrr_lats,temp_2m,transform=ccrs.PlateCarree())
                # ax.add_feature(cfeature.COASTLINE, edgecolor="black")
                # ax.set_title('2m_Temp (K): '+ grb.validDate.strftime("%m/%d/%Y, %H:%M:%S"))
                # plt.savefig('test.png')
            