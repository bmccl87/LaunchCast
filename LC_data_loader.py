import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import shutil
import pygrib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy

#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=18
plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
#################################################################

hrrr_dir = '/ourdisk/hpc/ai2es/bmac87/HRRR/202206/'
files = os.listdir(hrrr_dir)

for i, file in enumerate(files):
    if i==0:
        fig = plt.figure(figsize=(10,8))
        f = hrrr_dir+file
        grbs = pygrib.open(f)

        for i,grb in enumerate(grbs):
            if i==615:#2m temperature 

                hrrr_lats, hrrr_lons = grb.latlons()

                west_extent = -82
                east_extent = -79.5
                south_extent = 27
                north_extent = 29

                hrrr_lats_1d = np.squeeze(hrrr_lats[:,0])
                hrrr_lons_1d = np.squeeze(hrrr_lons[0,:])
                
                hrrr_lats_south_idx = np.where(hrrr_lats_1d>=south_extent)[0][0]
                y_idx1 = hrrr_lats_south_idx
                hrrr_lats_north_idx = np.where(hrrr_lats_1d>=north_extent)[0][0]+1
                y_idx2 = hrrr_lats_north_idx

                hrrr_lons_west_idx = np.where(hrrr_lons_1d>=west_extent)[0][0]
                x_idx1 = hrrr_lons_west_idx
                hrrr_lons_east_idx = np.where(hrrr_lons_1d>=east_extent)[0][0]+1
                x_idx2 = hrrr_lons_east_idx

                hrrr_LC_lats = hrrr_lats_1d[hrrr_lats_south_idx:hrrr_lats_north_idx]
                hrrr_LC_lons = hrrr_lons_1d[hrrr_lons_west_idx:hrrr_lons_east_idx]
            
                temp_2m = grb.values #(lat,lon); (y,x); (1059,1799)
                ax = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree())
                temp_cbar = ax.contourf(hrrr_LC_lons,hrrr_LC_lats,temp_2m[y_idx1:y_idx2,x_idx1:x_idx2],cmap='coolwarm',transform=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE, edgecolor="black")
                ax.set_title('2m_Temp (K): '+ grb.validDate.strftime("%m/%d/%Y, %H:%M:%S"),fontsize=12)

            if i==618:#2m dewpoint

                hrrr_lats, hrrr_lons = grb.latlons()

                west_extent = -82
                east_extent = -79.5
                south_extent = 27
                north_extent = 29

                hrrr_lats_1d = np.squeeze(hrrr_lats[:,0])
                hrrr_lons_1d = np.squeeze(hrrr_lons[0,:])
                
                hrrr_lats_south_idx = np.where(hrrr_lats_1d>=south_extent)[0][0]
                y_idx1 = hrrr_lats_south_idx
                hrrr_lats_north_idx = np.where(hrrr_lats_1d>=north_extent)[0][0]+1
                y_idx2 = hrrr_lats_north_idx

                hrrr_lons_west_idx = np.where(hrrr_lons_1d>=west_extent)[0][0]
                x_idx1 = hrrr_lons_west_idx
                hrrr_lons_east_idx = np.where(hrrr_lons_1d>=east_extent)[0][0]+1
                x_idx2 = hrrr_lons_east_idx

                hrrr_LC_lats = hrrr_lats_1d[hrrr_lats_south_idx:hrrr_lats_north_idx]
                hrrr_LC_lons = hrrr_lons_1d[hrrr_lons_west_idx:hrrr_lons_east_idx]

                td_2m = grb.values
                ax = fig.add_subplot(2,2,2, projection=ccrs.PlateCarree())
                temp_cbar = ax.contourf(hrrr_LC_lons,hrrr_LC_lats,td_2m[y_idx1:y_idx2,x_idx1:x_idx2],cmap='coolwarm',transform=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE, edgecolor="black")
                ax.set_title('2m_Td (K): '+ grb.validDate.strftime("%m/%d/%Y, %H:%M:%S"),fontsize=12)
            
            if i==588:#mean sea level pressure 

                hrrr_lats, hrrr_lons = grb.latlons()

                west_extent = -82
                east_extent = -79.5
                south_extent = 27
                north_extent = 29

                hrrr_lats_1d = np.squeeze(hrrr_lats[:,0])
                hrrr_lons_1d = np.squeeze(hrrr_lons[0,:])
                
                hrrr_lats_south_idx = np.where(hrrr_lats_1d>=south_extent)[0][0]
                y_idx1 = hrrr_lats_south_idx
                hrrr_lats_north_idx = np.where(hrrr_lats_1d>=north_extent)[0][0]+1
                y_idx2 = hrrr_lats_north_idx

                hrrr_lons_west_idx = np.where(hrrr_lons_1d>=west_extent)[0][0]
                x_idx1 = hrrr_lons_west_idx
                hrrr_lons_east_idx = np.where(hrrr_lons_1d>=east_extent)[0][0]+1
                x_idx2 = hrrr_lons_east_idx

                hrrr_LC_lats = hrrr_lats_1d[hrrr_lats_south_idx:hrrr_lats_north_idx]
                hrrr_LC_lons = hrrr_lons_1d[hrrr_lons_west_idx:hrrr_lons_east_idx]

                mslp = grb.values
                ax = fig.add_subplot(2,2,3,projection=ccrs.PlateCarree())
                ax.contour(hrrr_LC_lons,hrrr_LC_lats,mslp[y_idx1:y_idx2,x_idx1:x_idx2]/100,transform=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE,edgecolor="black")
                ax.set_title('MSLP (mb)')
            
            if i==621: #10 meter u wind (m/s)
                u_wind = grb.values
            
            if i==622: #10 meter v wind (m/s)
                v_wind = grb.values
                     
ax = fig.add_subplot(2,2,4,projection=ccrs.PlateCarree())
ax.quiver(hrrr_LC_lons,hrrr_LC_lats,u_wind[y_idx1:y_idx2,x_idx1:x_idx2],v_wind[y_idx1:y_idx2,x_idx1:x_idx2],regrid_shape=20)
ax.add_feature(cfeature.COASTLINE,edgecolor="black")
ax.set_title('10 meter Wind Quivers (m/s)')
plt.savefig('test.png')

# 588 589:MSLP (MAPS System Reduction):Pa (instant):lambert:meanSea:level 0:fcst time 0 hrs:from 202206081700
# 621 622:10 metre U wind component:m s**-1 (instant):lambert:heightAboveGround:level 10 m:fcst time 0 hrs:from 202206081700
# 622 623:10 metre V wind component:m s**-1 (instant):lambert:heightAboveGround:level 10 m:fcst time 0 hrs:from 202206081700
# 618 619:2 metre dewpoint temperature:K (instant):lambert:heightAboveGround:level 2 m:fcst time 0 hrs:from 202206081700

            