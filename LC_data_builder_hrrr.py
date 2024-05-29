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
print("hello world")
#set the directory and load the file names
hrrr_dir = '/ourdisk/hpc/ai2es/bmac87/HRRR/202206/'
files = os.listdir(hrrr_dir)
print(hrrr_dir)

#set the extent of LaunchCast bounds
west_extent = -81.61
east_extent = -79.82
south_extent = 27.7
north_extent = 29.3

#initialize the data structures to 
time_axis = []
td_array = []
u_array = []
v_array = []
mslp_array = []

fig_bool=0

for i, file in enumerate(files):
    if i>=0:#testing statement 
        if i%10==0:
            print(i)

        #load the grib file
        f = hrrr_dir+file
        grbs = pygrib.open(f)

        #append the data of the file
        time_axis.append(grbs[1].validDate)

        #get the lat lon grid
        hrrr_lats, hrrr_lons = grbs[1].latlons()

        #get the 1d arrays for lat/lon values
        hrrr_lats_1d = np.squeeze(hrrr_lats[:,0])
        hrrr_lons_1d = np.squeeze(hrrr_lons[0,:])

        #find the indices that dictate the LaunchCast model domain set by the extent variables
        #latitudes
        hrrr_lats_south_idx = np.where(hrrr_lats_1d>=south_extent)[0][0]-1
        y_idx1 = hrrr_lats_south_idx
        hrrr_lats_north_idx = np.where(hrrr_lats_1d>=north_extent)[0][0]
        y_idx2 = hrrr_lats_north_idx

        #longitudes
        hrrr_lons_west_idx = np.where(hrrr_lons_1d>=west_extent)[0][0]
        x_idx1 = hrrr_lons_west_idx
        hrrr_lons_east_idx = np.where(hrrr_lons_1d>=east_extent)[0][0]
        x_idx2 = hrrr_lons_east_idx

        #generate the lat lon grid on the 64x64 grid
        hrrr_LC_lats = hrrr_lats_1d[hrrr_lats_south_idx:hrrr_lats_north_idx]
        hrrr_LC_lons = hrrr_lons_1d[hrrr_lons_west_idx:hrrr_lons_east_idx]

        #get the 2 meter temp, get the 64x64 subset, then append to the data structure
        temp_2m = grbs[616].values #2m Temp in Kelvin; (lat,lon); (y,x); (1059,1799)
        temp_2m_K = temp_2m[y_idx1:y_idx2,x_idx1:x_idx2]

        if i==0:
            temp_array = temp_2m_K.reshape(temp_2m_K.shape[0],temp_2m_K.shape[1],1)
        else:
            temp_array = np.concatenate([temp_array,
                                         temp_2m_K.reshape(temp_2m_K.shape[0],temp_2m_K.shape[1],1)],
                                         axis=2)
        
        #get the 2m dewpoint
        td_2m = grbs[619].values #2m dewpoint in Kelvin; same shape as temp
        td_2m_K = td_2m[y_idx1:y_idx2,x_idx1:x_idx2]
        if i==0:
            td_array = td_2m_K.reshape(td_2m_K.shape[0],temp_2m_K.shape[1],1)
        else:
            td_array = np.concatenate([td_array,
                                       td_2m_K.reshape(td_2m_K.shape[0],td_2m_K.shape[1],1)],
                                       axis=2)

        #mslp in Pascal; same shape as above, #downsample to 64x64 grid and convert to mb
        mslp_mb = grbs[589].values*.01
        mslp_mb = mslp_mb[y_idx1:y_idx2,x_idx1:x_idx2]
        if i==0:
            mslp_array = mslp_mb.reshape(mslp_mb.shape[0],mslp_mb.shape[1],1)
        else:
            mslp_array = np.concatenate([mslp_array,
                                        mslp_mb.reshape(mslp_mb.shape[0],mslp_mb.shape[1],1)],
                                        axis=2)
        

        #get the u wind component in m/s, #downsample to the 64x64 grid
        u_wind = grbs[622].values #10 meter u wind component in m/s
        u = u_wind[y_idx1:y_idx2,x_idx1:x_idx2]
        if i==0:
            u_array = u.reshape(u.shape[0],u.shape[1],1)
        else:
            u_array = np.concatenate([u_array,
                                     u.reshape(u.shape[0],u.shape[1],1)],
                                     axis=2)
        
        #get the v wind component in m/s, #downsample to the 64x64 grid
        v_wind = grbs[623].values #10 meter v wind component in m/s
        v = v_wind[y_idx1:y_idx2,x_idx1:x_idx2]
        
        if i==0:
            v_array = v.reshape(v.shape[0],v.shape[1],1)
        else:
            v_array = np.concatenate([v_array,
                                    v.reshape(v.shape[0],v.shape[1],1)],
                                    axis=2)

        grbs.close()

        if fig_bool==1:
            #create the 2x2 plot
            fig = plt.figure(figsize=(10,8))

            #plot the temperature
            ax = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree())
            temp_cbar = ax.contourf(hrrr_LC_lons,hrrr_LC_lats,temp_2m_K,cmap='OrRd',transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, edgecolor="black")
            ax.set_title('2m_Temp (K)',fontsize=12)
            # plt.colorbar(temp_cbar, ax=ax)

            #plot the dewpoint
            ax = fig.add_subplot(2,2,2, projection=ccrs.PlateCarree())
            td_cbar = ax.contourf(hrrr_LC_lons,hrrr_LC_lats,td_2m_K,cmap='OrRd',transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, edgecolor="black")
            ax.set_title('2m_Td (K)',fontsize=12)
            
            #plot the mslp
            ax = fig.add_subplot(2,2,3,projection=ccrs.PlateCarree())
            mslp_cbar = ax.contour(hrrr_LC_lons,hrrr_LC_lats,mslp_mb,transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE,edgecolor="black")
            ax.set_title('MSLP (mb)',fontsize=12)
                    
            #plot the wind quivers
            g = 5
            ax = fig.add_subplot(2,2,4,projection=ccrs.PlateCarree())
            ax.quiver(hrrr_LC_lons[::g],hrrr_LC_lats[::g],u[::g,::g],v[::g,::g])
            ax.add_feature(cfeature.COASTLINE,edgecolor="black")
            ax.set_title('10 meter Wind Quivers (m/s)',fontsize=12)
            plt.suptitle(grbs[1].validDate.strftime("%m/%d/%Y, %H:%M:%S"))
            plt.savefig('images/64_64_'+str(i)+'.png')
            plt.close()

hrrr_ds = xr.Dataset(
    data_vars = dict(hrrr_2m_temp=(['lat','lon','time'],temp_array),
                    hrrr_2m_td=(['lat','lon','time'],td_array),
                    hrrr_mslp=(['lat','lon','time'],mslp_array),
                    hrrr_u=(['lat','lon','time'],u_array),
                    hrrr_v=(['lat','lon','time'],v_array)
                    ),

    coords=dict(time=time_axis,
                lon=hrrr_LC_lons,
                lat=hrrr_LC_lats
                ),

    attrs=dict(description="HRRR 10m winds in m/s, 2m Temp and Dewpoint in K, MSLP in millibars over Cape Canaveral on 64x64 grid for ML"))

print(hrrr_ds)
hrrr_ds.to_netcdf('hrrr_64_64.nc',engine='netcdf4')

#explicit output from the HRRR grid files
# 588 589:MSLP (MAPS System Reduction):Pa (instant):lambert:meanSea:level 0:fcst time 0 hrs:from 202206081700
# 621 622:10 metre U wind component:m s**-1 (instant):lambert:heightAboveGround:level 10 m:fcst time 0 hrs:from 202206081700
# 622 623:10 metre V wind component:m s**-1 (instant):lambert:heightAboveGround:level 10 m:fcst time 0 hrs:from 202206081700
# 618 619:2 metre dewpoint temperature:K (instant):lambert:heightAboveGround:level 2 m:fcst time 0 hrs:from 202206081700

            