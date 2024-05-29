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
import pickle
from LC_util import *
from datetime import datetime as dt
from datetime import timedelta

def to_netcdf(ltg_df, #pandas dataframe with time series lightning data (CC or CG)
              start_time, #string MM/DD/YYYY HH:MM:SS
              end_time, #string MM/DD/YYYY HH:MM:SS
              time_delta, #int, number of seconds to bin the data
              ltg_type#string: CC or CG
              ):

    start_dt = dt.strptime(start_time,'%m/%d/%Y %H:%M:%S')
    print(start_dt)
    end_dt = dt.strptime(end_time,'%m/%d/%Y %H:%M:%S')
    print(end_dt)

    time_sample = timedelta(0, time_delta)

    time_list = []
    while(start_dt<=end_dt):
        time_list.append(start_dt)
        start_dt = start_dt+time_sample
        

    #load the HRRR grid
    hrrr_data = xr.open_dataset('/scratch/bmac87/hrrr_64_64.nc')
    hrrr_lats = pickle.load(open('/scratch/bmac87/HRRR_ltg_bin_grid_lats_1d.p','rb'))
    hrrr_lons = pickle.load(open('/scratch/bmac87/HRRR_ltg_bin_grid_lons_1d.p','rb'))

    #convert longitudes to plus 360
    xedge = hrrr_lons+360
    yedge = hrrr_lats

    xmid = [] #Blank array
    ymid = [] #Blank array

    ltg_df['Lon_Decimal_360'] = ltg_df['Lon_Decimal']+360
    ltg_df_subset = ltg_df.loc[ltg_df['Lon_Decimal_360']>=xedge[0]]
    ltg_df_subset = ltg_df_subset.loc[ltg_df_subset['Lon_Decimal_360']<=xedge[len(xedge)-1]]
    ltg_df_subset = ltg_df_subset.loc[ltg_df_subset['Lat_Decimal']>=yedge[0]]
    ltg_df_subset = ltg_df_subset.loc[ltg_df_subset['Lat_Decimal']<=yedge[len(yedge)-1]]

    i=0
    while(i < len(xedge)-1):
        xmid.append((xedge[i]+xedge[i+1])/2) #Calculate and append midpoints
        i+=1
    i=0

    while(i < len(yedge)-1):
        ymid.append((yedge[i]+yedge[i+1])/2) #Calculate and append midpoints
        i+=1

    binned_ltg_list = []
    time_ltg_list = []

    #grid the lightning data, rounded down.  i.e. 00-01Z lightning is binned to the 00Z lightning
    for t in range(len(time_list)-1):
        print(time_list[t])
        temp_df = ltg_df_subset[slice(time_list[t],time_list[t+1])]
        if len(temp_df)>0:
            gridded_ltg = boxbin(temp_df['Lon_Decimal']+360,temp_df['Lat_Decimal'],xedge,yedge,mincnt=0)
        else:
            gridded_ltg=np.zeros((len(xmid),len(ymid)))

        temp_ds = xr.Dataset(
                data_vars = dict(strikes=(["x","y"],gridded_ltg)),
                coords=dict(lon=(["x"],xmid),
                            lat=(["y"],ymid)),
                attrs=dict(description="MERLIN Flashes: "+ltg_type)
            )
        temp_ds = temp_ds.fillna(0)    
        binned_ltg_list.append(temp_ds)
        time_ltg_list.append(time_list[t])
            
    ds = xr.concat(binned_ltg_list, data_vars='all',dim='time')
    ds = ds.assign_coords(time=time_ltg_list)
    dt_str = (start_dt-time_sample-time_sample).strftime('%m%Y')
    ds.to_netcdf('/scratch/bmac87/binned_'+ltg_type+'/'+dt_str+'.nc',mode='w',engine='netcdf4')
    print(ds)

    

def main():
    #declare start and end time
    start_time = '06/01/2022 00:00:00'
    end_time = '07/01/2022 00:00:00'

    #load the in cloud data (already June 2022)
    cc_ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/KSC_Weather_Archive/MERLIN_Flashes/merlin_cc_flash_time_series/2022_06_fl_cc.nc',engine='netcdf4')
    cc_df = cc_ds.to_dataframe()
    print(cc_df)

    #load the cg data 
    cg_ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/KSC_Weather_Archive/MERLIN_Flashes/merlin_cg_flash_time_series/merlin_cg_flashes.nc',engine='netcdf4')
    cg_df = cg_ds.to_dataframe()
    cg_df.index = pd.to_datetime(cg_df.index)
    cg_df = cg_df[slice(start_time,end_time)]
    print(cg_df)

    to_netcdf(ltg_df=cg_df,#dataframe
              start_time=start_time,#string
              end_time=end_time,#string
              time_delta=3600,#int in seconds
              ltg_type='CG'
              )

if __name__=='__main__':
    main()