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
from LC_util import *
from datetime import datetime as dt
from datetime import timedelta

def to_netcdf(ltg_df, #pandas dataframe with time series lightning data (CC or CG)
              start_time, #string MM/DD/YYYY HH:MM:SS
              end_time, #string MM/DD/YYYY HH:MM:SS
              time_delta #int, number of seconds to bin the data
              ):

    start_dt = dt.strptime(start_time,'%m/%d/%Y %H:%M:%S')
    print(start_dt)
    end_dt = dt.strptime(end_time,'%m/%d/%Y %H:%M:%S')
    print(end_dt)

    time_sample = timedelta(0, time_delta)

    time_list = []
    while(start_dt<end_dt):
        start_dt = start_dt+time_sample
        print(start_dt)
        time_list.append(start_dt)


    #load the HRRR grid
    hrrr_data = xr.open_dataset('hrrr_64_64.nc')
    hrrr_lats = hrrr_data['lat'].values
    hrrr_lons = hrrr_data['lon'].values

    #convert longitudes to plus 360
    xedge = hrrr_lons+360
    yedge = hrrr_lats

    xmid = [] #Blank array
    ymid = [] #Blank array
    print(yedge[0])
    print(yedge[len(yedge)-1])
    print(xedge[0])


    ltg_df['Lon_Decimal_360'] = ltg_df['Lon_Decimal']+360
    ltg_df_subset = ltg_df.loc[ltg_df['Lon_Decimal_360']>=xedge[0]]
    ltg_df_subset = ltg_df_subset.loc[ltg_df_subset['Lon_Decimal_360']<=xedge[len(xedge)-1]]
    ltg_df_subset = ltg_df_subset.loc[ltg_df_subset['Lat_Decimal']>=yedge[0]]
    ltg_df_subset = ltg_df_subset.loc[ltg_df_subset['Lat_Decimal']<=yedge[len(yedge)-1]]

    # print(ltg_df_subset)

    i=0
    while(i < len(xedge)-1):
        xmid.append((xedge[i]+xedge[i+1])/2) #Calculate and append midpoints
        i+=1
    i=0

    while(i < len(yedge)-1):
        ymid.append((yedge[i]+yedge[i+1])/2) #Calculate and append midpoints
        i+=1

    #consider rounding 

    #first time rounded down/backward to the nearest 5 minutes
    # first_time = temp1_subset['Date_Time'].iloc[0]
    # print(first_time)
    # if round(first_time.minute,-1)==60:
    #     start_time = first_time.replace(minute=55,second=0,microsecond=0,nanosecond=0) #start_slicing with this time
    # else:
    #     start_time = first_time.replace(minute=round(first_time.minute,-1),second=0,microsecond=0,nanosecond=0)
    # #last time rounded up to the forward nearest 5 minutes
    # last_time = temp1['Date_Time'].iloc[len(temp1)-1]
    # end_time = last_time.replace(minute=round(last_time.minute,+1),second=0,microsecond=0,nanosecond=0)
    # print(end_time)

    # # last_time = temp1['Date_Time'][len(temp1)-1]
    # time_sample = dt.timedelta(0, 3600)
    # temp_array = xr.Dataset()
    # tempArrayList = []
    # tempArrayTimeList = []

    # t=0
    # while(start_time<=end_time):
    #     temp_df = temp1[slice(start_time,start_time+time_sample)]
    #     if len(temp_df)>0: #deviates from randy's and tobias's code
    #         C = util.boxbin(temp_df['Lon_Decimal']+360, temp_df['Lat_Decimal'], xedge, yedge, mincnt=0)
    #         tempArray = xr.Dataset(
    #             data_vars=dict(strikes=(["x", "y"], C)),
    #             coords=dict(
    #                 lon=(["x"], xmid),
    #                 lat=(["y"], ymid),
    #             ),
    #             attrs=dict(description="Lightning data"),
    #         )  # Create dataset
    #         tempArrayList.append(tempArray)
    #         tempArrayTimeList.append(start_time)

    #     start_time = start_time+time_sample
    #     t=t+1
    
    # tempArray = xr.concat(tempArrayList, data_vars='all', dim='time')
    # tempArray = tempArray.assign_coords(time=tempArrayTimeList)
    # tempArray = tempArray.fillna(0)
    # print(tempArray)
    
    # # tempArray.to_netcdf('/Users/brandonmcclung/Data/netcdfs/merlin_cc_16jan24/'+mo+yr+'.nc',mode='w',format="NETCDF4") #Save
    #  #Print save message
    # return tempArray

def main():
    #declare start and end time
    start_time = '06/01/2022 00:00:00'
    end_time = '07/01/2022 00:00:00'

    #load the in cloud data (already June 2022)
    cc_df = pd.read_pickle('/ourdisk/hpc/ai2es/bmac87/Lightning_Project/data/pickles/merlin_cc_df.p')

    #load the cg data 
    cg_df = pd.read_pickle('/ourdisk/hpc/ai2es/bmac87/Lightning_Project/data/pickles/merlin_cg_df.p')
    cg_df.index = pd.to_datetime(cg_df.index)
    cg_df = cg_df[slice(start_time,end_time)]

    to_netcdf(ltg_df=cc_df,#dataframe
              start_time=start_time,#string
              end_time=end_time,#string
              time_delta=3600)#int in seconds

if __name__=='__main__':
    main()




