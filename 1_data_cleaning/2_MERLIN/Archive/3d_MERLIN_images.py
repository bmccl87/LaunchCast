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

def save_yr_mo():
    if args['ltg_type']=='CC':
        print('loading the CC dataset')
        ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_MRMS_grid/CC_mrms.nc',engine='netcdf4')
    
    if args['ltg_type']=='CG':
        print('loading the CG dataset')
        ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_MRMS_grid/CG_mrms.nc',engine='netcdf4')

    #downselect for the months and years of the valid data times
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    years = ['2018','2019','2020','2021','2022','2023','2024']

    months_int = np.arange(1,13)
    yrs_int = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

    print(months_int)
    for y,yr in enumerate(yrs_int):
        for m,month in enumerate(months_int):
            print('selecting:',yr,month)
            mo_yr_ds = ds.sel(time=(ds['time'].dt.year == yr) & (ds['time'].dt.month == month))
            save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_MRMS_grid/monthly/'
            fsave = '%s_%s_%s.nc'%(months[m],years[y],args['ltg_type'])
            print(fsave)
            mo_yr_ds.to_netcdf(save_dir+fsave,engine='netcdf4')
            del mo_yr_ds

def build_yr_mo_images(year='2022',month='06'):
    print('building images for: ',year,month)
    data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_MRMS_grid/monthly/'
    cc_fname = '%s_%s_%s.nc'%(month,year,'CC')
    cg_fname = '%s_%s_%s.nc'%(month,year,'CG')

    cc_ds = xr.open_dataset(data_dir+cc_fname,engine='netcdf4')
    cg_ds = xr.open_dataset(data_dir+cg_fname,engine='netcdf4')
    
    lat = cc_ds['lat'].values
    lon = cc_ds['lon'].values
    cc_valid_times = cc_ds['time'].values

    for v,vt in enumerate(cc_valid_times):
        if v%1000==0:
            print(v,len(cc_valid_times))
        cc_data = cc_ds.sel(time=vt)
        cg_data = cg_ds.sel(time=vt)
        
        cc_fed = cc_data['fed'].values
        cg_fed = cg_data['fed'].values

        ts = pd.Timestamp(vt)
        mi = f"{ts.minute:02}"
        hr = f"{ts.hour:02}"
        day = f"{ts.day:02}"
        mo = f"{ts.month:02}"
        yr = f"{ts.year:04}"

        title_str = '%s/%s/%s %s:%s UTC'%(mo,day,yr,hr,mi)
        fname = '%s_%s_%s_%s_%s.png'%(yr,mo,day,hr,mi)
        
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
        plt.suptitle(title_str)

        save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_MRMS_grid/monthly/images/'
        if os.path.isdir(save_dir)==False:
            os.makedirs(save_dir)
        plt.savefig(save_dir+fname)
        plt.close()
        del cc_data, cg_data

def build_hourly_image(year='2022',mo='06',day='30',hr='20'):

    data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2a_MERLIN_HRRR_grid/%s%s/'%(year,mo)
    fname = 'MERLIN_hrrr_%s%s%s%s.nc'%(year,mo,day,hr)
    ds = xr.open_dataset(data_dir+fname,engine='netcdf4')

    cc_fed = ds['cc'].values.astype(float)
    cc_fed_plot = cc_fed
    cc_fed_plot[cc_fed_plot<1] = np.nan

    cg_fed = ds['cg'].values.astype(float)
    cg_fed_plot = cg_fed
    cg_fed_plot[cg_fed<1] = np.nan
    
    cmap='Reds'
    lon = ds['lon'].values
    lat = ds['lat'].values

    x_idxs = [1422,1486]
    y_idxs = [176,240]
    ksc_idxs = [y_idxs[0],y_idxs[1],x_idxs[0],x_idxs[1]]
    lon = lon[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
    lat = lat[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
    cc_fed = cc_fed[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
    cc_fed_plot = cc_fed_plot[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
    cg_fed = cg_fed[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
    cg_fed_plot = cg_fed_plot[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]

    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
    cc_cb = ax.pcolormesh(lon,lat,cc_fed_plot,cmap=cmap)
    ax.coastlines()
    ax.set_title('IC Flashes')

    ax = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
    cc_cb = ax.pcolormesh(lon,lat,cg_fed_plot,cmap=cmap)
    ax.coastlines()
    ax.set_title('CG Flashes')
    fsave = 'MERLIN_hrrr_%s%s%s%s.png'%(year,mo,day,hr)
    save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2aa_MERLIN_HRRR_grid_png/'
    plt.savefig('%s%s'%(save_dir,fsave))
    plt.close()

    ds2 = xr.Dataset(data_vars = dict(cc=(['y','x'],cc_fed),
                                                cg=(['y','x'],cg_fed)),
                                coords=dict(time=ds['time'],
                                            lon=(['y','x'],lon),
                                            lat=(['y','x'],lat)),
                                attrs=dict(description="MERLIN lightning data on the HRRR grid, downselected to 64x64.  cc is the number of \
                                    flashes per hrrr grid. cg is the number of flashes per hrrr grid. this is for the \
                                        hrrr grid. This is one hour temporal resolution, with the lightning binned over\
                                            the next hour. Thus a time of 06-30-2022 01Z has lightning valid between 01-02Z."))

    save_dir = '/scratch/bmac87/LC_scratch/3_MERLIN_KSC/%s%s/'%(year,mo)
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    fname = 'MERLIN_hrrr_%s%s%s%s.nc'%(year,mo,day,hr)
    ds2.to_netcdf(save_dir+fname,engine='netcdf4')


if __name__=='__main__':

    
    print('making merlin video')
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--year',type=int, default=1, help='The year to process')
    # parser.add_argument('--month',type=int, default=6, help='The month to process')
    # parser.add_argument('--day',type=int, default=30, help='The day to process')
    # parser.add_argument('--ltg_type',type=str,default='CC',help='The type of lightning')
    # args = vars(parser.parse_args())
    # print(args)

    # if args['year']==0:
    #     year='2019'
    # elif args['year']==1:
    #     year='2020'
    # elif args['year']==2:
    #     year='2021'
    # elif args['year']==3:
    #     year='2022'
    # elif args['year']==4:
    #     year='2023'
    # else:
    #     year='2024'

    years = ['2023','2024']

    hrs = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    mos = ['01','02','03','04','05','06','07','08','09','10','11','12']

    yrs_dict = {}
    for yr in years:
        mos_dict = {}
        for mo in mos: #for each month
            days = []
            mo_jul = []
            if mo=='01' or mo=='03' or mo=='05' or mo=='07' or mo=='08' or mo=='10' or mo=='12':
                for t in range(1,32):
                    day_str = f"{t:02}"
                    days.append(day_str)
            elif mo=='02':
                if yr=='2020' or yr=='2024':
                    for t in range(1,30):
                        days.append(f"{t:02}")
                else:
                    for t in range(1,29):
                        days.append(f"{t:02}")
            else:
                for t in range(1,31):
                    days.append(f"{t:02}")
            mos_dict.update({mo:days})
        yrs_dict.update({yr:mos_dict})

    for yr in years:
        mos_dict = yrs_dict[yr]
        for mo in mos:
            days = mos_dict[mo]
            for day in days:
                print(yr,mo,day)
                for hr in hrs:
                    build_hourly_image(year=yr,mo=mo,day=day,hr=hr)