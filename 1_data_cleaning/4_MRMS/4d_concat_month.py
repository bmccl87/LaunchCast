import os
import pickle
import xarray as xr
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def create_parser():
    parser = argparse.ArgumentParser(description='Concat', fromfile_prefix_chars='@')
    parser.add_argument('--var',type=int,default='0',help='variable to process')
    args = parser.parse_args()
    variable = args.var
    return variable

def years_dict():
    """
    This code generates a dictionary for the date and hour information. 
    """
    years = ['2018','2019','2020','2021','2022','2023','2024']
    hrs = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    mos = ['01','02','03','04','05','06','07','08','09','10','11','12']

    yrs_dict = {}
    for yr in years:#for each year
        mos_dict = {}
        for mo in mos: #for each month
            days = []
            mo_jul = []
            if mo=='01' or mo=='03' or mo=='05' or mo=='07' or mo=='08' or mo=='10' or mo=='12':
                for t in range(1,32):
                    days.append(f"{t:02}")
            elif mo=='02':
                if yr=='2020':
                    for t in range(1,30):
                        days.append(f"{t:02}")
                elif yr=='2024':
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
    return yrs_dict

def concat_month(month='06',year='2022',variable='Reflectivity_0C_00.50'):
    date_dict = years_dict()
    yr_dict = date_dict[year]
    days = yr_dict[month]
    data_list = []
    time_list = []

    hrrr_grid = pickle.load(open('./grid_dict_2d.pkl','rb'))
    for key in hrrr_grid:
        print(key)
    lat = hrrr_grid['LC_lats']
    lon = hrrr_grid['LC_lons']
    for day in days:
        data_dir = '/scratch/bmac87/MRMS_on_HRRR_grid/%s/%s%s%s/'%(variable,year,month,day)
        if os.path.isdir(data_dir)==True:
            files = sorted(os.listdir(data_dir))
            for f,file in enumerate(files):
                temp_data = pickle.load(open(data_dir+file,'rb'))
                ds = xr.Dataset(data_vars = dict(radar_data=(['y','x'],temp_data['data'])),
                                    coords=dict(lon=(['y','x'],lon),
                                                lat=(['y','x'],lat)))
                data_list.append(ds)
                time_list.append(temp_data['time'])
                del temp_data, ds
    
    if len(data_list)>0:
        ds1 = xr.concat(data_list, data_vars='all', dim='time')
        ds1 = ds1.assign_coords(time=time_list)
        ds1 = ds1.sortby('time')
        fsave = '%s%s.nc'%(year,month)
        save_dir = '/scratch/bmac87/MRMS_moyr_concat/%s/'%variable

        if os.path.isdir(save_dir)==False:
            os.makedirs(save_dir)
        ds1.to_netcdf(save_dir+fsave,engine='netcdf4')

def concat_month_driver():
    print('in 4d_concat_month.py')
    years = ['2021','2022','2023','2024']
    months = ['01','02','03','04','05','06',
            '07','08','09','10','11','12']
    variables = ['MergedZdr_00.50',#0
                'MergedZdr_00.75',#1
                'MergedZdr_01.00',#2
                'MergedZdr_01.25',#3
                'MergedZdr_01.50',#4
                'MergedZdr_01.75',#5
                'MergedZdr_02.00',#6
                'MergedZdr_02.25',#7
                'MergedZdr_02.50',#8
                'MergedZdr_02.75',#9
                'MergedZdr_03.00',#10
                'MergedZdr_03.50',#11
                'MergedZdr_04.00',#12
                'MergedZdr_04.50',#13
                'MergedZdr_05.00',#14
                'MergedZdr_05.50',#15
                'MergedZdr_06.00',#16
                'MergedZdr_06.50',#17
                'MergedZdr_07.00',#18
                'MergedZdr_07.50',#19
                'MergedZdr_08.00',#20
                'MergedZdr_08.50',#21
                'MergedZdr_09.00',#22
                'MergedZdr_10.00',#23
                'MergedZdr_11.00',#24
                'MergedZdr_12.00',#25
                'MergedZdr_13.00',#26
                'MergedZdr_14.00',#27
                'MergedZdr_15.00',#28
                'MergedZdr_16.00',#29
                'MergedZdr_17.00',#30
                'MergedZdr_18.00']#31

    var_idx = create_parser()
    variable = variables[var_idx]
    print(variable)
    for year in years:
        for month in months:
            concat_month(variable=variable,month=month,year=year)

def test_concat_month_imshow(variable='VII_00.50',year='2022',month='06'):
    
    print('testing concatenating and regridding')
    data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/MRMS_moyr_concat/%s/'%(variable)
    file = '%s%s.nc'%(year,month)
    ds = xr.open_dataset(data_dir+file,engine='netcdf4')
    save_dir = '/scratch/bmac87/MRMS_test_concat/%s/'%variable
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    radar_data = ds['radar_data'].values
    valid_times = ds['time'].values
    del ds

    for v,vt in enumerate(valid_times):
        if v%100==0:
            ts = pd.Timestamp(vt)
            sec = ts.second
            minute = ts.minute
            hour = ts.hour
            day = ts.day
            month = ts.month
            year = ts.year
            fsave = '%s%s%s_%s%s%s.png'%(
                f"{year:04}",
                f"{month:02}",
                f"{day:02}",
                f"{hour:02}",
                f"{minute:02}",
                f"{sec:02}"
            )
            data = radar_data[v,:,:]
            max = np.max(np.max(data))
            min = np.min(np.min(data))
            plt.figure(figsize=(10,10))
            im = plt.imshow(data)
            cb = plt.colorbar(im)
            plt.title('max: '+str(max)+' '+'min: '+str(min))
            plt.savefig(save_dir+fsave)
            plt.close()
            del data, max, min, im, cb,ts,sec,minute,hour,day,month,year
    del radar_data, valid_times, data_dir, save_dir

if __name__=='__main__':

    years = ['2022','2023','2024']
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    variables = ['MergedZdr_00.50',#0
                'MergedZdr_00.75',#1
                'MergedZdr_01.00',#2
                'MergedZdr_01.25',#3
                'MergedZdr_01.50',#4
                'MergedZdr_01.75',#5
                'MergedZdr_02.00',#6
                'MergedZdr_02.25',#7
                'MergedZdr_02.50',#8
                'MergedZdr_02.75',#9
                'MergedZdr_03.00',#10
                'MergedZdr_03.50',#11
                'MergedZdr_04.00',#12
                'MergedZdr_04.50',#13
                'MergedZdr_05.00',#14
                'MergedZdr_05.50',#15
                'MergedZdr_06.00',#16
                'MergedZdr_06.50',#17
                'MergedZdr_07.00',#18
                'MergedZdr_07.50',#19
                'MergedZdr_08.00',#20
                'MergedZdr_08.50',#21
                'MergedZdr_09.00',#22
                'MergedZdr_10.00',#23
                'MergedZdr_11.00',#24
                'MergedZdr_12.00',#25
                'MergedZdr_13.00',#26
                'MergedZdr_14.00',#27
                'MergedZdr_15.00',#28
                'MergedZdr_16.00',#29
                'MergedZdr_17.00',#30
                'MergedZdr_18.00',#31
                'Reflectivity_-10C_00.50',#32
                'Reflectivity_-15C_00.50',#33
                'Reflectivity_-20C_00.50',#34
                'Reflectivity_-5C_00.50',#35
                'Reflectivity_0C_00.50',#36
                'VII_00.50',#37
                'VIL_00.50']#38
    
    var_idx = create_parser()
    variable = variables[var_idx]

    for year in years:
        for month in months:
            print(variable,year,month)
            try:
                test_concat_month_imshow(variable=variable,year=year,month=month)
            except Exception as e:
                print(e)
