import numpy as np
import os
import pickle
import xarray as xr


# yr is four digit string
# julian is an int
def get_day_mo(yr,julian):
    """
    This code helps generate strings from julian ints. This is good for experiment control.
    """
    import datetime
    yr=yr[2:]
    julian=f"{julian:03}"
    date_time = datetime.datetime.strptime(yr+julian, '%y%j').date()
    month = date_time.month
    day = date_time.day
    return f"{month:02}",f"{day:02}"

def time_stuff():
    years = ['2018','2019','2020','2021','2022','2023','2024']
    months = ['01','02','03','04','05','06',
            '07','08','09','10','11','12']
    hours = ['00','01','02','03','04','05','06',
            '07','08','09','10','11','12',
            '13','14','15','16','17','18',
            '19','20','21','22','23']
    half_hours = ['00','30']
    return years, months, hours, half_hours

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
    return yrs_dict
    
def misc():
    """
    This is old code. 
    """
    mins_prior = ['55','50','45','40','35','30','25','20','15','10','05','00']
    keys = ['1','2','4','5','6','7','8','9','10','11','12','14','15','16','17',
            '18','20','21','22','24','25','26','27','28','29','30','31','32',
            '34','35']
    stats = ['mean','std']
    feature_list = get_feature_list(stats=stats, mins_prior=mins_prior, keys=keys)

def ds_example():
    ltg_ds = xr.Dataset(data_vars = dict(cc=(['y','x'],cc_grid.astype(int)),
                                                cg=(['y','x'],cg_grid.astype(int))),
                                coords=dict(time=slice_times[t],
                                            lon=(['y','x'],hrrr_lon),
                                            lat=(['y','x'],hrrr_lat)),
                                attrs=dict(description="MERLIN lightning data on the HRRR grid.  cc is the number of \
                                    flashes per hrrr grid. cg is the number of flashes per hrrr grid. this is for the \
                                        hrrr grid. This is one hour temporal resolution, with the lightning binned over\
                                            the next hour. Thus a time of 06-30-2022 01Z has lightning valid between 01-02Z."))

