import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import shutil
import pickle
from LC_util import *
from datetime import datetime as dt
from datetime import timedelta

def load_LC_data(hrrr_dir,
                 ltg_dir):

    x = xr.open_dataset(hrrr_dir,engine='netcdf4')
    y = xr.open_dataset(ltg_dir,engine='netcdf4')

    x = x.sortby('time')
    y = y.sortby('time')

    x = x.transpose()
    print(x)

    x_dict = {'hrrr_sfc_u':x['hrrr_u'].values,
              'hrrr_sfc_v':x['hrrr_v'].values,
              'hrrr_sfc_temp':x['hrrr_2m_temp'].values,
              'hrrr_sfc_moist':x['hrrr_2m_td'].values,
              'hrrr_sfc_pres':x['hrrr_mslp'].values
              }
    print(x_dict['hrrr_sfc_u'].shape)
    y_dict = {'MERLIN_CG':y['strikes'].values}
    print(y)

    trng_val_data = {'x':x_dict, 'y':y_dict}
    pickle.dump(trng_val_data,open('/scratch/bmac87/0622_trng_val.p','wb'))
    return x_dict, y_dict



def main():
    hrrr_file = '/scratch/bmac87/hrrr_64_64.nc'
    ltg_file = '/scratch/bmac87/binned_CG/062022.nc'

    x_dict,y_dict = load_LC_data(hrrr_dir=hrrr_file,
                 ltg_dir=ltg_file)

    

if __name__=='__main__':
    main()