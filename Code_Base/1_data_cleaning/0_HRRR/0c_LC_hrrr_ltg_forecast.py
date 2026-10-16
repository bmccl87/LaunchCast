import pygrib
import os
import xarray as xr
import pickle
import pandas as pd
import numpy as np

def main():
    print('test_ds')
    nc_dir = '/scratch/bmac87/'
    test_ds = xr.open_dataset(nc_dir+'LC_2024_test_ds_w_hrrr_ltg.nc',engine='netcdf4')
    samples_df = pickle.load(open('/home/bmac87/LaunchCast/1_data_cleaning/5_Merge/2024_samples_w_ltg_counts_w_efm_flag.pkl','rb'))
    test_times = pd.to_datetime(test_ds['sample_time'].values)
    df_test = samples_df[samples_df['R60245'].isin(test_times)]
    h_inits = df_test['H_INIT'].values
    ltg_list = []
    
    for h,h_init in enumerate(h_inits):
        if h>=0:
            print(h)
            ts = pd.Timestamp(h_init)
            year = f"{ts.year:04}"
            mo = f"{ts.month:02}"
            day = f"{ts.day:02}"
            hour = f"{ts.hour:02}"
            fstr = 'hrrr_%s%s%s%s_f002.grib'%(year,mo,day,hour)

            grb_dir = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_f02/%s%s/'%(year,mo)
            grb = pygrib.open(grb_dir+fstr)
            
            LC_hrrr_indexes = pickle.load(open('../LC_hrrr_indexes.pkl','rb'))

            x1 = LC_hrrr_indexes['x'].start
            x2 = LC_hrrr_indexes['x'].stop
            y1 = LC_hrrr_indexes['y'].start
            y2 = LC_hrrr_indexes['y'].stop

            ltg = grb.select(name='Lightning')[0].values[y1:y2,x1:x2]
            print(grb.select(name='Lightning')[0])
            ltg_list.append(ltg)
            grb.close()
            del ltg
    ltg_all = np.stack(ltg_list,axis=0)
    print(ltg_all.shape)
    test_ds['hrrr_ltg_f002'] = (('sample_time','y','x'),ltg_all)
    test_ds.to_netcdf('/scratch/bmac87/LC_2024_test_ds_w_hrrr_ltg_f01_f02.nc',engine='netcdf4')
    plt.figure()
    plt.hist(np.ravel(ltg_all),bins=20)
    plt.savefig('./temp_hrrr_ltg.png')
    plt.close()

if __name__=='__main__':
    main()