import pickle
from helper import *
from LC_util import *
import sys
import os
import pygrib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

def main():
    print('3e_MERLIN_downselect.py')
    data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2a_MERLIN_HRRR_grid/'
    save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/3_LC_ltg/'
    files = os.listdir(data_dir)

    for f,fname in enumerate(files):
        if f==1000:
            print(f,fname)
            ds = xr.open_dataset(data_dir+fname,engine='netcdf4')
            valid_times = ds['time'].values
            print(valid_times[0])
            cc_np = ds['cc'].values
            cg_np = ds['cg'].values
            print(cg_np.shape)
            print(np.max(np.max(np.max(cg_np))))


            lat = ds['lat'].values
            lon = ds['lon'].values

            x_idxs = [1422,1486]
            y_idxs = [176,240]
            ksc_idxs = [y_idxs[0],y_idxs[1],x_idxs[0],x_idxs[1]]
            
            lc_cc_np = cc_np[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3],:]
            lc_cg_np = cg_np[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3],:]
            print(lc_cg_np.shape)
            print(np.max(np.max(np.max(lc_cg_np))))
            lc_lat = lat[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
            lc_lon = lon[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]

            ds2 = xr.Dataset(data_vars = dict(cc=(['y','x','valid_times'],lc_cc_np.astype(int)),
                                                cg=(['y','x','valid_times'],lc_cg_np.astype(int))),
                            coords=dict(valid_times=valid_times,
                                        lon=(['y','x'],lc_lon),
                                        lat=(['y','x'],lc_lat)),
                            attrs=dict(description="MERLIN lightning data downselected on the LaunchCast HRRR grid."))
            print(ds2)
            print(np.max(np.max(np.max(ds2['cg'].values))))
            ts = pd.Timestamp(valid_times[0])
            day = ts.day
            mo = ts.month
            yr = ts.year
            fsave = '%s_%s_%s.nc'%(yr,mo,day)
            print(fsave)
            ds2.to_netcdf(save_dir+fsave,engine='netcdf4')
            del ds2, ds, valid_times,cc_np,cg_np,lc_cc_np,lc_cg_np,lat,lon,lc_lat,lc_lon

if __name__=='__main__':
    main()