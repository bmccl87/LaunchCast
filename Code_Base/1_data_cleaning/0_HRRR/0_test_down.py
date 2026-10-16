import matplotlib.pyplot
import os
import xarray as xr
import numpy as np

def main():
    year='2022'
    month='06'
    data_dir = '/scratch/bmac87/HRRR_Subhourly_Downselect_v3/%s%s/'%(year,month)
    files = os.listdir(data_dir)
    for file in files:
        ds = xr.open_dataset(data_dir+file,engine='netcdf4')
        pr = ds['precip_rate'].values
        print(pr.shape)
        print(np.nanmax(pr))
        print(np.nanmin(pr))

if __name__=='__main__':
    main()