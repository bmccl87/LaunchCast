import os
import xarray as xr
import matplotlib.pyplot as plt

def main():
    file = '/ourdisk/hpc/ai2es/datasets/GLM/G16/2019/177/OR_GLM-L2-LCFA_G16_s20191772359200_e20191772359400_c20191772359426.nc'
    ds = xr.open_dataset(file)
    group_area = ds['group_area']
    plt.figure()
    plt.plot(group_area.values)
    plt.savefig('test_group_area.png')
    group_lat = ds['group_lat']
    group_lon = ds['group_lon']

if __name__=='__main__':
    main()