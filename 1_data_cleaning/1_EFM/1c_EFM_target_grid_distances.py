import os
import pandas as pd
import shutil
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import glob
import argparse
from helper import *
import time
import cartopy.geodesic as gd

def calc_distances_from_efm_2_targets(hrrr_dict = {}, efm_df=pd.DataFrame()):
    """
    This function calculates the distance between each HRRR data point, and the EFM locations.
    """

    

    print('calculating the distances to the EFM sites')
    site_names = efm_df['SiteName'].values
    site_lats = efm_df['Latitude'].values
    site_lons = efm_df['Longitude'].values

    ########DOWNSELECT INDEXES FOR THE FINAL TARGET#############
    #x_idxs for slicin: 23:39
    #y_idxs for slicin: 26:42
    ############################################################

    target_lat = hrrr_dict['LC_lats']
    target_lon = hrrr_dict['LC_lons']
    target_dict = {'target_lat':target_lat,'target_lon':target_lon}
    pickle.dump(target_dict,open('target_grid_2d.pkl','wb'))
    k = gd.Geodesic()

    ######EXAMPLE USING K#######################
    # Define the two points as NumPy arrays [longitude, latitude]
    # coord1 = np.array([77.343750, 22.593726])
    # coord2 = np.array([86.945801, 23.684774])
    # distance_meters = k.inverse(coord1, coord2)[0, 0]
    ############################################

    distances = np.zeros((target_lon.shape[0],target_lon.shape[0],len(site_names)))
    for i in range(target_lon.shape[0]):
        for j in range(target_lon.shape[0]):
            for sn in range(len(site_names)):
                hrrr_coord = np.array([target_lon[i,j],target_lat[i,j]])
                efm_coord = np.array([site_lons[sn],site_lats[sn]])
                distances[i,j,sn] = k.inverse(hrrr_coord, efm_coord)[0, 0]
                del hrrr_coord, efm_coord
    
    ds = xr.Dataset(data_vars = dict(dist=(['y','x','sn'],distances)),
                                coords=dict(site_names=(['sn'],site_names),
                                            lon=(['y','x'],target_lon),
                                            lat=(['y','x'],target_lat)),
                                attrs=dict(description="This dataset has the fixed distances to the final/target hrrr grid in meters."))
    ds.to_netcdf('./efms2hrrrdist.nc',engine='netcdf4')

def visualize():
    ds = xr.open_dataset('./efms2hrrrdist.nc',engine='netcdf4')
    data_np = ds['dist'].values
    for i in range(data_np.shape[-1]):
        plt.figure()
        plt.pcolormesh(data_np[:,:,i])
        plt.savefig('dist_%s.png'%i)
        plt.close()

def test():
    ds = xr.open_dataset('./efms2hrrrdist.nc',engine='netcdf4')
    print(ds['site_names'].values)

def merge_into_final_xr():
    dist_ds = xr.open_dataset('./efms2hrrrdist.nc',engine='netcdf4')
    site_names = dist_ds['site_names'].values
    years = ['2021','2022','2023','2024']
    data_dir = '/scratch/bmac87/88_LC_observed_datasets/6_final_xr_observed/'
    FM_keys = ['FM01', 'FM02', 'FM04', 'FM05', 'FM06', 'FM07', 'FM08', 'FM09', 'FM10', 'FM11', 'FM12', 'FM14', 'FM15', 'FM16', 'FM17', 'FM18', 'FM19', 'FM20', 'FM21', 'FM22', 'FM24', 'FM25', 'FM26', 'FM27', 'FM28', 'FM29', 'FM30', 'FM31', 'FM32', 'FM34', 'FM35']
    dist_np = dist_ds['dist'].values
    print('dist_np.shape',dist_np.shape)
    dist_list = []
    for f,fm in enumerate(FM_keys):
        fm_idx = np.where(site_names==FM_keys[f])[0][0]
        dist_list.append(dist_np[:,:,fm_idx])
    dist_np2 = np.stack(dist_list,axis=-1)

    for year in years:
        print(year)
        fname = 'LC_observed_%s.nc'%year
        data_ds = xr.open_dataset(data_dir+fname,engine='netcdf4')
        sample_times = data_ds['sample_time'].values
        num_samples = len(sample_times)
        num_prior = 4
        sample_list = []
        for s in range(num_samples):
            num_list = []
            for n in range(num_prior):
                num_list.append(dist_np2)
            num_stack = np.stack(num_list,axis=0)
            sample_list.append(num_stack)
            del num_list, num_stack
        efm_dist_samples = np.stack(sample_list,axis=0)
        data_ds['efm_distances'] = (['sample_time','pt','y','x','efm_dist'],efm_dist_samples)
        data_ds.to_netcdf(data_dir+fname+'_w_EFM_dist.nc',engine='netcdf4')
        
def main():
    # print('calculating the distances to the HRRR grid points')

    # print('loading the efm locations')
    # efm_df = pd.read_excel('EFM_Locations.xlsx')
    # print('efm locations loaded successfully')

    # print('loading the hrrr grid')
    # hrrr_dict = pickle.load(open('../2_MERLIN/grid_dict_2d.pkl','rb'))
    # print('hrrr grid loaded successfully')

    # print('calculating the distances to the target grid')
    # calc_distances_from_efm_2_targets(hrrr_dict = hrrr_dict, efm_df=efm_df)
    # print('all distances calculated successfully')

    # visualize()
    # test()
    merge_into_final_xr()

if __name__=='__main__':
    main()