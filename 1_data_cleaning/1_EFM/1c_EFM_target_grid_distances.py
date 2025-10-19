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

    target_lat = hrrr_dict['LC_lats'][23:39,26:42]
    target_lon = hrrr_dict['LC_lons'][23:39,26:42]
    target_dict = {'target_lat':target_lat,'target_lon':target_lon}
    pickle.dump(target_dict,open('target_grid_2d.pkl','wb'))
    k = gd.Geodesic()

    ######EXAMPLE USING K#######################
    # Define the two points as NumPy arrays [longitude, latitude]
    # coord1 = np.array([77.343750, 22.593726])
    # coord2 = np.array([86.945801, 23.684774])
    # distance_meters = k.inverse(coord1, coord2)[0, 0]
    ############################################

    distances = np.zeros((len(site_names),target_lon.shape[0],target_lon.shape[0]))
    for sn in range(len(site_names)):
        for i in range(16):
            for j in range(16):
                hrrr_coord = np.array([target_lon[i,j],target_lat[i,j]])
                efm_coord = np.array([site_lons[sn],site_lats[sn]])
                distances[sn,i,j] = k.inverse(hrrr_coord, efm_coord)[0, 0]
                del hrrr_coord, efm_coord
    
    ds = xr.Dataset(data_vars = dict(dist=(['sn','y','x'],distances)),
                                coords=dict(site_names=(['sn'],site_names),
                                            lon=(['y','x'],target_lon),
                                            lat=(['y','x'],target_lat)),
                                attrs=dict(description="This dataset has the fixed distances to the final/target hrrr grid in meters."))
    ds.to_netcdf('./efms2hrrrdist.nc',engine='netcdf4')

def visualize():
    
    ds = xr.open_dataset('./efms2hrrrdist.nc',engine='netcdf4')
    data_np = ds['dist'].values
    for i in range(data_np.shape[0]):
        plt.figure()
        plt.pcolormesh(data_np[i,:,:])
        plt.savefig('dist_%s.png'%i)
        plt.close()

def main():
    print('calculating the distances to the HRRR grid points')

    print('loading the efm locations')
    efm_df = pd.read_excel('EFM_Locations.xlsx')
    print('efm locations loaded successfully')

    print('loading the hrrr grid')
    hrrr_dict = pickle.load(open('../2_MERLIN/grid_dict_2d.pkl','rb'))
    print('hrrr grid loaded successfully')

    print('calculating the distances to the target grid')
    calc_distances_from_efm_2_targets(hrrr_dict = hrrr_dict, efm_df=efm_df)
    print('all distances calculated successfully')

    visualize()


if __name__=='__main__':
    main()