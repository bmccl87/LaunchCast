import os
import pandas as pd
import shutil
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import glob
import argparse
import time
from cartopy.geodesic import Geodesic
import xarray as xr

def calc_distances_from_efm_ctr_2_targets(target_lat=[],target_lon=[], efm_df=pd.DataFrame()):
    """
    This function calculates the distance between each HRRR data point, and the EFM locations.
    """
    print('calculating the distances to the EFM sites')
    site_names = efm_df['SiteName'].values
    site_lats = efm_df['Latitude'].values
    site_lons = efm_df['Longitude'].values
    print(efm_df)
    print('n_sites')
    print(len(site_names))
    k = Geodesic()

    ny, nx = target_lon.shape
    n_sites = len(site_names)

    distances = np.zeros((ny, nx, n_sites))
    proximity = np.zeros((ny, nx, n_sites))

    #calculate the distance to each target site
    for e in range(n_sites):
        efm_coord = np.array([site_lons[e], site_lats[e]])
        for i in range(ny):
            for j in range(nx):
                hrrr_coord = np.array([target_lon[i, j], target_lat[i, j]])

                # distance in meters
                distances[i, j, e] = k.inverse(hrrr_coord, efm_coord)[0, 0]
    
    #calculate the proximity
    for e in range(n_sites):
        temp_distance = distances[:, :, e]
        norm_distance = ((temp_distance - np.nanmin(temp_distance)) /
            (np.nanmax(temp_distance) - np.nanmin(temp_distance)))
        proximity[:, :, e] = 1 - norm_distance
    
    NM_TO_M = 1852
    threshold_5nm = 5 * NM_TO_M
    mask_5nm_any = np.any(distances <= threshold_5nm, axis=2).astype(np.float32)
    min_distance = np.squeeze(np.min(distances,axis=-1))
    max_proximity = np.squeeze(np.max(proximity,axis=-1))
    
    ds = xr.Dataset(data_vars = dict(min_dist_1d=(['y','x'],min_distance),
                                    proximity_1d=(['y','x'],max_proximity),
                                    distances=(['y','x','e'],distances),
                                    proximities=(['y','x','e'],proximity),
                                    mask_5nm=(['y','x'],mask_5nm_any)),
                                coords=dict(lon=(['y','x'],target_lon),
                                            lat=(['y','x'],target_lat),
                                            site_names=(['e'],site_names)),
                                attrs=dict(description="This dataset has the fixed distances to the final/target hrrr grid in meters."))
    print(ds)
    ds.to_netcdf('./efm_distance_proximity.nc',engine='netcdf4')
    ds.to_netcdf('../../2_model_training/efm_distance_proximity.nc',engine='netcdf4')
    ds.to_netcdf('../../3_model_analysis/efm_distance_proximity.nc',engine='netcdf4')
    ds.to_netcdf('../../4_Paper_Figures/efm_distance_proximity.nc',engine='netcdf4')
    
def visualize():
    
    ds = xr.open_dataset('./min_efm_dist.nc',engine='netcdf4')
    data_np = ds['dist_norm'].values
    plt.figure()
    plt.pcolormesh(data_np)
    plt.savefig('min_dist_norm.png')
    plt.close()

    data_np = ds['proximity'].values
    plt.figure()
    plt.pcolormesh(data_np)
    plt.savefig('proximity.png')
    plt.close()

def main():
    # print('calculating the distances to the HRRR grid points')

    print('loading the efm locations')
    efm_df = pd.read_pickle(open('./efm_loc_df.pkl','rb'))
    print('efm locations loaded successfully')

    print('loading the hrrr grid')
    hrrr_dict = pickle.load(open('../0_HRRR/hrrr_og_grid.pkl','rb'))
    slices_dict = pickle.load(open('../LC_hrrr_indexes.pkl','rb'))

    xi = slices_dict['x']
    yi = slices_dict['y']
    hrrr_lat = hrrr_dict['lat']
    hrrr_lat = hrrr_lat[yi.start:yi.stop,xi.start:xi.stop]
    hrrr_lat = hrrr_lat[24:40,24:40]

    hrrr_lon = hrrr_dict['lon']
    hrrr_lon = hrrr_lon[yi.start:yi.stop,xi.start:xi.stop]
    hrrr_lon = hrrr_lon[24:40,24:40]

    print('hrrr grid loaded successfully')

    print('calculating the distances to the target grid')
    calc_distances_from_efm_ctr_2_targets(target_lat=hrrr_lat,target_lon=hrrr_lon, efm_df=efm_df)
    print('all distances calculated successfully')

    # visualize()

if __name__=='__main__':
    main()