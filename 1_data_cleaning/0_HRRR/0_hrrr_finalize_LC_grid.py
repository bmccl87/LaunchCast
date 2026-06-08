import pygrib
import os
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt

def main():

    #for reference
    #set the extent of LaunchCast bounds over KSC
    west_extent = -81.61
    west_extent = west_extent+360
    print('west_extent',west_extent)

    east_extent = -79.82
    east_extent = east_extent+360
    print('east_extent',east_extent)

    south_extent = 27.7
    north_extent = 29.3

    print('printing the gribs for hrrr')
    file = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_f00/202306/hrrr_2023061519_f000.grib'
    grbs = pygrib.open(file)
    hrrr_lat, hrrr_lon = grbs[1].latlons()
    hrrr_lon = hrrr_lon+360
    print(hrrr_lat.shape)
    print('hrrr_lat>=south_extent')
    y_idx1 = np.where((hrrr_lat>=south_extent))
    print(y_idx1[0])
    print(y_idx1[1])

    print('hrrr_lat<=north_extent')
    y_idx2 = np.where((hrrr_lat<=north_extent))
    print(y_idx2[0])
    print(y_idx2[1])
    
    print('hrrr_lon<=east_extent')
    x_idx1 = np.where((hrrr_lon<=east_extent))
    print(x_idx1[0])
    print(x_idx1[1])

    print('hrrr_lon>=west_extent')
    x_idx2 = np.where((hrrr_lon>=west_extent))
    print(x_idx2[0])
    print(x_idx2[1])
    
    projection_params = grbs[1].projparams
    proj_a = projection_params['a']
    proj_b = projection_params['b']
    lon_0 = projection_params['lon_0']
    lat_0 = projection_params['lat_0']
    lat_parallel = projection_params['lat_1']

    print('creating the hrrr ccrs projection')
    hrrr_proj = ccrs.LambertConformal(central_longitude=lon_0, 
                                        central_latitude=lat_0,
                                        globe=ccrs.Globe(semimajor_axis=proj_a,
                                                            semiminor_axis=proj_b))

    x_idxs = [1422,1486]
    y_idxs = [176,240]
    lats = hrrr_lat[y_idxs[0]:y_idxs[1],x_idxs[0]:x_idxs[1]]
    lons = hrrr_lon[y_idxs[0]:y_idxs[1],x_idxs[0]:x_idxs[1]]

    print(lats.shape)
    print(lons.shape)

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1, projection=hrrr_proj)
    ax.pcolormesh(lons,lats,lats,transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.savefig('testhrrr.png')
    plt.close()
    


if __name__=='__main__':
    main()