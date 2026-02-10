import tensorflow as tf
import numpy as np
import xarray as xr
from Figure_1_MERLIN_Climo import grid_flashes
import pygrib
import cartopy.crs as ccrs
import pickle

def number_of_samples():

    print('generating the numbers for the table')
    for year in ['2021','2022','2023','2024']:
        print(year)
        tfds = tf.data.Dataset.load('/scratch/bmac87/LC_Forecast_HRRR_MRMS_noZdr_GLM_no_EFM_2_MERLIN_%s.tfds'%(year))
        y_true = tf.concat([y for _, y in tfds], axis=0).numpy()
        print(y_true.shape)
        del y_true, tfds

def number_of_flashes():

    #hrrr indices for 64x64 downselection
    # x_idxs = [1422,1486]
    # y_idxs = [176,240]

    # hrrr indices for 66x66 downselection - faster processing for lightning location
    x_idxs = [1421,1487]
    y_idxs = [175,241]

    #get the lat lon grid
    og_hrrr = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_f00/201801/hrrr_2018012115_f000.grib'
    grbs = pygrib.open(og_hrrr)
    
    hrrr_lat,hrrr_lon = grbs[1].latlons()
    print(hrrr_lon.shape)
    hrrr_lat = hrrr_lat[y_idxs[0]:y_idxs[1],x_idxs[0]:x_idxs[1]]
    hrrr_lon = hrrr_lon[y_idxs[0]:y_idxs[1],x_idxs[0]:x_idxs[1]]
    hrrr_lon=hrrr_lon+360
    print(hrrr_lon.shape)
    hrrr_lat_1d = np.ravel(hrrr_lat)
    hrrr_lon_1d = np.ravel(hrrr_lon)

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

    print('creating the plot transform')                                                      
    plot_proj = ccrs.PlateCarree()

    print('creating the hrrr_xy transform')
    hrrr_xyz = hrrr_proj.as_geocentric()

    print('transforming the hrrr_lat/lon to hrrr_xyz2')
    hrrr_xyz2 = hrrr_xyz.transform_points(hrrr_proj,hrrr_lon,hrrr_lat)
    hrrr_x = hrrr_xyz2[:,:,0]
    hrrr_x_1d = np.ravel(hrrr_x)
    
    hrrr_y = hrrr_xyz2[:,:,1]
    hrrr_y_1d = np.ravel(hrrr_y)
    
    hrrr_z = hrrr_xyz2[:,:,2]
    hrrr_z_1d = np.ravel(hrrr_z)

    cg_df = pickle.load(open('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/MERLIN_CG_all_flashes.pkl','rb'))
    cg_df['Date'] = cg_df.index

    cc_df = pickle.load(open('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/MERLIN_CC_all_flashes.pkl','rb'))
    cc_df['Date'] = cc_df.index
    
    flash_num_dict = {}
    for year in ['2021','2022','2023','2024']:
        
        cc_df2 = cc_df[cc_df['Date'].dt.year==int(year)]
        cc_lats = cc_df2['Lat_Decimal'].values
        cc_lons = cc_df2['Lon_Decimal'].values+360

        cg_df2 = cg_df[cg_df['Date'].dt.year==int(year)]
        cg_lats = cg_df2['Lat_Decimal'].values
        cg_lons = cg_df2['Lon_Decimal'].values+360

        cc_grid = np.zeros(hrrr_lon.shape)
        cc_grid = grid_flashes(flash_lats=cc_lats,
                            flash_lons=cc_lons,
                            flash_grid=cc_grid,
                            hrrr_xyz=hrrr_xyz,
                            hrrr_proj=hrrr_proj,
                            hrrr_x_1d=hrrr_x_1d,
                            hrrr_y_1d=hrrr_y_1d,
                            hrrr_z_1d=hrrr_z_1d,
                            hrrr_lon=hrrr_lon)
        print(year,'cc',np.sum(cc_grid[1:64,1:64]))
        flash_num_dict.update({'%s_%s'%(year,'cc'):np.sum(cc_grid[1:64,1:64])})
        del cc_grid

        cg_grid = np.zeros(hrrr_lon.shape)
        cg_grid = grid_flashes(flash_lats=cg_lats,
                            flash_lons=cg_lons,
                            flash_grid=cg_grid,
                            hrrr_xyz=hrrr_xyz,
                            hrrr_proj=hrrr_proj,
                            hrrr_x_1d=hrrr_x_1d,
                            hrrr_y_1d=hrrr_y_1d,
                            hrrr_z_1d=hrrr_z_1d,
                            hrrr_lon=hrrr_lon)
        print(year,'cg',np.sum(cg_grid[1:64,1:64]))
        flash_num_dict.update({'%s_%s'%(year,'cg'):np.sum(cg_grid[1:64,1:64])})
        del cg_grid
    pickle.dump(flash_num_dict,open('./pickles/flash_num_dict.pkl','wb'))

if __name__=='__main__':
    number_of_flashes()