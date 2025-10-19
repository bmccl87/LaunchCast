import pickle
from helper import *
from LC_util import *
import sys
import os
import pygrib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='LaunchCast', fromfile_prefix_chars='@')
    parser.add_argument('--exp',type=int,default=0)
    args = parser.parse_args()
    return args

def grid_flashes(flash_lats,flash_lons,flash_grid,hrrr_x_1d,hrrr_y_1d,hrrr_z_1d,hrrr_xyz,hrrr_proj,hrrr_lon):
    flash_xyz = hrrr_xyz.transform_points(hrrr_proj,flash_lons,flash_lats)#brilliant!!
    for fl in range(flash_xyz.shape[0]):
        fl_x = flash_xyz[fl,0]
        fl_y = flash_xyz[fl,1]
        fl_z = flash_xyz[fl,2]

        dx = hrrr_x_1d-fl_x
        dy = hrrr_y_1d-fl_y
        dz = hrrr_z_1d-fl_z

        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        idx1, idx2 = np.unravel_index(np.argmin(dist, axis=None), hrrr_lon.shape)
        flash_grid[idx1,idx2]+=1
    return flash_grid

def get_hrrr_grid():
    args = parse_args()

    #get the lat lon grid
    og_hrrr = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_f00/201801/hrrr_2018012115_f000.grib'
    grbs = pygrib.open(og_hrrr)
    
    hrrr_lat,hrrr_lon = grbs[1].latlons()
    hrrr_lon=hrrr_lon+360
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
    
    dt = np.timedelta64(1,'h')
    if args.exp==0:
        start_time = np.datetime64('2018-01-01T00:00:00')
        end_time = np.datetime64('2019-01-01T00:00:00')
    elif args.exp==1:
        start_time = np.datetime64('2019-01-01T00:00:00')
        end_time = np.datetime64('2020-01-01T00:00:00')
    elif args.exp==2:
        start_time = np.datetime64('2020-01-01T00:00:00')
        end_time = np.datetime64('2021-01-01T00:00:00')
    elif args.exp==3:
        start_time = np.datetime64('2021-01-01T00:00:00')
        end_time = np.datetime64('2022-01-01T00:00:00')
    elif args.exp==4:
        start_time = np.datetime64('2022-01-01T00:00:00')
        end_time = np.datetime64('2023-01-01T00:00:00')
    elif args.exp==5:
        start_time = np.datetime64('2023-01-01T00:00:00')
        end_time = np.datetime64('2024-01-01T00:00:00')
    else:
        start_time = np.datetime64('2024-01-01T00:00:00')
        end_time = np.datetime64('2025-01-01T00:00:00')

    slice_times = []
    while start_time<=end_time:
        slice_times.append(start_time)
        start_time+=dt

    cc_df = pickle.load(open('/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Flashes/merlin_cc_df.pkl','rb'))
    cc_ds = cc_df.to_xarray()
    cg_df = pickle.load(open('/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Flashes/merlin_cg_df.pkl','rb'))
    cg_ds = cg_df.to_xarray()
    cc_list = []
    cg_list = []
    time_list = []
    f=0

    for t in range(len(slice_times)-1):
        if t>=0:
            cc_grid = np.zeros(hrrr_lon.shape)
            print('cc:',t,len(slice_times))
            temp_cc = cc_ds.sel(index=slice(slice_times[t],slice_times[t+1]))
            cc_lats = temp_cc['Lat_Decimal'].values
            cc_lons = temp_cc['Lon_Decimal'].values+360
            cc_grid = grid_flashes(flash_lats=cc_lats,
                                    flash_lons=cc_lons,
                                    flash_grid=cc_grid,
                                    hrrr_xyz=hrrr_xyz,
                                    hrrr_proj=hrrr_proj,
                                    hrrr_x_1d=hrrr_x_1d,
                                    hrrr_y_1d=hrrr_y_1d,
                                    hrrr_z_1d=hrrr_z_1d,
                                    hrrr_lon=hrrr_lon)
    
            cg_grid = np.zeros(hrrr_lon.shape)
            temp_cg = cg_ds.sel(index=slice(slice_times[t],slice_times[t+1]))
            cg_lats = temp_cg['Lat'].values
            cg_lons = temp_cg['Lon'].values+360
            cg_grid = grid_flashes(flash_lats=cg_lats,
                                    flash_lons=cg_lons,
                                    flash_grid=cg_grid,
                                    hrrr_proj=hrrr_proj,
                                    hrrr_xyz=hrrr_xyz,
                                    hrrr_x_1d=hrrr_x_1d,
                                    hrrr_y_1d=hrrr_y_1d,
                                    hrrr_z_1d=hrrr_z_1d,
                                    hrrr_lon=hrrr_lon)

            print('max_cg_grid:',np.max(np.max(cg_grid)))
            ltg_ds = xr.Dataset(data_vars = dict(cc=(['y','x'],cc_grid.astype(int)),
                                                cg=(['y','x'],cg_grid.astype(int))),
                                coords=dict(time=slice_times[t],
                                            lon=(['y','x'],hrrr_lon),
                                            lat=(['y','x'],hrrr_lat)),
                                attrs=dict(description="MERLIN lightning data on the HRRR grid.  cc is the number of \
                                    flashes per hrrr grid. cg is the number of flashes per hrrr grid. this is for the \
                                        hrrr grid. This is one hour temporal resolution, with the lightning binned over\
                                            the next hour. Thus a time of 06-30-2022 01Z has lightning valid between 01-02Z."))

            
            ts_file = pd.Timestamp(slice_times[t])
            fhour = f"{ts_file.hour:02}"
            fday = f"{ts_file.day:02}"
            fmo = f"{ts_file.month:02}"
            fyear = f"{ts_file.year:04}"
            save_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2a_MERLIN_HRRR_grid/%s%s/'%(fyear,fmo)
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)
            fstr = '%s%s%s%s'%(fyear,fmo,fday,fhour)
            fsave = 'MERLIN_hrrr_%s.nc'%fstr
            print('saving:',save_dir+fsave)
            ltg_ds.to_netcdf(save_dir+fsave,engine='netcdf4')
            print('saved successfully')
            del cg_grid, cc_grid, ltg_ds

def main():
    get_hrrr_grid()
    
if __name__=='__main__':
    main()