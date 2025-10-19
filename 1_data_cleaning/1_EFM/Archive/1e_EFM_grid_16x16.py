import numpy as np
import cartopy.crs as ccrs
import os
import matplotlib.pyplot as plt
import xarray as xr
import pickle
import pandas as pd
from scipy.interpolate import Rbf
import pygrib
import argparse
from helper import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',type=str,default='2022',help='The year')
    parser.add_argument('--day',type=int,default=180,help='Slurm array index for day of the julian day')
    args = vars(parser.parse_args())
    year = args['year'] #string
    day = args['day'] #int
    month, day = get_day_mo(year,day) #returns string, string
    return year, month, day

def check_runs():

    years = ['2018','2019','2020','2021','2022','2023','2024']
    stats = ['mean','median','std','min','max']

    days = range(1,367)
    bad_days = []
    for year in years:
        for day in days:
            month, day = get_day_mo(year,day)
            efm_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/9_EFM_16x16/%s%s/'
            for stat in stats:
                fname = '%s%s%s_EFM_%s.nc'%(year,month,day,stat)
                if os.path.isfile(efm_dir+fname)==False:
                    if '%s%s%s'%(year,month,day) not in bad_days:
                        bad_days.append('%s%s%s'%(year,month,day))
                        print(day,year)
    print(len(bad_days))


def extract_slurm_env():
    slurm_vars = {key: os.environ[key] for key in os.environ if key.startswith("SLURM_")}
    if slurm_vars:
        print("SLURM Environment Variables:")
        for key, value in slurm_vars.items():
            print(f"{key}: {value}")
    else:
        print("No SLURM environment variables found.")

def stack_ds_slice_2_np(ds_slice=xr.Dataset(),efm_info_dict={}):
    site_lats = []
    site_lons = []
    data_np = []
    k=0
    for key in ds_slice:
        site_lats.append(efm_info_dict[key]['lat'])
        site_lons.append(efm_info_dict[key]['lon'])
        data_np.append(ds_slice[key].values)
    print(type(data_np))
    data_np2 = np.stack(data_np)
    print(type(data_np2))
    return np.array(site_lats), np.array(site_lons), data_np2

def grid2hrrr16x16(stat='mean',year='2022',month='06',day='28'):
    grbs00 = pygrib.open('/ourdisk/hpc/ai2es/datasets/HRRR/HRRR-Subhourly/202206/hrrr-subh_2022062813_f000.grib')
    x_idxs = [1422,1486]
    y_idxs = [176,240]
    hrrr_lat, hrrr_lon = grbs00[1].latlons()
    projection_params = grbs00[1].projparams
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

    ksc_idxs = [y_idxs[0],y_idxs[1],x_idxs[0],x_idxs[1]]
    LC_lats = hrrr_lat[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
    LC_lons = hrrr_lon[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]+360

    row_index_0 = 26
    row_index_1 = 42
    col_index_0 = 23
    col_index_1 = 39
    efm_lat_grid = LC_lats[row_index_0:row_index_1,col_index_0:col_index_1]
    efm_lon_grid = LC_lats[row_index_0:row_index_1,col_index_0:col_index_1]
    
    efm_lat_grid_1d = np.ravel(efm_lat_grid)
    efm_lon_grid_1d = np.ravel(efm_lon_grid)

    loc_df = pd.read_excel('EFM_Locations.xlsx')
    loc_df = loc_df.loc[loc_df['IsActive']==True]
    site_names = loc_df['SiteName'].values
    site_lats = loc_df['Latitude'].values
    site_lons = loc_df['Longitude'].values+360

    data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/7_EFM_qcd_15s_ds/%s%s/'%(year,month)
    fname = '%s%s%s_EFM_%s.nc'%(year,month,day,stat)

    ds = xr.open_dataset(data_dir+fname,engine='netcdf4')
    valid_times = ds['Date_Time'].values
    
    efm_info_dict = {}
    efm_site_lats = []
    efm_site_lons = []

    for key in ds:
        if len(key)==1:
            label = 'FM0'+key
        else:
            label = 'FM'+key
        site_details = loc_df.loc[loc_df['SiteName']==label]
        site_lat = site_details['Latitude'].values[0]
        site_lon = site_details['Longitude'].values[0]+360
        efm_info_dict.update({key:{'label':label,'lat':site_lat,'lon':site_lon}})

    site_lats, site_lons, efm_stack_np = stack_ds_slice_2_np(ds_slice=ds,efm_info_dict=efm_info_dict)
    data_list = []
    time_list = []
    for v,vt in enumerate(valid_times):
        if v%100==0:
            print(v,len(valid_times))
        data2grid = efm_stack_np[:,v]
        
        clean_data = data2grid[~np.isnan(data2grid)]
        clean_lat = site_lats[~np.isnan(data2grid)]
        clean_lon = site_lons[~np.isnan(data2grid)]

        if len(clean_data)>0:
            rbf_ez = Rbf(clean_lon,clean_lat,clean_data,function='gaussian')
            efm_interpolated = rbf_ez(efm_lon_grid_1d,efm_lat_grid_1d)
            efm_interpolated = efm_interpolated.reshape(efm_lat_grid.shape)

            #build the mask when the interpolated values are less than the minimum
            efm_2d_mask = np.ones(efm_interpolated.shape)
            for row in range(efm_interpolated.shape[0]):
                for col in range(efm_interpolated.shape[1]):
                    if efm_interpolated[row,col]<=min(data2grid):
                        efm_2d_mask[row,col]=0.0

            efm_16x16 = efm_interpolated[row_index_0:row_index_1,col_index_0:col_index_1]
            efm_mask_16x16 = efm_2d_mask[row_index_0:row_index_1,col_index_0:col_index_1]
            ds2 = xr.Dataset(data_vars = dict(efm_grid= (["efm_lon","efm_lat"],efm_16x16),
                                                efm_mask = (["efm_lon","efm_lat"],efm_mask_16x16)),
                                coords = dict(efm_lon = (["xx_efm","yy_efm"],efm_lon_grid),
                                            efm_lat = (["xx_efm","yy_efm"],efm_lat_grid)))
            data_list.append(ds2)
            time_list.append(vt)

            del ds2, vt, efm_mask_16x16, efm_16x16, clean_data, clean_lat, clean_lon
            del efm_2d_mask, efm_interpolated, rbf_ez, data2grid
        else:
            pass

    ds3 = xr.concat(data_list, data_vars='all', dim='time')
    ds3 = ds3.assign_coords(time=time_list)
    ds3 = ds3.sortby('time')

    save_dir = '/scratch/bmac87/9_EFM_16x16/%s%s/'%(year,month)
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    fsave = '%s%s%s_EFM_16x16_%s.nc'%(year,month,day,stat)
    ds3.to_netcdf(save_dir+fsave,engine='netcdf4')
    del ds3, save_dir, fsave
    
def make_figures(efm_lon_grid,efm_lat_grid,efm_interpolated,row_index_0,row_index_1,col_index_0,col_index_1):
    save_dir = '/scratch/bmac87/EFM_figures/'

    fig = plt.figure(figsize=(10,8))
    plot_proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1,1,1, projection=plot_proj)
    im = ax.pcolormesh(efm_lon_grid[row_index_0:row_index_1,col_index_0:col_index_1],efm_lat_grid[row_index_0:row_index_1,col_index_0:col_index_1],efm_interpolated[row_index_0:row_index_1,col_index_0:col_index_1],transform=plot_proj,cmap='coolwarm')
    plt.colorbar(im,ax=ax,label = "Potential Gradient (V/m)")
    ax.coastlines(linewidth=2.0,color='grey')
    ax.scatter(site_lons,site_lats,
                c=data2grid,
                cmap='coolwarm',
                transform=plot_proj,
                edgecolors='black',
                s=150)
    plt.savefig('EFM_%s.jpg'%n)
    plt.close()

    fig = plt.figure(figsize=(10,8))
    plot_proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1,1,1, projection=plot_proj)
    im = ax.pcolormesh(efm_lon_grid[row_index_0:row_index_1,col_index_0:col_index_1],efm_lat_grid[row_index_0:row_index_1,col_index_0:col_index_1],efm_2d_mask[row_index_0:row_index_1,col_index_0:col_index_1],transform=plot_proj,cmap='Greys')
    ax.coastlines(linewidth=2.0,color='grey')
    plt.savefig('mask_%s.png'%n)
    plt.close()

def main():

    extract_slurm_env()
    # year, month, day = parse_args()

    # grid2hrrr16x16(stat='mean',year=year,month=month,day=day)
    # grid2hrrr16x16(stat='min',year=year,month=month,day=day)
    # grid2hrrr16x16(stat='max',year=year,month=month,day=day)
    # grid2hrrr16x16(stat='median',year=year,month=month,day=day)
    # grid2hrrr16x16(stat='std',year=year,month=month,day=day)

    check_runs()

if __name__=='__main__':
    main()