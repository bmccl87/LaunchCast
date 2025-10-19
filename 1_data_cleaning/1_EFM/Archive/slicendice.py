import xarray as xr
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse 
from helper import *
import pygrib
import cartopy.crs as ccrs

def extract_slurm_env():
    slurm_vars = {key: os.environ[key] for key in os.environ if key.startswith("SLURM_")}
    if slurm_vars:
        print("SLURM Environment Variables:")
        for key, value in slurm_vars.items():
            print(f"{key}:{value}")
    else:
        print("No SLURM environment variables found.")

def parse_args():
    parser = argparse.ArgumentParser(description='LaunchCast', fromfile_prefix_chars='@')
    parser.add_argument('--exp',type=int,default=0)
    args = parser.parse_args()
    return args.exp

def stack_ds_data(ds=xr.Dataset()):
    data_list = []
    key_list = []
    for k,key in enumerate(ds.keys()):
        data_list.append(ds[key].values)
        key_list.append(key)
    data = np.stack(data_list)
    labels = np.array(key_list)
    return data,labels

def build_figure(stat='mean'):
    print('building the figure for:',stat)
    ds = concat_years(stat=stat)
    valid_times = ds['Date_Time'].values
    x_ticks = [0]
    x_tick_labels = ['2018-01']
    old_year=2018
    old_month=1
    for v,vt in enumerate(valid_times):
        ts = pd.Timestamp(vt)
        year = ts.year
        month = ts.month
        if month!=old_month:
            x_ticks.append(v)
            x_tick_labels.append('%s-%s'%(f"{year:04}",f"{month:02}"))
        old_month=month
        old_year=year
    if stat=='mean':
        vmax=1000
        vmin=-500
        cbar_ticks = [-500,-400,-300,-200,-100,0,100,200,300,400,500,600,700,800,900,1000]
    if stat=='std':
        vmax=75
        vmin=0
        cbar_ticks = [0,25,50,75]
    data_np,y_tick_labels = stack_ds_data(ds)

    plt.figure(figsize=(30,15))
    im=plt.pcolormesh(data_np,vmin=vmin,vmax=vmax,cmap='coolwarm')
    cbar = plt.colorbar(im)
    cbar.set_label('$\\nabla \phi$ (V/m)',size=18)
    cbar.set_ticks(cbar_ticks)
    cbar_ax = cbar.ax
    cbar_ax.tick_params(labelsize=18)
    plt.yticks(np.arange(len(y_tick_labels))+.5,y_tick_labels,fontsize=18)
    plt.ylabel('EFM Site #',fontsize=18)
    n = 3
    plt.xticks(x_ticks[n-1::n],x_tick_labels[n-1::n],fontsize=18,rotation=70)
    plt.xlabel('Time (Year-Month)',fontsize=18)
    plt.title('EFM Values (5-minute): %s'%stat,fontsize=24)
    save_dir='./test_images/'
    plt.savefig('%sEFM_all_%s.png'%(save_dir,stat))
    plt.close()
    ds.close()

def concat_years(stat='mean'):
    print('concatenating:')
    years = ['2018','2019','2020','2021','2022','2023','2024']
    efm_dir='/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/8_EFM_5min_annual_ds/%s/'%(stat)
    ds_list = []
    for year in years:
        print(year)
        ds_list.append(xr.open_dataset(efm_dir+'EFM_%s_%s.nc'%(year,stat),engine='netcdf4'))
    ds3 = xr.concat(ds_list,dim='Date_Time')
    return ds3.sortby('Date_Time',ascending=True)

def build_slices():
    period_start = np.datetime64('2018-01-01T00:00:00.000000000','ns')
    period_end = np.datetime64('2025-01-01T00:00:00.000000000','ns')

    efm_start_times = []
    efm_end_times = []
    ltg_start_times = []
    ltg_end_times = []
    hrrr_valid_times = []

    dt_5min = np.timedelta64(5,'m')
    dt_30min = np.timedelta64(30,'m')
    dt_hour = np.timedelta64(1,'h')

    efm_start_time=period_start-np.timedelta64(1,'h')
    ltg_start_time=period_start

    time_inc=0
    while ltg_start_time<period_end:
        ltg_end_time = ltg_start_time+dt_30min
        ltg_start_times.append(ltg_start_time)
        ltg_end_times.append(ltg_end_time)        
        ltg_start_time=ltg_start_time+dt_5min

        efm_end_time = efm_start_time+dt_hour
        efm_start_times.append(efm_start_time)
        efm_end_times.append(efm_end_time)
        efm_start_time = efm_start_time+dt_5min

        #this does not take into account model latency
        hrrr_ts = pd.Timestamp(efm_end_time)
        hrrr_hr = hrrr_ts.hour
        hrrr_day = hrrr_ts.day
        hrrr_month = hrrr_ts.month
        hrrr_year = hrrr_ts.year
        hrrr_date_str = '%s-%s-%sT%s:00:00.000000000'%(f"{hrrr_year:04}",f"{hrrr_month:02}",f"{hrrr_day:02}",f"{hrrr_hr:02}")
        hrrr_dt64 = np.datetime64(hrrr_date_str)
        hrrr_valid_times.append(hrrr_dt64)

        
        time_inc+=1
    
    slices_dict = {'efm_start_times':efm_start_times,
                    'efm_end_times':efm_end_times,
                    'ltg_start_times':ltg_start_times,
                    'ltg_end_times':ltg_end_times,
                    'hrrr_valid_times':hrrr_valid_times}
    return slices_dict

def get_feature_list(stats=['mean','std'],mins_prior=[],keys=[]):
    features = []
    for stat in stats:
        for min_prior in mins_prior:
            for key in keys:
                feature='%s_%s_%s'%(key,min_prior,stat)
                features.append(feature)
    return features
    
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

    #establish indices for 66x66, for faster processing 
    x_idxs = [1421,1487]
    y_idxs = [175,241]
    ksc_idxs = [y_idxs[0],y_idxs[1],x_idxs[0],x_idxs[1]]

    #get the lat lon grid
    og_hrrr = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_f00/201801/hrrr_2018012115_f000.grib'
    grbs = pygrib.open(og_hrrr)
    hrrr_lat,hrrr_lon = grbs[1].latlons()
    hrrr_lon=hrrr_lon+360

    hrrr_lon = hrrr_lon[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
    hrrr_lat = hrrr_lat[ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]

    print('hrrr_lon.shape,',hrrr_lon.shape)
    print('hrrr_lat.shape,',hrrr_lat.shape)

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
    return hrrr_lon, hrrr_lat, hrrr_xyz, hrrr_proj, hrrr_x_1d, hrrr_y_1d, hrrr_z_1d

def generate_merlin_samples(slices_dict={},start_idx = 0, end_idx=10):
    print('generating merlin samples')
    ltg_start_times = slices_dict['ltg_start_times']
    ltg_end_times = slices_dict['ltg_end_times']
    indexes = np.arange(start_idx,end_idx)

    #load the lightning data 
    cc_df = pickle.load(open('/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Flashes/merlin_cc_df.pkl','rb'))
    cc_ds = cc_df.to_xarray()
    cg_df = pickle.load(open('/ourdisk/hpc/ai2es/datasets/KSC_Weather_Archive/MERLIN_Flashes/merlin_cg_df.pkl','rb'))
    cg_ds = cg_df.to_xarray()
    del cc_df, cg_df

    cc_list = []
    cg_list = []
    time_list = []
    
    hrrr_lon, hrrr_lat, hrrr_xyz, hrrr_proj, hrrr_x_1d, hrrr_y_1d, hrrr_z_1d = get_hrrr_grid()

    for idx in indexes:
        start_time = ltg_start_times[idx]
        end_time = ltg_end_times[idx]
        if idx%25==0:
            print('generating merlin samples:',idx, len(indexes))
        
        #get the sliced lightning information
        cc_grid = np.zeros(hrrr_lon.shape)
        temp_cc = cc_ds.sel(index=slice(start_time,end_time))
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
        temp_cg = cg_ds.sel(index=slice(start_time,end_time))
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

        time_list.append(start_time)
        cc_list.append(cc_grid)
        cg_list.append(cg_grid)
        del cg_grid, cc_grid, temp_cg, temp_cc, cg_lats, cg_lons, cc_lats, cc_lons

    print('stacking the data')
    cc_np = np.stack(cc_list)
    cg_np = np.stack(cg_list)
    ltg_stack_np = np.stack([cc_np,cg_np],axis=-1)

    print('both cc and cg. cc index 0, cg index 1')
    print(ltg_stack_np.shape)
    ltg_stack_np = ltg_stack_np[:,1:65,1:65,:]
    return ltg_stack_np, time_list

def generate_efm_samples(slices_dict={},
                        start_idx = 0, 
                        end_idx=10):

    # get the dataset for all of the years for a specific statistic
    ds_mean = xr.open_dataset('/scratch/bmac87/EFM_mean_all_ds.nc',engine='netcdf4')
    ds_std = xr.open_dataset('/scratch/bmac87/EFM_std_all_ds.nc',engine='netcdf4')

    #get the efm times
    efm_start_times = slices_dict['efm_start_times']
    efm_end_times = slices_dict['efm_end_times']
    
    #declare a timedleta for re-indexing
    dt_5min = np.timedelta64(5,'m')

    all_efm_samples = []
    time_list = []
    indexes = np.arange(start_idx,end_idx)

    
    for i in indexes:
        if i%25==0:
            print('generating efm samples:',i,len(indexes))
        reindex_times = []
        start_time = efm_start_times[i]
        end_time = efm_end_times[i]

        #get the data for the specific slice time
        efm_ds_mean = ds_mean.sel(Date_Time=slice(start_time,end_time))
        efm_ds_std = ds_std.sel(Date_Time=slice(start_time,end_time))

        #reindex the times to account for missing times, and to ensure
        #12 samples within the hour
        while start_time<end_time:
            reindex_times.append(start_time)
            start_time+=dt_5min
        efm_ds_mean = efm_ds_mean.reindex(Date_Time=reindex_times)
        efm_ds_std = efm_ds_std.reindex(Date_Time=reindex_times)
        
        #get the valid times of the data, including nans
        valid_times = sorted(efm_ds_mean['Date_Time'].values)

        #declare the keys for feature labels
        efm_sample_mean = np.full(360,np.nan)
        efm_sample_std = np.full(360,np.nan)

        idx=0
        for v,vt in enumerate(valid_times):
            for k,key in enumerate(keys):
                uno_ds_mean = efm_ds_mean[key]
                uno_ds_std = efm_ds_std[key]

                uno_value_mean = uno_ds_mean.sel(Date_Time=vt).values
                efm_sample_mean[idx] = uno_value_mean

                uno_value_std = uno_ds_std.sel(Date_Time=vt).values
                efm_sample_std[idx] = uno_value_std

                idx+=1
                del uno_value_std, uno_value_mean, uno_ds_mean, uno_ds_std
        del idx

        #store the data
        efm_2stat_sample = np.concatenate([efm_sample_mean,efm_sample_std])
        all_efm_samples.append(efm_2stat_sample)
        time_list.append(end_time)
        
        #memory clean up
        del efm_2stat_sample, efm_sample_mean, efm_sample_std
        efm_ds_mean.close()
        efm_ds_std.close()
    efm_sample_stack_np = np.stack(all_efm_samples)
    del all_efm_samples
    return efm_sample_stack_np, time_list
    
def merge_all_merlin():
    load_dir = '/scratch/bmac87/15_MERLIN_samples_only_pkl/'
    files = sorted(os.listdir(load_dir))
    time_list = []
    efm_list = []

    for f,file in enumerate(files):
        if f>=0:
            merlin_dict = pickle.load(open(load_dir+file,'rb'))
            efm_list.append(merlin_dict['y_reg'])
            time_list.append(merlin_dict['valid_times'])

    time_list_all = np.concatenate(time_list,axis=0)
    y_reg_all = np.concatenate(efm_list,axis=0)
    y_binary_all = np.copy(y_reg_all)
    y_binary_all[y_binary_all>=1] = 1.0
    y_binary_all[y_binary_all<1] = 0.0
    y_sums = np.sum(np.sum(np.sum(y_reg_all,axis=3),axis=2),axis=1)
    ltg_idxs = np.where(y_sums>=1)[0]
    y_reg_all2 = y_reg_all[ltg_idxs,:,:,:]
    y_bin_all2 = y_binary_all[ltg_idxs,:,:,:]
    valid_times = time_list_all[ltg_idxs]
    hrrr_lon, hrrr_lat, hrrr_xyz, hrrr_proj, hrrr_x_1d, hrrr_y_1d, hrrr_z_1d = get_hrrr_grid()

    ds = xr.Dataset({'y_binary':(['valid_times','lon','lat','binary_targets'],y_bin_all2),
                        'y_reg':(['valid_times','lon','lat','regression_targets'],y_reg_all2)},
                    coords={'valid_times':(['tt'],valid_times),
                            'lon':(['xx','yy'],hrrr_lon[1:65,1:65]),
                            'lat':(['xx','yy'],hrrr_lat[1:65,1:65]),
                            'binary_targets':['cc_binary','cg_binary'],
                            'regression_targets':['cc_reg','cg_reg']})
    ds = ds.sortby('valid_times')
    ds.to_netcdf('/scratch/bmac87/LC_ytrue.nc')
    print('ltg data saved to netcdf')



if __name__=='__main__':

    # exp = parse_args()

    #build the list of the dates to be sliced concurrently. 
    #1-hour of EFM data to predict the next 30 minutes of lightning
    slices_dict = build_slices()
    merge_all_merlin()


