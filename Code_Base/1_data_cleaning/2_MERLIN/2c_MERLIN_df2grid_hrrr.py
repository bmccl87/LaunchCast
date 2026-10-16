import pickle
import sys
import os
import pygrib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import argparse
import numpy as np
import xarray as xr
import glob
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy.core")
import time

def parse_args():
    parser = argparse.ArgumentParser(description='LaunchCast', fromfile_prefix_chars='@')
    parser.add_argument('--exp',type=int,default=0,help='0-500')
    args = parser.parse_args()
    exp = args.exp
    start_idx = exp*72
    end_idx = start_idx+72
    if exp==500:
        end_idx=-1
    return start_idx,end_idx

def grid_flashes(flash_lats,flash_lons,flash_grid,hrrr_x_1d,hrrr_y_1d,hrrr_z_1d,hrrr_xyz,hrrr_proj,hrrr_lon):

    #turn the lat lon of the flashes into an xyz coordingate system
    flash_xyz = hrrr_xyz.transform_points(hrrr_proj,flash_lons,flash_lats)#brilliant!!

    #for each flash, calculate the distance to each hrrr point
    for fl in range(flash_xyz.shape[0]):
        fl_x = flash_xyz[fl,0]
        fl_y = flash_xyz[fl,1]
        fl_z = flash_xyz[fl,2]

        dx = hrrr_x_1d-fl_x
        dy = hrrr_y_1d-fl_y
        dz = hrrr_z_1d-fl_z

        #calculate the distance
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        #find the index of the closest distance
        idx1, idx2 = np.unravel_index(np.argmin(dist, axis=None), hrrr_lon.shape)
        flash_grid[idx1,idx2]+=1

    return flash_grid

def grid_ltg_hrrr():
    start_idx, end_idx = parse_args()
    print(start_idx,end_idx)

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
    
    cc_df = pickle.load(open('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/MERLIN_CC_all_flashes.pkl','rb'))
    cc_ds = cc_df.to_xarray()

    cg_df = pickle.load(open('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/MERLIN_CG_all_flashes.pkl','rb'))
    cg_ds = cg_df.to_xarray()

    LC_slices_df = pickle.load(open('../samples_meta_gt100.pkl','rb'))#created locally
    LC_slices_df = LC_slices_df.loc[LC_slices_df['GT_100_CG']==1.0]
    LC_slices_df = LC_slices_df.loc[LC_slices_df['M00215'].dt.year.isin(np.arange(2021,2025))]
    LC_slices_df = LC_slices_df.loc[LC_slices_df['M00215'].dt.month.isin(np.arange(3,12))]
    num_slices = len(LC_slices_df)
    print(num_slices,' # of total samples')
    merlin_slices_df = LC_slices_df[['M00215', 'M15230', 'M30245','M45260']]
    merlin_slices_df = merlin_slices_df.reset_index(drop=True)
    merlin_prior_slices_df = LC_slices_df[['G60245','G45230','G30215','G15200']]
    merlin_prior_slices_df.reset_index(drop=True)
    merlin_slices_df = merlin_slices_df.iloc[start_idx:end_idx]
    merlin_prior_slices_df = merlin_prior_slices_df.iloc[start_idx:end_idx]
    dt = np.timedelta64(15,'m')
    print(len(merlin_slices_df))

    for t in range(len(merlin_slices_df)):
        if t%10==0:
            print('forecast progress:',t,len(merlin_slices_df))
        times = merlin_slices_df.iloc[t].values

        #set up the file from the first time step
        ts_file = pd.Timestamp(times[0])
        fminute = f"{ts_file.minute:02}"
        fhour = f"{ts_file.hour:02}"
        fday = f"{ts_file.day:02}"
        fmo = f"{ts_file.month:02}"
        fyear = f"{ts_file.year:04}"
        save_dir = '/scratch/bmac87/LaunchCast_scratch/data/2_MERLIN_Forecast_Data/2_MERLIN_grid_gt100/%s%s/'%(fyear,fmo)
        if os.path.isdir(save_dir)==False:
            os.makedirs(save_dir)
        fstr = '%s%s%s%s%s'%(fyear,fmo,fday,fhour,fminute)
        fsave = 'MERLIN_hrrr_%s.nc'%fstr
        if os.path.isfile(save_dir+fsave)==True:
            print(fsave,' already exists')
            continue

        cc_4_list = []
        cg_4_list = []
        for tt in range(len(times)):
            cc_grid = np.zeros(hrrr_lon.shape)
            temp_cc = cc_ds.sel(index=slice(times[tt],times[tt]+dt))
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
            cc_4_list.append(np.expand_dims(cc_grid,axis=0))
            del cc_grid

            cg_grid = np.zeros(hrrr_lon.shape)
            temp_cg = cg_ds.sel(index=slice(times[tt],times[tt]+dt))
            cg_lats = temp_cg['Lat_Decimal'].values
            cg_lons = temp_cg['Lon_Decimal'].values+360
            cg_grid = grid_flashes(flash_lats=cg_lats,
                                    flash_lons=cg_lons,
                                    flash_grid=cg_grid,
                                    hrrr_proj=hrrr_proj,
                                    hrrr_xyz=hrrr_xyz,
                                    hrrr_x_1d=hrrr_x_1d,
                                    hrrr_y_1d=hrrr_y_1d,
                                    hrrr_z_1d=hrrr_z_1d,
                                    hrrr_lon=hrrr_lon)
            cg_4_list.append(np.expand_dims(cg_grid,axis=0))
            del cg_grid

        cc_4_np = np.concatenate(cc_4_list,axis=0)
        cg_4_np = np.concatenate(cg_4_list,axis=0)
        ltg_ds = xr.Dataset(data_vars = dict(cc=(['t','y','x'],cc_4_np.astype(int)),
                                            cg=(['t','y','x'],cg_4_np.astype(int))),
                            coords=dict(time=(['t'],times),
                                        lon=(['y','x'],hrrr_lon),
                                        lat=(['y','x'],hrrr_lat)),
                            attrs=dict(description="MERLIN lightning data on the HRRR grid.  cc is the number of \
                                flashes per hrrr grid. cg is the number of flashes per hrrr grid. this is for the \
                                    hrrr grid."))
                                    
        # print('saving:',save_dir+fsave)
        ltg_ds.to_netcdf(save_dir+fsave,engine='netcdf4')
        # print('saved successfully')
        del ltg_ds, ts_file, cc_4_np, cg_4_np, cc_4_list, cg_4_list

    for t in range(len(merlin_prior_slices_df)):
        if t%10==0:
            print('prior progress:',t,len(merlin_prior_slices_df))
        times = merlin_prior_slices_df.iloc[t].values

        #set up the file from the first time step
        ts_file = pd.Timestamp(times[0])
        fminute = f"{ts_file.minute:02}"
        fhour = f"{ts_file.hour:02}"
        fday = f"{ts_file.day:02}"
        fmo = f"{ts_file.month:02}"
        fyear = f"{ts_file.year:04}"
        save_dir = '/scratch/bmac87/LaunchCast_scratch/data/2_MERLIN_Hour_Prior_Data/2_MERLIN_grid_gt100/%s%s/'%(fyear,fmo)
        if os.path.isdir(save_dir)==False:
            os.makedirs(save_dir)
        fstr = '%s%s%s%s%s'%(fyear,fmo,fday,fhour,fminute)
        fsave = 'MERLIN_hrrr_%s.nc'%fstr
        if os.path.isfile(save_dir+fsave)==True:
            print(fsave,' already exists')
            continue

        cc_4_list = []
        cg_4_list = []
        for tt in range(len(times)):
            cc_grid = np.zeros(hrrr_lon.shape)
            temp_cc = cc_ds.sel(index=slice(times[tt],times[tt]+dt))
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
            cc_4_list.append(np.expand_dims(cc_grid,axis=0))
            del cc_grid

            cg_grid = np.zeros(hrrr_lon.shape)
            temp_cg = cg_ds.sel(index=slice(times[tt],times[tt]+dt))
            cg_lats = temp_cg['Lat_Decimal'].values
            cg_lons = temp_cg['Lon_Decimal'].values+360
            cg_grid = grid_flashes(flash_lats=cg_lats,
                                    flash_lons=cg_lons,
                                    flash_grid=cg_grid,
                                    hrrr_proj=hrrr_proj,
                                    hrrr_xyz=hrrr_xyz,
                                    hrrr_x_1d=hrrr_x_1d,
                                    hrrr_y_1d=hrrr_y_1d,
                                    hrrr_z_1d=hrrr_z_1d,
                                    hrrr_lon=hrrr_lon)
            cg_4_list.append(np.expand_dims(cg_grid,axis=0))
            del cg_grid

        cc_4_np = np.concatenate(cc_4_list,axis=0)
        cg_4_np = np.concatenate(cg_4_list,axis=0)
        ltg_ds = xr.Dataset(data_vars = dict(cc=(['t','y','x'],cc_4_np.astype(int)),
                                            cg=(['t','y','x'],cg_4_np.astype(int))),
                            coords=dict(time=(['t'],times),
                                        lon=(['y','x'],hrrr_lon),
                                        lat=(['y','x'],hrrr_lat)),
                            attrs=dict(description="MERLIN lightning data on the HRRR grid.  cc is the number of \
                                flashes per hrrr grid. cg is the number of flashes per hrrr grid. this is for the \
                                    hrrr grid."))
                                    
        # print('saving:',save_dir+fsave)
        ltg_ds.to_netcdf(save_dir+fsave,engine='netcdf4')
        # print('saved successfully')
        del ltg_ds, ts_file, cc_4_np, cg_4_np, cc_4_list, cg_4_list

def check_files():
    bad_times = []
    LC_slices_df = pickle.load(open('../samples_meta_gt100.pkl','rb'))#created locally
    LC_slices_df = LC_slices_df.loc[LC_slices_df['GT_100_CG']==1.0]
    LC_slices_df = LC_slices_df.loc[LC_slices_df['M00215'].dt.year.isin(np.arange(2021,2025))]
    LC_slices_df = LC_slices_df.loc[LC_slices_df['M00215'].dt.month.isin(np.arange(3,12))]
    num_slices = len(LC_slices_df)
    print(num_slices,' # of total samples')
    merlin_slices_df = LC_slices_df[['M00215', 'M15230', 'M30245','M45260']]
    merlin_slices_df = merlin_slices_df.reset_index(drop=True)
    merlin_prior_slices_df = LC_slices_df[['G60245','G45230','G30215','G15200']]
    merlin_prior_slices_df.reset_index(drop=True)
    dt = np.timedelta64(15,'m')
    print(len(merlin_slices_df))
    for t in range(len(merlin_prior_slices_df)):
        times = merlin_prior_slices_df.iloc[t].values
        ts_file = pd.Timestamp(times[0])
        fminute = f"{ts_file.minute:02}"
        fhour = f"{ts_file.hour:02}"
        fday = f"{ts_file.day:02}"
        fmo = f"{ts_file.month:02}"
        fyear = f"{ts_file.year:04}"
        save_dir = '/scratch/bmac87/LaunchCast_scratch/data/2_MERLIN_Hour_Prior_Data/2_MERLIN_grid_gt100/%s%s/'%(fyear,fmo)
        fstr = '%s%s%s%s%s'%(fyear,fmo,fday,fhour,fminute)
        fsave = 'MERLIN_hrrr_%s.nc'%fstr
        if os.path.isfile(save_dir+fsave)==False:
            print(fsave)
            bad_times.append(times[0])
    print(len(bad_times))
    return bad_times

def visualize():

    month = '06'
    year = '2022'
    day = '30'
    hour = '22'

    data_dir = '/scratch/bmac87/LaunchCast_scratch/data/2_MERLIN_Hour_Prior_Data/2_MERLIN_grid_gt100/%s%s/'%(year,month)
    files = glob.glob(data_dir+'MERLIN_hrrr_%s%s%s%s*.nc'%(year,month,day,hour))
    im_count=0
    for file in files:
        ds = xr.open_dataset(file,engine='netcdf4')
        ds = ds.swap_dims({"t": "time"})
        if "t" in ds.coords:
            ds = ds.drop_vars("t")

        for t in ds['time'].values:
            print(t)
            data = ds.sel(time=t)
            cc_data = data['cc'].values.astype(float)
            cc_data[cc_data==0] = np.nan
            cg_data = data['cg'].values.astype(float)
            cg_data[cg_data==0] = np.nan

            fig = plt.figure(figsize=(20,10))
            ax = fig.add_subplot(1,2,1,projection=ccrs.PlateCarree())
            im = ax.pcolormesh(ds['lon'].values,ds['lat'].values,cc_data,cmap='viridis')
            ax.coastlines()
            cb = plt.colorbar(im,ax=ax,label='Num Flashes')
            ax.set_title('CC Lightning',fontsize=24)

            ax = fig.add_subplot(1,2,2,projection=ccrs.PlateCarree())
            im = ax.pcolormesh(ds['lon'].values,ds['lat'].values,cg_data,cmap='viridis')
            cb = plt.colorbar(im,ax=ax,label='Num Flashes')
            ax.coastlines()
            ax.set_title('CG Lightning',fontsize=24)
            plt.savefig('./test_ltg_images/test_ltg_%s.png'%im_count)
            plt.close()
            im_count+=1

def main():
    # st_t = time.time()
    # grid_ltg_hrrr()
    # e_t = time.time()
    # print(e_t-st_t,' seconds to run one sample')
    # check_files()
    visualize()
    
if __name__=='__main__':
    main()