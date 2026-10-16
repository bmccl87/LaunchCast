import os
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import time
import matplotlib.pyplot as plt
import glob
import time
import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pygrib
from skimage.morphology import convex_hull_image

EFM_MASK_VALUE=-999999.0
ZDR_MASK_VALUE=-99

def create_parser():
    parser = argparse.ArgumentParser(description='Max', fromfile_prefix_chars='@')
    parser.add_argument('--var',type=int,default='2',help='variable to process')
    parser.add_argument('--efm_keys',nargs=31,type=str,default=['FM02', 'FM04', 'FM05', 'FM06', 'FM07', 'FM08', 'FM09', 'FM10', 'FM11', 'FM12', 'FM14', 'FM15', 'FM16', 'FM17', 'FM18', 'FM19', 'FM20', 'FM21', 'FM22', 'FM24', 'FM25', 'FM26', 'FM27', 'FM29', 'FM30', 'FM31', 'FM32', 'FM34'])

    parser.add_argument('--year',type=str,default='2022')
    parser.add_argument('--exp',type=int,default=1)
    args = parser.parse_args()
    years = ['2021','2022','2023','2024']
    year = years[args.exp]
    return year,args

def grid_glm(year='2022',slices_df=pd.DataFrame()):

    load_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/3_GLM_Groups/annual_rc/'
    file = '%s.pkl'%(year)
    glm_df = pickle.load(open(load_dir+file,'rb'))
    glm_ds = glm_df.to_xarray()
    slices_subset = slices_df[slices_df['H_INIT'].dt.year==int(year)]
    glm_headers = ['G60245','G45230','G30215','G15200']
    glm_t_df = slices_subset[glm_headers]
    glm_dt = np.timedelta64(15,'m')-np.timedelta64(1,'ns')

    time_list = []
    area_list = []
    energy_list = []
    for q in range(len(glm_t_df)):
        if q%1000==0:
            print(q,len(glm_t_df))
        if q>=0:
            glm_slices = glm_t_df.iloc[q].values
            glm_energies = np.zeros((4,66,66))
            glm_areas = np.zeros((4,66,66))
            for t in range(4):
                temp_ds=glm_ds.sel(time=slice(glm_slices[t],glm_slices[t]+glm_dt))
                times = temp_ds['time'].values
                if len(times)>0:
                    temp_df = temp_ds.to_dataframe()
                    area_pivot=temp_df.pivot_table(
                        index='r_idx_kdtree',
                        columns='c_idx_kdtree',
                        values='area',
                        aggfunc='max').reindex(index=np.arange(0,66), columns=np.arange(0,66))
                    area_pivot = area_pivot.fillna(0)
                    area_np = area_pivot.to_numpy()
                    glm_areas[t,:,:] = area_np
                    del area_np, area_pivot

                    energy_pivot=temp_df.pivot_table(
                        index='r_idx_kdtree',
                        columns='c_idx_kdtree',
                        values='energy',
                        aggfunc='max').reindex(index=np.arange(0,66), columns=np.arange(0,66))
                    energy_pivot = energy_pivot.fillna(0)
                    energy_np = energy_pivot.to_numpy()
                    glm_energies[t,:,:] = energy_np
                    del energy_np, energy_pivot
                    del temp_df
                del temp_ds

            time_list.append(glm_slices)
            energy_list.append(glm_energies)
            area_list.append(glm_areas)
            del glm_energies
            del glm_areas
    data_dict = {'time_list':time_list,'energy_grid':energy_list,'area_list':area_list}
    pickle.dump(data_dict,open('/scratch/bmac87/glm_v3/GLM_grid_list_%s.pkl'%year,'wb'))

def grid_glm_driver(slices_df=pd.DataFrame(),year='2021'):
    start_clock = time.time()
    grid_glm(year=year,slices_df=slices_df)
    end_clock = time.time()
    diff_clock = (end_clock-start_clock)/60
    print('run time:',diff_clock,'minutes')

def MRMS_annual_concat(variable='Reflectivity_-20C_00.50'):
    print('in MRMS() function',variable)

    radar_dir = '/scratch/bmac87/MRMS_moyr_concat/%s/'%(variable)
    save_dir = '/scratch/bmac87/MRMS_annual_concat/%s/'%(variable)
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    years = ['2021','2022','2023','2024']
    concat_dim='time'

    for y,year in enumerate(years):
        files = sorted(glob.glob(radar_dir+year+'*.nc'))
        ds_list = []
        for f,file in enumerate(files):
            print(file)
            ds = xr.open_dataset(file,engine='netcdf4')
            ds_list.append(ds)
            del ds
        ds_annual = xr.concat(ds_list,dim=concat_dim)
        ds_annual = ds_annual.sortby(concat_dim)
        ds_annual.to_netcdf(save_dir+year+'.nc',engine='netcdf4')
        del ds_annual, ds_list, 

def MRMS_max_time(variable='Reflectivity_-20C_00.50',slices_df=pd.DataFrame(),year='2022'):
    
    print(variable, 'calculating the max time',year)
    radar_dt = np.timedelta64(15,'m')-np.timedelta64(1,'ns')
    temp_slices = slices_df[slices_df['H_INIT'].dt.year==int(year)]
    radar_dir = '/scratch/bmac87/MRMS_annual_concat/%s/'%(variable)
    radar_ds = xr.open_dataset(radar_dir+year+'.nc',engine='netcdf4')
    radar_mask=(radar_ds['time'].dt.minute == 15) & (radar_ds['time'].dt.second == 0)&(radar_ds['time'].dt.microsecond == 0) & (radar_ds['time'].dt.nanosecond == 0)

    lat = radar_ds['lat'].values
    lon = radar_ds['lon'].values

    time_list = []
    sample_list = []

    for q in range(len(temp_slices)):
        if q%100==0:
            print(q,len(temp_slices))

        if q>=0:
            slices = temp_slices.iloc[q]
            one_var_sample = np.zeros((4,64,64))
            try:
                for t in np.arange(1,5):
                    temp_ds = radar_ds.sel(time=slice(slices.iloc[t],slices.iloc[t]+radar_dt))
                    data_np = temp_ds['radar_data'].values
                    one_var_sample[t-1,:,:] = np.max(data_np,axis=0)
                    del temp_ds, data_np
                time_list.append(slices['R60245'])
                ds_store = xr.Dataset(data_vars = dict(radar_data=(['tt','y','x'],one_var_sample)),
                                    coords=dict(prior_times=(['tt'],slices[1:5]),
                                            lon=(['y','x'],lon),
                                            lat=(['y','x'],lat)))
                sample_list.append(ds_store)
                del one_var_sample, slices
            except Exception as e:
                print(e)
                print(slices.iloc[0])

    #build the dataset
    ds_save = xr.concat(sample_list,data_vars='all',dim='sample_time')
    ds_save = ds_save.assign_coords(sample_time=time_list)
    ds_save = ds_save.sortby('sample_time')
    ds_save = ds_save.rename({'radar_data':'%s'%variable})

    print('saving the dataset')
    save_dir = '/scratch/bmac87/radar_data_max_nc_v3/'
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    fname = '%s_%s_max.nc'%(variable,year)
    ds_save.to_netcdf(save_dir+fname,engine='netcdf4')
    print(save_dir+fname, ' saved successfully')
    print(ds_save)
    del ds_save, save_dir, fname

def MRMS_merge_zdr_refl(year='2022'):
    zdr_types = ['low','mid','high']
    zdr_dir = '/scratch/bmac87/radar_data_max_nc_v3/'
    ds_list = []
    for zdr_type in zdr_types:
        fname = '%s_zdr_dB_%s.nc'%(zdr_type,year)
        zdr_ds = xr.open_dataset(zdr_dir+fname,engine='netcdf4')
        ds_list.append(zdr_ds)
        del zdr_ds,fname
    
    var_dir = '/scratch/bmac87/radar_data_max_nc_v3/' 
    variables = [           
                'Reflectivity_-10C_00.50',#0
                'Reflectivity_-15C_00.50',#1
                'Reflectivity_-20C_00.50',#2
                'Reflectivity_-5C_00.50',#3
                'Reflectivity_0C_00.50',#4
                'VII_00.50',#5
                'VIL_00.50'#6
                ]
    
    for variable in variables:
        fname = '%s_%s_max.nc'%(variable,year)
        radar_ds = xr.open_dataset(var_dir+fname,engine='netcdf4')
        ds_list.append(radar_ds)
        del radar_ds,fname
    print('len(ds_list)',len(ds_list))
    
    aligned_ds = xr.align(ds_list[0],
                            ds_list[1],
                            ds_list[2],
                            ds_list[3],
                            ds_list[4],
                            ds_list[5],
                            ds_list[6],
                            ds_list[7],
                            ds_list[8],
                            ds_list[9],
                            join='inner')
    merged_zdr = xr.merge(aligned_ds)
    merged_zdr.to_netcdf('/scratch/bmac87/MRMS_all_%s.nc'%year,engine='netcdf4')
    del merged_zdr, aligned_ds

def MRMS_merge_zdr(year='2022',zdr_variables=[]):

    print('merging the zdrs for:',year)
    ds_list = []
    for variable in zdr_variables:
        radar_dir = '/scratch/bmac87/MRMS_annual_concat/%s/'%variable
        fname = year+'.nc'
        ds = xr.open_dataset(radar_dir+fname,engine='netcdf4')
        ds = ds.rename({'radar_data':variable})
        ds_list.append(ds)
    
    aligned_ds = xr.align(ds_list[0],
                            ds_list[1],
                            ds_list[2],
                            ds_list[3],
                            ds_list[4],
                            ds_list[5],
                            ds_list[6],
                            ds_list[7],
                            ds_list[8],
                            ds_list[9],
                            ds_list[10],
                            ds_list[11],
                            ds_list[12],
                            ds_list[13],
                            ds_list[14],
                            ds_list[15],
                            ds_list[16],
                            ds_list[17],
                            ds_list[18],
                            ds_list[19],
                            ds_list[20],
                            ds_list[21],
                            ds_list[22],
                            ds_list[23],
                            ds_list[24],
                            ds_list[25],
                            ds_list[26],
                            ds_list[27],
                            ds_list[28],
                            ds_list[29],
                            ds_list[30],
                            ds_list[31],
                            join='inner')
    merged_zdr = xr.merge(aligned_ds)
    del aligned_ds
    print(merged_zdr)
    merged_zdr.to_netcdf('/scratch/bmac87/merged_zdr_%s.nc'%(year),engine='netcdf4')
    del merged_zdr

def MRMS_max_max_zdr(year='2022',zdr_variables=[],slices_df=pd.DataFrame(),zdr_type='low'):
    print('maxing over time and elevation for the zdr data, and slicing')
    radar_dir = '/scratch/bmac87/'
    fname = 'merged_zdr_%s.nc'%year

    #has the zdr data for all 
    ds = xr.open_dataset(radar_dir+fname,engine='netcdf4')

    lat = ds['lat'].values
    lon = ds['lon'].values
    mrms_dt = np.timedelta64(15,'m')-np.timedelta64(1,'ns')
    sample_list = []
    time_list = []
    
    for q in range(len(slices_df)):
        if q%100==0:
            print(q,len(slices_df))
            
        if q>=0:
            slices = slices_df.iloc[q]
            time_list.append(slices['R60245'])#save the time for the hour prior
            one_sample = np.zeros((4,64,64))

            for t in np.arange(1,5):#15 minute intervals, index zero is H_init
                try:
                    print('slice times')
                    print(slices.iloc[t],slices.iloc[t]+mrms_dt)
                    temp_ds = ds.sel(time=slice(slices.iloc[t],slices.iloc[t]+mrms_dt))
                    temp_ds = temp_ds[zdr_variables]

                    np_list = []
                    for variable in zdr_variables:
                        temp_np = np.expand_dims(temp_ds[variable].values,axis=-1)#scans,64,64,1
                        temp_np = np.where((temp_np >= -2) & (temp_np <= 5), temp_np, np.nan)#qc
                        np_list.append(temp_np)
                    data_np = np.concatenate(np_list,axis=-1)#scans,64,64,zdr elevations
                    print('data_np.shape should be (2/3, 64, 64, 10)')
                    print(data_np.shape)
                    print('data is finite')
                    print(np.any(np.isfinite(data_np)))
                    print('data_np.max,min')
                    print(np.nanmax(data_np),np.nanmin(data_np))

                    #max the different zdrs to mimic zdr columns
                    #high positive Zdr implies oblate drops, or liquid water
                    #negative values imply vertical/ice 
                    if np.any(np.isfinite(data_np))==True:#if there are some non-nan values
                        #take the max over the elevation angles
                        data_max_elev = np.squeeze(np.nanmax(data_np,axis=-1))#scans,64,64
                        print('data_max_elev.shape')
                        print(data_max_elev.shape)
                        print(np.nanmax(data_max_elev),np.nanmin(data_max_elev))
                        
                        #take the max over time
                        data_max_time = np.squeeze(np.nanmax(data_max_elev,axis=0))#64,64
                        print('data_max_time.shape')
                        print(data_max_time.shape)
                        print(np.nanmax(data_max_time),np.nanmin(data_max_time))

                        #set any nan pixels to zero
                        data_max_time[np.isnan(data_max_time)] = ZDR_MASK_VALUE
                        print('after changing nans to ',ZDR_MASK_VALUE)
                        print(data_max_time.shape)
                        print(np.nanmax(data_max_time),np.nanmin(data_max_time))

                    else:#if all nans (this is fine, the MRMS values are -99 when there's no storms)
                        print('all nans in the MRMS data. dont freak out, this is fine')
                        print('setting the data to zeros')
                        data_max_time = np.ones((4,64,64))*ZDR_MASK_VALUE
                    
                    #store the data
                    one_sample[t-1,:,:]=data_max_time
                    print('one_sample max/min')
                    print(np.nanmax(one_sample),np.nanmin(one_sample))
                    del temp_ds, data_max_elev, data_max_time
                    del data_np, np_list, temp_np

                #in theory, this shouldn't be triggered
                except Exception as e:
                    print(e)
                    print(slices.iloc[t])
                    print(slices.iloc[t]+mrms_dt)
            
            #store the data
            ds_store = xr.Dataset(data_vars = dict(zdr=(['tt','y','x'],one_sample)),
                                    coords=dict(prior_times=(['tt'],slices[1:5]),
                                            lon=(['y','x'],lon),
                                            lat=(['y','x'],lat)))
            sample_list.append(ds_store)
            del ds_store

    print('concatenating the time')
    ds_save = xr.concat(sample_list,data_vars='all',dim='sample_time')
    ds_save = ds_save.assign_coords(sample_time=time_list)
    ds_save = ds_save.sortby('sample_time')
    ds_save = ds_save.rename({'zdr':'%s_zdr_dB'%zdr_type})

    print('saving')
    fsave = '%s_zdr_dB_%s.nc'%(zdr_type,year)
    save_dir='/scratch/bmac87/'
    ds_save.to_netcdf(save_dir+fsave,engine='netcdf4')
    print('ds_save max/min')
    print(ds_save.max())
    print(ds_save.min())
    print(ds_save)
    print('successfully saved:',save_dir+fsave)
    del ds_save, save_dir

def MRMS_driver(samples_df=pd.DataFrame(),year='2021'):

    zdr_variables = [
                'MergedZdr_00.50',#0
                'MergedZdr_00.75',#1
                'MergedZdr_01.00',#2
                'MergedZdr_01.25',#3
                'MergedZdr_01.50',#4
                'MergedZdr_01.75',#5
                'MergedZdr_02.00',#6
                'MergedZdr_02.25',#7
                'MergedZdr_02.50',#8
                'MergedZdr_02.75',#9
                'MergedZdr_03.00',#10
                'MergedZdr_03.50',#11
                'MergedZdr_04.00',#12
                'MergedZdr_04.50',#13
                'MergedZdr_05.00',#14
                'MergedZdr_05.50',#15
                'MergedZdr_06.00',#16
                'MergedZdr_06.50',#17
                'MergedZdr_07.00',#18
                'MergedZdr_07.50',#19
                'MergedZdr_08.00',#20
                'MergedZdr_08.50',#21
                'MergedZdr_09.00',#22
                'MergedZdr_10.00',#23
                'MergedZdr_11.00',#24
                'MergedZdr_12.00',#25
                'MergedZdr_13.00',#26
                'MergedZdr_14.00',#27
                'MergedZdr_15.00',#28
                'MergedZdr_16.00',#29
                'MergedZdr_17.00',#30
                'MergedZdr_18.00',#31
    ]

    #set indices to extract zdr values form low, mid, high elevations
    low_zdr = zdr_variables[0:10]
    mid_zdr = zdr_variables[10:22]
    high_zdr = zdr_variables[22:32]

    variables = [           
                'Reflectivity_-10C_00.50',#0
                'Reflectivity_-15C_00.50',#1
                'Reflectivity_-20C_00.50',#2
                'Reflectivity_-5C_00.50',#3
                'Reflectivity_0C_00.50',#4
                'VII_00.50',#5
                'VIL_00.50'#6
                ]

    # for variable in zdr_variables:
    #     MRMS_annual_concat(variable=variable)

    # for variable in variables:
    #     MRMS_annual_concat(variable=variable)
    
    # for radar_var in variables:
    #     MRMS_max_time(variable=radar_var,slices_df=samples_df,year=year)

    # MRMS_merge_zdr(year=year,zdr_variables=zdr_variables)
    MRMS_max_max_zdr(year=year,zdr_variables=low_zdr,slices_df=samples_df,zdr_type='low')
    MRMS_max_max_zdr(year=year,zdr_variables=mid_zdr,slices_df=samples_df,zdr_type='mid')
    MRMS_max_max_zdr(year=year,zdr_variables=high_zdr,slices_df=samples_df,zdr_type='high')

def MRMS_GLM_merge(year='2022'):
    print('merging the GLM and MRMS data')

    print('loading the MRMS data')
    mrms_dir = '/scratch/bmac87/'
    mrms_fname = 'MRMS_all_%s.nc'%year
    mrms_ds = xr.open_dataset(mrms_dir+mrms_fname,engine='netcdf4')
    print('MRMS data loaded successfully')
    lat = mrms_ds['lat'].values
    lon = mrms_ds['lon'].values

    glm_dir = '/scratch/bmac87/glm_v3/'
    glm_fname = 'GLM_grid_list_%s.pkl'%year
    glm_dict = pickle.load(open(glm_dir+glm_fname,'rb'))#time_list, energy_grid, area_list
    glm_times = glm_dict['time_list']
    glm_energy = glm_dict['energy_grid']
    glm_area = glm_dict['area_list']
    glm_ds_list = []
    time_list = []

    for t in range(len(glm_times)):
        if t%100==0:
            print(t,len(glm_times))
        if t>=0:
            print(glm_times[t])
            uno_energy = glm_energy[t][:,1:65,1:65]
            print(uno_energy.shape)

            uno_area = glm_area[t][:,1:65,1:65]
            print(uno_area.shape)

            uno_ds = xr.Dataset(data_vars = dict(group_area=(['tt','y','x'],uno_area),
                                                group_energy=(['tt','y','x'],uno_energy)),
                                    coords=dict(prior_times=(['tt'],glm_times[t]),
                                            lon=(['y','x'],lon),
                                            lat=(['y','x'],lat)))
            glm_ds_list.append(uno_ds)
            time_list.append(glm_times[t][0])

    glm_ds = xr.concat(glm_ds_list,data_vars='all',dim='sample_time')
    glm_ds = glm_ds.assign_coords(sample_time=time_list)
    glm_ds = glm_ds.sortby('sample_time')
    
    aligned_ds = xr.align(glm_ds,mrms_ds,join='inner')
    merged_ds = xr.merge(aligned_ds)
    merged_ds.to_netcdf('/scratch/bmac87/GLM_MRMS_all_%s.nc'%(year),engine='netcdf4')

def EFM_reindex_mask(year='2022'):

    print('here we go with the EFM data')
    load_dir='/scratch/bmac87/efm_v2/pkl/annual_concat/'
    fload = '%s.nc'%year
    efm_ds = xr.open_dataset(load_dir+fload,engine='netcdf4')
    efm_ds = efm_ds.sortby("time")
    
    print('num time steps before re-indexing')
    print(len(efm_ds['time']))
    efm_ds = efm_ds.sortby("station")
    efm_times = efm_ds['time'].values
    reindex_times = pd.date_range(start='%s-01-01T00:00:00.000000000'%year,end='%s-12-31T00:00:00.000000000'%year,freq="30s")
    print('reindex times')
    print(len(reindex_times))
    print(reindex_times[0:10])
    print('...')
    print(reindex_times[-30:-1])
    efm_ds = efm_ds.reindex(time=reindex_times)
    mask = efm_ds['mask']
    mask = mask.fillna(0).astype(np.float32)
    efm_ds = efm_ds.where(mask != 0, EFM_MASK_VALUE)
    efm_ds = efm_ds.astype(np.float32)
    efm_ds.to_netcdf('/scratch/bmac87/efm_v2/%s_masked_reindexed.nc'%year,engine='netcdf4')

def EFM_generate_samples(slices_df=pd.DataFrame(),year='2022'):
    
    efm_ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/scratch_dump/efm_v3/%s_masked_reindexed.nc'%year,engine='netcdf4')
    efm_dt = np.timedelta64(1,'h')
    t_eps = np.timedelta64(1,'ms')
    efm_start_times = slices_df['E60245'].values
    
    stations = efm_ds['station'].values
    features = ['count_gt_thresh','max_abs', 'mean', 'rms', 'std','var']
    ftr_ds = xr.concat([efm_ds[f] for f in features], dim='efm_stats')
    ftr_ds = ftr_ds.assign_coords(efm_stats=features)
    ftr_ds.name = 'data'
    ftr_ds = ftr_ds.astype(np.float32)
    del efm_ds
    for s,stn in enumerate(stations):
        if s>=0:
            one_ds = ftr_ds.sel(station=stn) 
            time_list=[]
            data_list=[]
            for t,st in enumerate(efm_start_times):
                if t>=0:
                    data_np = one_ds.sel(time=slice(st,st+efm_dt)).data
                    data_np = data_np.swapaxes(0,1)
                    # print(data_np.shape)
                    if data_np.shape[0]!=121:
                        print(st)
                    time_list.append(st)
                    data_list.append(data_np)
            one_efm_samples = np.stack(data_list,axis=0)
            da = xr.DataArray(one_efm_samples,
                dims=('sample_time', 'efm_ts', 'efm_stats'),
                coords={'sample_time': time_list,
                        'efm_ts': np.arange(121),
                        'efm_stats': features},
                name=stn)
            da.to_netcdf('/scratch/bmac87/efm_samples_v3/%s_%s.nc'%(stn,year),engine='netcdf4')
            del da, one_efm_samples, data_list, time_list

def EFM_driver(year='2021',slices_df=pd.DataFrame()):
    
    efm_slices = slices_df[['H_INIT','E60245','E45230','E30215','E15200']]
    # EFM_reindex_mask(year=year)
    EFM_generate_samples(slices_df=efm_slices,year=year)
    # merge_EFM_MRMS_GLM(year='2021')

def merge_EFM_MRMS_GLM(args,year='2022'):

    print('merging the EFM data into the other datasets')
    fm_keys = args.efm_keys
    
    ds_list = []
    for fm in fm_keys:
        print(fm)
        fname = '/scratch/bmac87/efm_samples_v3/%s_%s.nc'%(fm,year)
        temp_ds = xr.open_dataset(fname,engine='netcdf4')
        ds_list.append(temp_ds)
        del temp_ds
    aligned_ds=xr.align(*ds_list, join='inner')
    merged_efm_ds = xr.merge(aligned_ds)

    load_dir = '/scratch/bmac87/'
    fname = 'GLM_MRMS_all_%s.nc'%year

    other_ds = xr.open_dataset(load_dir+fname,engine='netcdf4')
    aligned_ds2 = xr.align(merged_efm_ds,other_ds,join='inner')
    merged_ds2 = xr.merge(aligned_ds2,compat="override")
    del aligned_ds2, other_ds, merged_efm_ds, aligned_ds, ds_list
    merged_ds2.to_netcdf('/scratch/bmac87/EFM_GLM_MRMS_%s.nc'%year,engine='netcdf4')
    print(merged_ds2,flush=True)
    for n in range(4):
        print()
    del merged_ds2

def HRRR_load(year='2022',slices_df=pd.DataFrame()):
    load_dir = '/scratch/bmac87/'
    fname = 'EFM_GLM_MRMS_%s.nc'%year
    ds = xr.open_dataset(load_dir+fname,engine='netcdf4')
    sample_times = ds['sample_time'].values

    slices_df = slices_df[slices_df['H_INIT'].dt.year==int(year)]
    slices_df = slices_df[slices_df['G60245'].isin(sample_times)]#only load model data when the observations are present
    hrrr_times = slices_df['H_INIT'].values
    x_dir = '/scratch/bmac87/HRRR_Subhourly_Downselect_v3/'
    time_list = []
    ds_list = []

    for h,ht in enumerate(hrrr_times):
        if h>=0:
            if h%100==0:
                print(h,len(hrrr_times))
            ts = pd.Timestamp(ht)
            minute=ts.minute
            hour = ts.hour
            day = ts.day
            month = ts.month
            year = ts.year
            hour = f"{hour:02}"
            day = f"{day:02}"
            month = f"{month:02}"
            year = f"{year:04}"
            hrrr_file = '%s%s%s%s.nc'%(year,month,day,hour)
            hrrr_dir = '%s%s%s/'%(x_dir,year,month)
            try:
                hrrr_ds = xr.open_dataset(hrrr_dir+hrrr_file,engine='netcdf4')
                lon=hrrr_ds['lon'].values
                lat=hrrr_ds['lat'].values
                u_10m = hrrr_ds['u_10m'].values
                v_10m = hrrr_ds['v_10m'].values
                div = calc_divergence(u=u_10m,v=v_10m,hrrr_lon=lon,hrrr_lat=lat)
                hrrr_ds['divergence_10m'] =(['ht','x','y'],div)
                temp_ds = hrrr_ds.isel(ht=slice(0, -1),y=slice(1, -1),x=slice(1, -1))
                temp_ds = temp_ds.assign_coords(y=np.arange(64),x=np.arange(64))
                ds_list.append(temp_ds)
                time_list.append(slices_df['G60245'].iloc[h])
                del div, u_10m, v_10m, hrrr_ds, temp_ds
            except Exception as e:
                print('bad HRRR data:',hrrr_dir+hrrr_file)
                print(e)
    
    # hrrr_ds = xr.concat(ds_list,data_vars='all',dim='sample_time')
    # hrrr_ds = hrrr_ds.assign_coords(sample_time=time_list)
    # hrrr_ds = hrrr_ds.sortby('sample_time')

    # aligned_ds = xr.align(hrrr_ds,ds,join='inner')
    # merged_ds = xr.merge(aligned_ds,compat='override')
    # merged_ds.to_netcdf('/scratch/bmac87/All_LC_Inputs_%s.nc'%year,engine='netcdf4')
    # print(merged_ds)
    # print('saved_successfully')

def calc_divergence(u=[],v=[],hrrr_lon=[],hrrr_lat=[]):
    from metpy.calc import wind_direction, divergence,lat_lon_grid_deltas
    from metpy.units import units
    dx,dy = lat_lon_grid_deltas(longitude=hrrr_lon, latitude=hrrr_lat)
    div_array = np.zeros(u.shape)
    for t in range(u.shape[0]):
        u_temp = u[t,:,:]
        v_temp = v[t,:,:]
        div_array[t,:,:] = divergence(u=u_temp,v=v_temp,dx=dx,dy=dy).magnitude
        del u_temp, v_temp
    return div_array

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

def MERLIN_grid(year='2022',slices_df=pd.DataFrame()):

    print('driving the MERLIN')
    #get the lat lon grid
    og_hrrr = '/ourdisk/hpc/ai2es/datasets/HRRR/HRRR_f00/201801/hrrr_2018012115_f000.grib'
    grbs = pygrib.open(og_hrrr)
    slices_dict = pickle.load(open('../LC_hrrr_indexes.pkl','rb'))
    xi = slices_dict['x']
    yi = slices_dict['y']
    hrrr_lat,hrrr_lon = grbs[1].latlons()
    print(hrrr_lon.shape)
    hrrr_lat = hrrr_lat[(yi.start-1):(yi.stop+1),(xi.start-1):(xi.stop+1)]
    hrrr_lon = hrrr_lon[(yi.start-1):(yi.stop+1),(xi.start-1):(xi.stop+1)]
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

    merlin_fcst_df = slices_df[['H_INIT','M00215','M15230','M30245','M45260','G60245']]
    merlin_fcst_df = merlin_fcst_df.reset_index(drop=True)
    merlin_dt = np.timedelta64(15,'m')-np.timedelta64(1,'ns')
    sample_time = []
    ds_list = []

    for q in range(len(merlin_fcst_df)):
        if q%100==0:
            print(q,len(merlin_fcst_df))
        if q>=0:
            slices = merlin_fcst_df.iloc[q]
            fcst_slices = slices[['M00215','M15230','M30245','M45260']]
            
            cc_4_list = []
            cg_4_list = []
            for t in range(4):

                temp_cc = cc_ds.sel(index=slice(fcst_slices.iloc[t],fcst_slices.iloc[t]+merlin_dt))
                cc_lats = temp_cc['Lat_Decimal'].values
                cc_lons = temp_cc['Lon_Decimal'].values+360
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
                cc_4_list.append(np.expand_dims(cc_grid,axis=0))
                del cc_grid,temp_cc,cc_lats,cc_lons

                temp_cg = cg_ds.sel(index=slice(fcst_slices.iloc[t],fcst_slices.iloc[t]+merlin_dt))
                cg_lats = temp_cg['Lat_Decimal'].values
                cg_lons = temp_cg['Lon_Decimal'].values+360
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
                cg_4_list.append(np.expand_dims(cg_grid,axis=0))
                del cg_grid,cg_lats,cg_lons,temp_cg

            cc_4_np = np.concatenate(cc_4_list,axis=0)
            cg_4_np = np.concatenate(cg_4_list,axis=0)

            ds = xr.Dataset(data_vars = dict(cc=(['merlin_t','y','x'],cc_4_np[:,1:65,1:65]),
                                            cg=(['merlin_t','y','x'],cg_4_np[:,1:65,1:65])),
                            coords=dict(merlin_fcst_time=(['merlin_t'],fcst_slices),
                                        lon=(['y','x'],hrrr_lon[1:65,1:65]),
                                        lat=(['y','x'],hrrr_lat[1:65,1:65])))
            sample_time.append(slices['G60245'])
            ds_list.append(ds)
            del ds, cc_4_np, cg_4_np

    ds_save = xr.concat(ds_list,data_vars='all',dim='sample_time')
    ds_save = ds_save.assign_coords(sample_time=sample_time)
    ds_save = ds_save.sortby('sample_time')
    ds_save.to_netcdf('/scratch/bmac87/MERLIN_Forecast_%s.nc'%year,engine='netcdf4')

def concat_target_mask_months(year='2022',variable='target_mask'):
    
    hrrr_grid = pickle.load(open('../0_HRRR/hrrr_og_grid.pkl','rb'))
    hrrr_lat = hrrr_grid['lat']
    hrrr_lon = hrrr_grid['lon_360']
    print(hrrr_lat.shape)
    
    slices_dict = pickle.load(open('../LC_hrrr_indexes.pkl','rb'))
    xi = slices_dict['x']
    yi = slices_dict['y']
    lon = hrrr_lon[yi.start:yi.stop,xi.start:xi.stop]
    lat = hrrr_lat[yi.start:yi.stop,xi.start:xi.stop]
    print(lon.shape)
    for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
        data_list = []
        time_list = []
        data_dir = '/scratch/bmac87/MRMS_on_HRRR_grid/%s/%s%s/'%(variable,year,month)
        if os.path.isdir(data_dir)==True:
            files = sorted(os.listdir(data_dir))
            for f,file in enumerate(files):
                temp_data = pickle.load(open(data_dir+file,'rb'))
                ds = xr.Dataset(data_vars = dict(radar_data=(['y','x'],temp_data['data'])),
                                    coords=dict(lon=(['y','x'],lon),
                                                lat=(['y','x'],lat)))
                data_list.append(ds)
                time_list.append(temp_data['time'])
                del temp_data, ds
        
        if len(data_list)>0:
            ds1 = xr.concat(data_list, data_vars='all', dim='time')
            ds1 = ds1.assign_coords(time=time_list)
            ds1 = ds1.sortby('time')
            fsave = '%s%s.nc'%(year,month)
            save_dir = '/scratch/bmac87/MRMS_moyr_concat/%s/'%variable
            if os.path.isdir(save_dir)==False:
                os.makedirs(save_dir)
            ds1.to_netcdf(save_dir+fsave,engine='netcdf4')
            del ds1, data_list, time_list

def MERLIN_targets(year='2022',slices_df=pd.DataFrame()):

    radar_dir = '/scratch/bmac87/MRMS_annual_concat/target_mask/'
    radar_fname = '%s.nc'%(year)
    radar_ds = xr.open_dataset(radar_dir+radar_fname,engine='netcdf4')
    print(radar_ds)
    
    merlin_dir = '/scratch/bmac87/'  
    merlin_fname = 'MERLIN_Forecast_%s.nc'%(year)
    merlin_ds = xr.open_dataset(merlin_dir+merlin_fname,engine='netcdf4')
    print(merlin_ds)

    sample_times = merlin_ds['sample_time'].values
    merlin_start_times = sample_times+np.timedelta64(1,'h')

    dt = np.timedelta64(15,'m')-np.timedelta64(1,'ns')
    
    #only predict one lightning image. not IC and CG. 
    total_ltg = merlin_ds['cc'].values + merlin_ds['cg'].values
    total_ltg[total_ltg>0] = 1.0
    total_ltg[total_ltg==0] = 0.0

    qcd_sample_times = []
    final_target_list = []
    for i in range(len(merlin_start_times)):
        if i%10==0:
            print(i,len(sample_times))
        ltg_target_one_sample=[]
        try:
            start_time = merlin_start_times[i]
            for t in range(4):
                temp_ltg_data = total_ltg[i,t,:,:]
                radar_slice = radar_ds.sel(time=slice(start_time,start_time+dt))
                radar_slice_np = radar_slice['radar_data'].values
                radar_max_np = np.max(radar_slice_np,axis=0)
                convex_hull = convex_hull_image(temp_ltg_data)
                if convex_hull is None:
                    convex_hull=np.zeros((64,64))     
                ltg_target_one_sample.append(convex_hull*(radar_max_np>=30))
                start_time+=np.timedelta64(15,'m')
                del temp_ltg_data, convex_hull, radar_slice, radar_slice_np
            del start_time
            uno_sample = np.stack(ltg_target_one_sample)
            final_target_list.append(uno_sample)
            qcd_sample_times.append(sample_times[i])

        except Exception as e:
            print(e)
            print(i)
            continue
    total_targets=np.stack(final_target_list,axis=0)
    print(total_targets.shape)

    da = xr.DataArray(total_targets.astype(np.float64),
                dims=('sample_time', 'merlin_t', 'xx','yy'),
                coords={'sample_time': qcd_sample_times,
                        'merlin_t': np.arange(4),
                        'xx': np.arange(64),
                        'yy':np.arange(64)},
                name='targets')
    
    aligned_ds2 = xr.align(merlin_ds,da,join='inner')
    merged_ds2 = xr.merge(aligned_ds2,compat="override")
    print(merged_ds2)

    plt.figure()
    plt.hist(np.ravel(total_targets.astype(np.float64)))
    plt.savefig('%s_targets.png'%year)
    plt.close()

    merged_ds2.to_netcdf('/scratch/bmac87/%s_MERLIN_Forecast_w_targets.nc'%year,engine='netcdf4')

def MERLIN_merge_inputs_targets(year='2022'):
    print('merging the targets with the inputs')
    input_ds = xr.open_dataset('/scratch/bmac87/All_LC_Inputs_%s.nc'%year)
    merlin_ds = xr.open_dataset('/scratch/bmac87/%s_MERLIN_Forecast_w_targets.nc'%year)
    aligned_ds = xr.align(input_ds,merlin_ds,join='inner')
    merged_ds = xr.merge(aligned_ds,compat='override')
    merged_ds.to_netcdf('/scratch/bmac87/LC_Forecast_%s.nc'%(year))
    del merged_ds

def zdr_z_mask_log_glm(year='2021'):
    ds = xr.open_dataset('/scratch/bmac87/LC_Forecast_%s.nc'%(year),engine='netcdf4')
    z_np = ds['Reflectivity_0C_00.50'].values

    print('z_np.shape')
    print(z_np.shape)
    z_mask = z_np>0.0

    print('z_mask')
    print(np.nanmax(z_mask),np.nanmin(z_mask))
    print()

    for label in ['low','mid','high']:
        print(label)
        print('zdr_np')
        zdr_np = ds['%s_zdr_dB'%label].values
        print(np.nanmax(zdr_np),np.nanmin(zdr_np))
        zdr_np[~z_mask] = ZDR_MASK_VALUE
        zdr_mask = (~(zdr_np==ZDR_MASK_VALUE)).astype(np.float32)
        print('zdr_mask')
        print(np.nanmax(zdr_mask),np.nanmin(zdr_mask))
        ds['%s_zdr_dB'%label] = (('sample_time','tt','y','x'),zdr_np)
        ds['%s_zdr_mask'%label] = (('sample_time','tt','y','x'),zdr_mask)
        print(np.nanmax(zdr_np),np.nanmin(zdr_np))
        del zdr_np, zdr_mask

    ga = ds['group_area'].values
    ds['group_area_log'] = (('sample_time','tt','y','x'),np.log1p(ga))

    ge = ds['group_energy'].values
    ds['group_energy_log'] = (('sample_time','tt','y','x'),np.log1p(ge))

    VII = ds['VII_00.50'].values
    VII[VII<=0.0]=0.0
    ds['VII_00.50_log'] = (('sample_time','tt','y','x'),np.log1p(VII))
    ds = ds.astype(np.float32)
    ds.to_netcdf('/scratch/bmac87/LC_Forecast_Final_log1p_GLM_VII_%s.nc'%(year))
    print(ds)
    print(ds.max())
    print(ds.min())
    print('ds saved SUCCESSFULLY')
    ds.close()

def concat_total_ds():

    years = ['2021','2022','2023','2024']
    nc_dir = '/scratch/bmac87/'
    ds_list = []
    for year in years:
        fname = 'LC_Forecast_Final_log1p_GLM_VII_%s.nc'%year
        ds = xr.open_dataset(nc_dir+fname,engine='netcdf4')
        ds_list.append(ds)
        del ds
    total_ds = xr.concat(ds_list,dim='sample_time')
    total_ds = total_ds.astype(np.float32)
    total_ds.to_netcdf('/scratch/bmac87/LC_Forecast_total_ds.nc',engine='netcdf4')
    print(total_ds)

def dev3(args,year='2021'):
    
    print('working on the new samples')
    samples_df = pickle.load(open('./%s_samples_w_ltg_counts_w_efm_flag.pkl'%year,'rb'))
    samples_df = samples_df.loc[samples_df['H_INIT'].dt.month.isin(range(3,12))]
    samples_df = samples_df.loc[samples_df['withinEFM_5nm']==1]
    # MRMS_driver(samples_df=samples_df,year=year)
    # grid_glm_driver(slices_df=samples_df,year=year)
    # MRMS_merge_zdr_refl(year=year)
    # MRMS_GLM_merge(year=year)
    # EFM_driver(year=year,slices_df=samples_df)
    # merge_EFM_MRMS_GLM(args=args,year=year)
    # HRRR_load(year=year,slices_df=samples_df)
    # concat_target_mask_months(year=year,variable='target_mask')
    # MRMS_annual_concat(variable='target_mask')
    # MERLIN_grid(year=year,slices_df=samples_df)
    # MERLIN_targets(year=year,slices_df=samples_df)
    # MERLIN_merge_inputs_targets(year=year)
    # zdr_z_mask_log_glm(year=year)
    concat_total_ds()



if __name__=='__main__':
    start_clock = time.time()
    year,args=create_parser()
    dev3(args=args,year=year)
    end_clock = time.time()
    print((end_clock-start_clock)/60,' minutes to run')
    print('COMPLETE LC_merge_data.py')