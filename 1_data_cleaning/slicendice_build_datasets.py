import numpy as np
import pandas as pd
import pickle
import xarray as xr
import os
import sys
import copy 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def efm_fill_image(ds = xr.Dataset(),start_time = np.datetime64(), end_time = np.datetime64(), top_24_sites = []):
    #create the time slice
    time_slice = slice(start_time,end_time)

    #get the data between the times
    ds_slice = ds.sel(index=time_slice)

    #build the best time list ever for resampling
    resample_times = []
    dt_5s = np.timedelta64(5,'s')
    while (start_time<=end_time):
        resample_times.append(start_time)
        start_time=start_time+dt_5s
    
    #resample then ffill any missing values
    ds_slice = ds_slice.reindex(index=resample_times)
    ds_slice = ds_slice.ffill(dim="index")

    #convert the data to np
    stack = []
    for site in top_24_sites:
        stack.append(ds_slice[site].values)
    slice_np = np.stack(stack,axis=0)

    #take the mean across the time dimension
    time_mean = np.nanmean(slice_np,axis=0)

    #assign any missing values across the time dimension to the time average
    for s,site in enumerate(top_24_sites):
        nan_mask = np.isnan(slice_np[s,:])
        slice_np[s,nan_mask] = time_mean[nan_mask]
    return slice_np

def build_hrrr_list(year='2018'):
    hrrr_start_time = np.datetime64('%s-03-01T00:00:00.000000000'%year,'ns')
    hrrr_end_time = np.datetime64('%s-12-01T00:00:00.000000000'%year,'ns')
    hrrr_dt = np.timedelta64(1,'h')
    
    time_list = []
    while hrrr_start_time<hrrr_end_time:
        time_list.append(hrrr_start_time)
        hrrr_start_time+=hrrr_dt
    return time_list

def build_efm_list(hrrr_list = []):
    efm_1_dt = np.timedelta64(1,'h')
    efm_2_dt = np.timedelta64(45,'m')
    efm_3_dt = np.timedelta64(30,'m')
    efm_4_dt = np.timedelta64(15,'m')

    efm_res_dt = np.timedelta64(15,'m')
    hrrr_list2 = []
    efm_1_start_times = []
    efm_2_start_times = []
    efm_3_start_times = []
    efm_4_start_times = []
    
    for h,hrrr_time in enumerate(hrrr_list):#for each hrrr model run
        efm_1_start_time = hrrr_time - efm_1_dt
        efm_2_start_time = hrrr_time - efm_2_dt
        efm_3_start_time = hrrr_time - efm_3_dt
        efm_4_start_time = hrrr_time - efm_4_dt

        for i in range(4):#increment every 15 minutes over 1-hour
            hrrr_list2.append(hrrr_time)

            efm_1_start_times.append(efm_1_start_time)
            efm_1_start_time+=efm_res_dt

            efm_2_start_times.append(efm_2_start_time)
            efm_2_start_time+=efm_res_dt

            efm_3_start_times.append(efm_3_start_time)
            efm_3_start_time+=efm_res_dt

            efm_4_start_times.append(efm_4_start_time)
            efm_4_start_time+=efm_res_dt
        
        del efm_1_start_time, efm_2_start_time, efm_3_start_time, efm_4_start_time
    slices_dict = {'HRRR':hrrr_list2,'EFM1':efm_1_start_times,'EFM2':efm_2_start_times,'EFM3':efm_3_start_times,'EFM4':efm_4_start_times}
    slices_df = pd.DataFrame(slices_dict)
    return slices_df

def build_merlin_list(slices_df=pd.DataFrame()):

    hrrr_list = slices_df['HRRR'].values
    dt_1hr = np.timedelta64(1,'h')

    slices_df['MERLIN1'] = slices_df['EFM1']+dt_1hr
    slices_df['MERLIN2'] = slices_df['EFM2']+dt_1hr
    slices_df['MERLIN3'] = slices_df['EFM3']+dt_1hr
    slices_df['MERLIN4'] = slices_df['EFM4']+dt_1hr
    return slices_df

def build_rotation_slices(rot=1):

    #get the years for each rotation
    rots_dict = build_rots_dict()
    
    #get the years for the training, validation, and test datasets
    train_years = rots_dict[rot]['train']
    val_years = rots_dict[rot]['val']
    test_years = rots_dict[rot]['test']

    lc_slices_df = pickle.load(open('LC_slices_with_counts.pkl','rb'))
    print(lc_slices_df.columns)

    #get the data where 10 percent of the target images across the hour
    #the pixels have lightning in them. that about 102 pixels
    lc_slices_df = lc_slices_df.loc[lc_slices_df['cc_num_pixels']>=102]

    train_slices_df = lc_slices_df[lc_slices_df['HRRR'].dt.year.isin(train_years)]
    print('rotation, # of training samples: ',rot,len(train_slices_df))
    
    val_slices_df = lc_slices_df[lc_slices_df['HRRR'].dt.year.isin(val_years)]
    print('rotation, # of validation samples:',rot,len(val_slices_df))
    
    test_slices_df = lc_slices_df[lc_slices_df['HRRR'].dt.year.isin(test_years)]
    print('rotation, # of test samples:',rot,len(test_slices_df))

    build_dict_64_2_16_y2y(df = train_slices_df,rot=rot,dict_type='train')
    build_dict_64_2_16_y2y(df = val_slices_df,rot=rot,dict_type='val')
    build_dict_64_2_16_y2y(df = test_slices_df,rot=rot,dict_type='test')

def build_dict_64_2_16_y2y(rot=1,df = pd.DataFrame(), dict_type='train'): 
    print('in build_dict_64_2_16')

    #set the indices and variables
    sfc_vars = ['u','v','sfc_pres','LC_temp']
    x_idxs = [1422,1486]
    y_idxs = [176,240]
    ksc_idxs = [y_idxs[0],y_idxs[1],x_idxs[0],x_idxs[1]]

    input_list = []
    target_list = []

    for v in range(len(df)):
        if v>=0:
            if v%100==0:
                print(v,len(df))

            ltg_ts = pd.Timestamp(df['MERLIN1'].iloc[v])#MERLIN 1 has the file name
            ltg_minute = f"{ltg_ts.minute:02}"
            ltg_hour = f"{ltg_ts.hour:02}"
            ltg_day = f"{ltg_ts.day:02}"
            ltg_month = f"{ltg_ts.month:02}"
            ltg_year = f"{ltg_ts.year:02}"

            ltg_file = 'MERLIN_hrrr_%s%s%s%s%s.nc'%(ltg_year,ltg_month,ltg_day,ltg_hour,ltg_minute)
            ltg_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2d_MERLIN_binary_fed_target/%s%s/'%(ltg_year,ltg_month)
            ltg_ds = xr.open_dataset(ltg_dir+ltg_file,engine='netcdf4')

            input_cc = ltg_ds['input_cc'].values
            input_cg = ltg_ds['input_cg'].values
            ltg_input = np.stack([input_cc,input_cg],axis=-1)
            ltg_input[ltg_input>=1.0] = 1.0
            ltg_input[ltg_input==0.0] = 0.0

            target_cc = ltg_ds['target_binary_cc'].values
            target_cg = ltg_ds['target_binary_cg'].values
            target = np.stack([target_cc,target_cg],axis=-1)

            target_list.append(target)
            input_list.append(ltg_input)

            del ltg_ts, ltg_minute, ltg_hour, ltg_day, ltg_month, ltg_year
            del ltg_file, ltg_dir, ltg_ds, target_cc, target_cg, target

    y1_stack = np.stack(input_list,axis=0)
    x_target_idxs = [23,39]
    y_target_idxs = [26,42]
    y2_stack = y1_stack[:,:,y_target_idxs[0]:y_target_idxs[1],x_target_idxs[0]:x_target_idxs[1],:].copy()
    
    data_dict = {'y_input':y1_stack,'y_target':y2_stack}
    pickle.dump(data_dict,open('/scratch/bmac87/LC_y2y_64_2_16/%s_data.pkl'%dict_type,'wb'))
    del data_dict


def build_dict_no_prior_lightning(df = pd.DataFrame(),rot=1,dict_type='train'):

    #set the indices and variables
    sfc_vars = ['u','v','sfc_pres','LC_temp']
    x_idxs = [1422,1486]
    y_idxs = [176,240]
    ksc_idxs = [y_idxs[0],y_idxs[1],x_idxs[0],x_idxs[1]]

    hrrr_list = []
    target_list = []

    for v in range(len(df)):
        if v>=0:
            if v%100==0:
                print(v,len(df))
            ts = pd.Timestamp(df['HRRR'].iloc[v])
            hour = f"{ts.hour:02}"
            day = f"{ts.day:02}"
            month = f"{ts.month:02}"
            year = f"{ts.year:02}"

            hrrr_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/0_HRRR_Data/3_HRRR_Subhourly_Downselect/%s%s/'%(year,month)
            hrrr_file = '%s%s%s%s.pkl'%(year,month,day,hour)

            hrrr_data_np = pickle.load(open(hrrr_dir+hrrr_file,'rb'))
            
            x_one = np.zeros((4,64,64,4))
            for s,sfc_var in enumerate(sfc_vars):
                if sfc_var =='LC_temp':#you, dumbass, forgot to save the right downselected temperatures
                    temp = hrrr_data_np[sfc_var][ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
                else:
                    temp = hrrr_data_np[sfc_var]
                x_one[s,:,:,:] = temp
                del temp

            #swap the features and the time axis, so the time axis is first and channels is last
            swapped = np.swapaxes(x_one, 0, -1).copy()
            del x_one
            hrrr_list.append(swapped)
            del swapped

            ltg_ts = pd.Timestamp(df['MERLIN1'].iloc[v])#MERLIN 1 has the file name
            ltg_minute = f"{ltg_ts.minute:02}"
            ltg_hour = f"{ltg_ts.hour:02}"
            ltg_day = f"{ltg_ts.day:02}"
            ltg_month = f"{ltg_ts.month:02}"
            ltg_year = f"{ltg_ts.year:02}"

            ltg_file = 'MERLIN_hrrr_%s%s%s%s%s.nc'%(ltg_year,ltg_month,ltg_day,ltg_hour,ltg_minute)
            ltg_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2d_MERLIN_binary_fed_target/%s%s/'%(ltg_year,ltg_month)
            ltg_ds = xr.open_dataset(ltg_dir+ltg_file,engine='netcdf4')
            target_cc = ltg_ds['target_binary_cc'].values
            target_cg = ltg_ds['target_binary_cg'].values
            target = np.stack([target_cc,target_cg],axis=-1)
            target_list.append(target)

            del ltg_ts, ltg_minute, ltg_hour, ltg_day, ltg_month, ltg_year
            del ltg_file, ltg_dir, ltg_ds, target_cc, target_cg, target

    x_stack = np.stack(hrrr_list,axis=0)
    y_stack = np.stack(target_list,axis=0)

    #for 16x16 inputs on the same grid as the targets
    x_target_idxs = [23,39]
    y_target_idxs = [26,42]
    x_stack = x_stack[:,:,y_target_idxs[0]:y_target_idxs[1],x_target_idxs[0]:x_target_idxs[1],:]

    print('x_stack.shape',x_stack.shape,sys.getsizeof(x_stack)/1e9,'GB')
    print('y_stack.shape',y_stack.shape,sys.getsizeof(y_stack)/1e9,'GB')

    data_dict = {'x':x_stack,'y':y_stack,'hrrr_times':df['HRRR'].values,'MERLIN_times':df['MERLIN1'].values}
    save_dir = '/scratch/bmac87/HRRR_only_xy_102pixels/'
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    pickle.dump(data_dict,open(save_dir+'rot_%s_%s.pkl'%(rot,dict_type),'wb'))

def build_dict_no_prior_lightning_with_EFM(df = pd.DataFrame(),rot=1,dict_type='train'):

    #set the indices and variables
    sfc_vars = ['u','v','sfc_pres','LC_temp']
    x_idxs = [1422,1486]
    y_idxs = [176,240]
    ksc_idxs = [y_idxs[0],y_idxs[1],x_idxs[0],x_idxs[1]]

    #build out the matrix with the fixed efm distances to the target grid
    efm_distances = xr.open_dataset('./1_EFM/efms2hrrrdist.nc',engine='netcdf4')#the distances were pre-calculated
    site_names = efm_distances['site_names'].values

    #use the top 24 sites with the least amount of nans (pre determined; see 1e_concat_annual.py)
    top_24_sites = ['FM30', 'FM11', 'FM20', 'FM17', 'FM21', 'FM24', 'FM05', 'FM22', 'FM15', 'FM12', 'FM31', 'FM08', 'FM16', 'FM06', 'FM07', 'FM27', 'FM29', 'FM04', 'FM14', 'FM19', 'FM26', 'FM32', 'FM25', 'FM10']
    
    #find the indices of the dataset that pertain to the top 24 sites
    top_24_idxs = []
    for top_24 in top_24_sites:
        idx = np.where(site_names==top_24)
        top_24_idxs.append(idx[0][0])

    #get the distances in a numpy format
    dist_np = efm_distances['dist'].values[top_24_idxs,:,:]

    #put the 24 distances as the last feature
    dist_last_ftr = np.swapaxes(dist_np, 0, -1).copy()
    del dist_np, top_24_idxs, idx, site_names

    #load the efm stats data, after the winter months were dropped
    median_ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/1e_6_drop_winter/median.nc',engine='netcdf4')
    median_ds = median_ds[top_24_sites]
    
    max_ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/1e_6_drop_winter/max.nc',engine='netcdf4')
    max_ds = max_ds[top_24_sites]
    
    min_ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/1e_6_drop_winter/min.nc',engine='netcdf4')
    min_ds = min_ds[top_24_sites]
    
    std_ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/1_EFM_Data/1e_6_drop_winter/std.nc',engine='netcdf4')
    std_ds = std_ds[top_24_sites]

    hrrr_list = []
    target_list = []
    efm_dist_list = []
    efm_sample_list = []
    
    for v in range(len(df)):
        if v>=0:
            if v%10==0:
                print('rot',rot,dict_type,v,len(df))
            
            efm_time = df[['EFM1']].iloc[v].values[0]
            efm_dt = np.timedelta64(15,'m')

            efm_med_stack = []
            efm_min_stack = []
            efm_max_stack = []
            efm_std_stack = []

            for i in range(4):
                start_time = efm_time
                end_time = efm_time+efm_dt
                med_slice_np = efm_fill_image(ds = median_ds,start_time=efm_time,end_time=end_time,top_24_sites=top_24_sites)
                min_slice_np = efm_fill_image(ds = min_ds,start_time=efm_time,end_time=end_time,top_24_sites=top_24_sites)
                max_slice_np = efm_fill_image(ds = max_ds,start_time=efm_time,end_time=end_time,top_24_sites=top_24_sites)
                std_slice_np = efm_fill_image(ds = std_ds,start_time=efm_time,end_time=end_time,top_24_sites=top_24_sites)
                start_time=end_time
                
                efm_med_stack.append(med_slice_np)
                efm_min_stack.append(min_slice_np)
                efm_max_stack.append(max_slice_np)
                efm_std_stack.append(std_slice_np)
                del med_slice_np, min_slice_np, max_slice_np, std_slice_np

            efm_med_one_sample = np.stack(efm_med_stack,axis=0)#4, 24, 181 (time; num efms; time series)
            efm_min_one_sample = np.stack(efm_min_stack,axis=0)
            efm_max_one_sample = np.stack(efm_max_stack,axis=0)
            efm_std_one_sample = np.stack(efm_std_stack,axis=0)
            del efm_std_stack, efm_max_stack, efm_min_stack, efm_med_stack

            efm_one_sample = np.stack([efm_med_one_sample,efm_max_one_sample,efm_min_one_sample,efm_std_one_sample],axis=-1)
            del efm_med_one_sample,efm_max_one_sample,efm_min_one_sample,efm_std_one_sample
            
            efm_sample_list.append(efm_one_sample)
            del efm_one_sample

            ####HRRRR PROCESSING######
            ts = pd.Timestamp(df['HRRR'].iloc[v])
            hour = f"{ts.hour:02}"
            day = f"{ts.day:02}"
            month = f"{ts.month:02}"
            year = f"{ts.year:02}"
        
            hrrr_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/0_HRRR_Data/3_HRRR_Subhourly_Downselect/%s%s/'%(year,month)
            hrrr_file = '%s%s%s%s.pkl'%(year,month,day,hour)

            hrrr_data_np = pickle.load(open(hrrr_dir+hrrr_file,'rb'))#dictionary
            
            x_one = np.zeros((4,64,64,4))
            for s,sfc_var in enumerate(sfc_vars):
                if sfc_var =='LC_temp':#you, forgot to save the right downselected temperatures
                    temp = hrrr_data_np[sfc_var][ksc_idxs[0]:ksc_idxs[1],ksc_idxs[2]:ksc_idxs[3]]
                else:
                    temp = hrrr_data_np[sfc_var]
                x_one[s,:,:,:] = temp
                del temp

            #swap the features and the time axis, so the time axis is first and channels is last
            swapped = np.swapaxes(x_one, 0, -1).copy()
            del x_one
            hrrr_list.append(swapped)
            del swapped

            #####MERLIN Processing
            ltg_ts = pd.Timestamp(df['MERLIN1'].iloc[v])#MERLIN 1 has the file name
            ltg_minute = f"{ltg_ts.minute:02}"
            ltg_hour = f"{ltg_ts.hour:02}"
            ltg_day = f"{ltg_ts.day:02}"
            ltg_month = f"{ltg_ts.month:02}"
            ltg_year = f"{ltg_ts.year:02}"

            ltg_file = 'MERLIN_hrrr_%s%s%s%s%s.nc'%(ltg_year,ltg_month,ltg_day,ltg_hour,ltg_minute)
            ltg_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2d_MERLIN_binary_fed_target/%s%s/'%(ltg_year,ltg_month)
            ltg_ds = xr.open_dataset(ltg_dir+ltg_file,engine='netcdf4')
            target_cc = ltg_ds['target_binary_cc'].values
            target_cg = ltg_ds['target_binary_cg'].values
            target = np.stack([target_cc,target_cg],axis=-1)
            target_list.append(target)

            efm_dist_list.append(dist_last_ftr)

            del ltg_ts, ltg_minute, ltg_hour, ltg_day, ltg_month, ltg_year
            del ltg_file, ltg_dir, ltg_ds, target_cc, target_cg, target

    x_stack = np.stack(hrrr_list,axis=0)
    x_efm_dist_stack = np.stack(efm_dist_list,axis=0)
    x_efm_stack = np.stack(efm_sample_list,axis=0)
    y_stack = np.stack(target_list,axis=0)

    #for 16x16 inputs on the same grid as the targets
    x_target_idxs = [23,39]
    y_target_idxs = [26,42]
    x_stack = x_stack[:,:,y_target_idxs[0]:y_target_idxs[1],x_target_idxs[0]:x_target_idxs[1],:]

    data_dict = {'x_hrrr':x_stack,
                'x_efm_dist':x_efm_dist_stack,
                'x_efm_stats':x_efm_stack,
                'y':y_stack,
                'hrrr_times':df['HRRR'].values,
                'MERLIN_times':df['MERLIN1'].values,
                'EFM_times':df['EFM1'].values}

    save_dir = '/scratch/bmac87/HRRR_EFM_xy_102pixels/'
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    pickle.dump(data_dict,open(save_dir+'rot_%s_%s.pkl'%(rot,dict_type),'wb'))

def build_dict_y2y(df = pd.DataFrame(),rot=1,dict_type='train'):
    
    print('building the y2y')
    input_list = []
    target_list = []

    for i in range(len(df)):
        ltg_ts = pd.Timestamp(df['MERLIN1'].iloc[i])#MERLIN 1 has the file name
        ltg_minute = f"{ltg_ts.minute:02}"
        ltg_hour = f"{ltg_ts.hour:02}"
        ltg_day = f"{ltg_ts.day:02}"
        ltg_month = f"{ltg_ts.month:02}"
        ltg_year = f"{ltg_ts.year:02}"

        ltg_file = 'MERLIN_hrrr_%s%s%s%s%s.nc'%(ltg_year,ltg_month,ltg_day,ltg_hour,ltg_minute)
        ltg_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2d_MERLIN_binary_fed_target/%s%s/'%(ltg_year,ltg_month)
        ltg_ds = xr.open_dataset(ltg_dir+ltg_file,engine='netcdf4')
        
        input_cc = ltg_ds['input_cc'].values
        input_cc2 = copy.deepcopy(input_cc)
        input_cc2[input_cc2>=1.0]=1.0
        input_cc2[input_cc2==0.0]=0.0

        input_cg = ltg_ds['input_cg'].values
        input_cg2 = copy.deepcopy(input_cg)
        input_cg2[input_cg2>=1.0]=1.0
        input_cg2[input_cg2==0.0]=0.0

        ltg_input = np.stack([input_cc2,input_cg2],axis=-1)
        input_list.append(ltg_input)

        target_cc = ltg_ds['target_binary_cc'].values
        target_cg = ltg_ds['target_binary_cg'].values
        target = np.stack([target_cc,target_cg],axis=-1)
        target_list.append(target)

    x_stack = np.stack(input_list,axis=0)
    y_stack = np.stack(target_list,axis=0)

    data_dict = {'x':x_stack,'y':y_stack}
    save_dir = '/scratch/bmac87/LC_y2y_102_pixels/'
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    pickle.dump(data_dict,open(save_dir+'rot_%s_%s.pkl'%(rot,dict_type),'wb'))

def build_slices_all():
    print('generating the time information for the efm, hrrr, and merlin data')
    hrrr_all = []
    for year in ['2018','2019','2020','2021','2022','2023','2024']:
        hrrr_all.append(build_hrrr_list(year=year))
    hrrr_all = np.concatenate(hrrr_all)
    slices_df = build_efm_list(hrrr_list = hrrr_all)
    slices_df = build_merlin_list(slices_df=slices_df)
    slices_df.to_pickle(open('LC_slices.pkl','wb'))

def build_rots_dict():
    rots_dict = {
        1: {'train':[2018,2019,2020,2021,2022],
            'val':[2023],
            'test':[2024]},
        2: {
            'train':[2019,2020,2021,2022,2023],
            'val':[2024],
            'test':[2018]
        },
        3: {
            'train':[2020,2021,2022,2023,2024],
            'val':[2018],
            'test':[2019]
        },
        4: {
            'train':[2021,2022,2023,2024,2018],
            'val':[2019],
            'test':[2020]
        },
        5: {
            'train':[2022,2023,2024,2018,2019],
            'val':[2020],
            'test':[2021]
        },
        6: {
            'train':[2023,2024,2018,2019,2020],
            'val':[2021],
            'test':[2022]
        },
        7: {
            'train':[2024,2018,2019,2020,2021],
            'val':[2022],
            'test':[2023]
        }
    }
    return rots_dict

def get_cc_pixel_count():
    slices_df = pickle.load(open('LC_slices.pkl','rb'))
    merlin1_times = slices_df['MERLIN1'].values
    cc_pixel_count = np.zeros(len(merlin1_times))
    cg_pixel_count = np.zeros(len(merlin1_times))

    for mt,merlin_time in enumerate(merlin1_times):
        ts = pd.Timestamp(merlin_time)
        minute = f"{ts.minute:02}"
        hour = f"{ts.hour:02}"
        day = f"{ts.day:02}"
        month = f"{ts.month:02}"
        year = f"{ts.year:02}"

        ltg_file = 'MERLIN_hrrr_%s%s%s%s%s.nc'%(year,month,day,hour,minute)
        ltg_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2d_MERLIN_binary_fed_target/%s%s/'%(year,month)
        ltg_ds = xr.open_dataset(ltg_dir+ltg_file,engine='netcdf4')
        target_cc = ltg_ds['target_binary_cc'].values
        target_cg = ltg_ds['target_binary_cg'].values

        cc_pixel_count[mt] = np.sum(np.sum(np.sum(target_cc)))
        cg_pixel_count[mt] = np.sum(np.sum(np.sum(target_cg)))
        del ltg_ds, target_cc, target_cg

    slices_df['cc_num_pixels'] = cc_pixel_count
    slices_df['cg_num_pixels'] = cg_pixel_count
    slices_df.to_pickle(open('LC_slices_with_counts.pkl','wb'))

def min_max_np(data):

    shape = data.shape
    print('min_max_np: shape:',shape)
    norm_data = np.zeros(shape)
    
    dims = len(shape)
    print('# of dimensions:',dims)
    
    num_features = shape[-1]
    print('# of features:',num_features)

    for i in range(num_features):
        temp_data = np.squeeze(data[:,:,:,:,i])
        max = np.nanmax(np.nanmax(np.nanmax(np.nanmax(temp_data,axis=3),axis=2),axis=1),axis=0)
        min = np.nanmax(np.nanmax(np.nanmax(np.nanmax(temp_data,axis=3),axis=2),axis=1),axis=0)
        diff = max-min

        if diff!=0.0:
            norm_data[:,:,:,:,i] = (temp_data - min)/diff
    
    return norm_data

def z_score_norm(data):
    shape = data.shape
    norm_data = np.zeros(shape)
    num_features = shape[-1]
    for i in range(num_features):
        temp_data = copy.deepcopy(np.squeeze(data[:,:,:,:,i]))
        mean = np.mean(np.mean(np.mean(temp_data)))
        variance = np.var(temp_data)
        norm_data[:,:,:,:,i]=(temp_data-mean)/variance
        del temp_data
    return norm_data

def norm(rot=1):

    for ds_type in ['train','val','test']:
        data_dict = pickle.load(open('/scratch/bmac87/HRRR_only_xy_102pixels/rot_%s_%s.pkl'%(rot,ds_type),'rb'))
        x_norm = z_score_norm(data_dict['x'])
        data_dict.update({'x_norm':x_norm})
        pickle.dump(data_dict,open('/scratch/bmac87/HRRR_only_xy_102pixels/rot_%s_%s.pkl'%(rot,ds_type),'wb'))

def visualize_data(rot=1):
    print('visualizing the data for rotation %s'%rot)

    train_data_dict = pickle.load(open('/scratch/bmac87/HRRR_only_xy/rot_%s_train.pkl'%(rot),'rb'))
    sfc_vars = ['u','v','sfc_pres','LC_temp']

    #get the input and target grids
    ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2d_MERLIN_binary_fed_target/202206/MERLIN_hrrr_202206302230.nc',engine='netcdf4')
    
    input_lat = ds['input_lat'].values
    input_lon = ds['input_lon'].values
    
    target_lat = ds['target_lat'].values
    target_lon = ds['target_lon'].values

    #get the non-normalized inputs, normalized inputs, and target data
    x = train_data_dict['x']
    x_norm = train_data_dict['x_norm']

    y = train_data_dict['y']
    hrrr_times = train_data_dict['hrrr_times']
    merlin_times = train_data_dict['MERLIN_times']

    num_indices_to_select = 200
    random_indices = np.random.choice(np.arange(x_norm.shape[0]), size=num_indices_to_select, replace=False)
    x_random = x[random_indices,:,:,:,:]
    x_norm_random = x_norm[random_indices,:,:,:,:]
    y_random = y[random_indices,:,:,:,:]
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    for t in range(x_random.shape[0]):
        fig,axes = plt.subplots(nrows=4,ncols=3,figsize=(40,60),subplot_kw={'projection': ccrs.PlateCarree()})
        hrrr_time = hrrr_times[t]
        merlin_time = merlin_times[t]
        for h in range(4):#time

            merlin_ts = pd.Timestamp(merlin_time)
            merlin_minute = f"{merlin_ts.minute:02}"
            merlin_hour = f"{merlin_ts.hour:02}"
            merlin_day = f"{merlin_ts.day:02}"
            merlin_month = months[merlin_ts.month-1]
            merlin_year = f"{merlin_ts.year:04}"
            merlin_title = '%s:%s UTC %s %s %s'%(merlin_hour,merlin_minute,merlin_day,merlin_month,merlin_year)

            hrrr_ts = pd.Timestamp(hrrr_time)
            hrrr_minute = f"{hrrr_ts.minute:02}"
            hrrr_hour = f"{hrrr_ts.hour:02}"
            hrrr_day = f"{hrrr_ts.day:02}"
            hrrr_month = months[hrrr_ts.month-1]
            hrrr_year = f"{hrrr_ts.year:04}"
            hrrr_title = 'HRRR Valid Time: %s:%s UTC %s %s %s'%(hrrr_hour,hrrr_minute,hrrr_day,hrrr_month,hrrr_year)

            for j in range(2):#non-norm and normed inputs
                if j==0:
                    # ax.quiver(x, y, u, v, transform=vector_crs)
                    u = x_random[t,h,:,:,0]
                    v = x_random[t,h,:,:,1]
                    pres = x_random[t,h,:,:,2]
                    temp_K = x_random[t,h,:,:,3]
                    axes[h,j].pcolormesh(input_lon,input_lat,temp_K,cmap='coolwarm')
                    axes[h,j].quiver(input_lon[::3,::3],input_lat[::3,::3],u[::3,::3],v[::3,::3])#winds
                    axes[h,j].contour(input_lon,input_lat,pres,colors='black',linewidths=2.0)
                    axes[h,j].coastlines()
                    axes[h,j].set_title(hrrr_title,fontsize=24)
                    del u, v, pres, temp_K

                if j==1:
                    u = x_norm_random[t,h,:,:,0]
                    v = x_norm_random[t,h,:,:,1]
                    pres = x_norm_random[t,h,:,:,2]
                    temp_K = x_norm_random[t,h,:,:,3]
                    axes[h,j].pcolormesh(input_lon,input_lat,temp_K,cmap='coolwarm')
                    axes[h,j].quiver(input_lon[::3,::3],input_lat[::3,::3],u[::3,::3],v[::3,::3])#winds
                    axes[h,j].contour(input_lon,input_lat,pres,colors='black',linewidths=2.0)
                    axes[h,j].coastlines()
                    axes[h,j].set_title(hrrr_title,fontsize=24)
                    del u, v, pres, temp_K

            y_cc = y_random[t,h,:,:,0]
            y_cc[y_cc==0] = np.nan
            y_cg = y_random[t,h,:,:,1]
            y_cg[y_cg==0] = np.nan

            axes[h,2].pcolormesh(target_lon, target_lat, y_cc,cmap='Reds_r')
            axes[h,2].pcolormesh(target_lon, target_lat, y_cg,cmap='Blues_r')
            axes[h,2].coastlines()
            axes[h,2].set_title(merlin_title,fontsize=24)
            
            del y_cc, y_cg

            merlin_time+=np.timedelta64(15,'m')
            hrrr_time+=np.timedelta64(15,'m')

        del merlin_time, hrrr_time

        save_dir = '/scratch/bmac87/data_images/'
        if os.path.isdir(save_dir)==False:
            os.makedirs(save_dir)
        plt.savefig('%s/trng_rot_%s_%s.png'%(save_dir,rot,t))
        plt.close()

if __name__=='__main__':
    build_rotation_slices(rot=1)
