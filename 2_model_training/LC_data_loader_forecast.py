import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import shutil
import pickle
import glob
from LC_parser import *
import wandb
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import copy
import pygrib
import tensorflow as tf
from scipy.ndimage import binary_fill_holes, binary_closing
from skimage.morphology import convex_hull_image
from collections import OrderedDict

def min_max_norm_2d(key='group_area',data_np=[]):
    print('normalizing',key)
    radar_keys = ['Reflectivity_-10C_00.50', 'Reflectivity_-15C_00.50', 'Reflectivity_-20C_00.50', 'Reflectivity_-5C_00.50', 'Reflectivity_0C_00.50', 'VII_00.50', 'VIL_00.50']
    FM_keys = ['FM01', 'FM02', 'FM04', 'FM05', 'FM06', 'FM07', 'FM08', 'FM09', 'FM10', 'FM11', 'FM12', 'FM14', 'FM15', 'FM16', 'FM17', 'FM18', 'FM19', 'FM20', 'FM21', 'FM22', 'FM24', 'FM25', 'FM26', 'FM27', 'FM28', 'FM29', 'FM30', 'FM31', 'FM32', 'FM34', 'FM35']
    min_max_dict = pickle.load(open('overall_min_max_all_observed.pkl','rb'))
    data_max =  min_max_dict[key]['max']
    data_min = min_max_dict[key]['min']

    if key=='group_area' or key=='group_energy':#GLM
        data_norm = (data_np-data_min)/(data_max-data_min)
    elif key in z_keys:#MRMS
        data_np[data_np<0.0]=np.nan
        data_norm = (data_np-data_min)/(data_max-data_min)
        data_norm[np.isnan(data_norm)] = 0.0
    else:#HRRR
        data_norm=(data_np-data_min)/(data_max-data_min)
    return data_norm

def norm_efm_ts(key='FM01',data_np=[],stat='median'):
    #load the mins and maxes of the EFM data
    min_max_dict = pickle.load(open('overall_min_max_EFM_observed.pkl','rb'))
    single_min_max_dict = min_max_dict['%s_%s'%(key,stat)]
    data_max = single_min_max_dict['max']
    data_min = single_min_max_dict['min']
    
    #calculate the difference
    diff = data_max-data_min

    #pre-allocate the array
    data_norm = np.zeros(data_np.shape)

    #any values with -20000 are a mask convert to nan
    data_np[data_np==-20000]=np.nan

    #if the max and mins are real, normalize the data
    if np.isnan(data_max)==False and np.isnan(data_min)==False:
        data_norm = (data_np-data_min)/(diff)
    data_norm[np.isnan(data_norm)]=0.0
    return data_norm

def build_rotations(rot=1):
    if rot==1:
        train_files = ['LC_2021.tfds','LC_2022.tfds']
        val_files = ['LC_2023.tfds']
        test_files = ['LC_2024.tfds']
    elif rot==2:
        train_files = ['LC_2022.tfds','LC_2023.tfds']
        val_files = ['LC_2024.tfds']
        test_files = ['LC_2021.tfds']
    elif rot==3:
        train_files = ['LC_2023.tfds','LC_2024.tfds']
        val_files = ['LC_2021.tfds']
        test_files = ['LC_2022.tfds']
    else:
        train_files = ['LC_2024.tfds','LC_2021.tfds']
        val_files = ['LC_2022.tfds']
        test_files = ['LC_2023.tfds']
    return train_files, val_files, test_files

def norm_2_tfds(args):
    data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/6_final_xr/'
    radar_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/4_MRMS_Data/MRMS_annual_concat/Reflectivity_0C_00.50/'

    #load the keys from the arguments
    glm_keys = args.GLM_keys
    merlin_keys = args.merlin_keys
    zdr_keys = args.Zdr_keys
    z_keys = args.Z_keys
    vi_keys=args.VI_keys
    hrrr_features = args.hrrr_features
    efm_stats = args.efm_stats
    FM_keys = args.efm_keys
    years = args.years
    time_keys=['t0','t1','t2','t3']

    for y,year in enumerate(years):
        x_dict = OrderedDict()
        if y>=0:
            radar_fname = '%s.nc'%(year)
            radar_ds = xr.open_dataset(radar_dir+radar_fname,engine='netcdf4')
            dt = np.timedelta64(15,'m')

            fname='LC_%s.nc'%year
            ds = xr.open_dataset(data_dir+fname,engine='netcdf4')
            data_list = []
            sample_times = ds['sample_time'].values
            print(len(sample_times))
            merlin_fcst_time = ds['merlin_fcst_time'].values

            #only predict one lightning image. not IC and CG. 
            total_ltg = ds['cc_forecast'].values + ds['cg_forecast'].values
            total_ltg[total_ltg>0] = 1.0
            total_ltg[total_ltg==0] = 0.0

            qcd_sample_times = []
            final_target_list = []
            for i in range(len(sample_times)):
                if i%10==0:
                    print(i,len(sample_times))
                ltg_target_one_sample=[]
                try:
                    for t in range(4):
                        temp_ltg_data = total_ltg[i,t,:,:]
                        radar_slice = radar_ds.sel(time=slice(merlin_fcst_time[i][t],merlin_fcst_time[i][t]+dt))
                        radar_slice_np = radar_slice['radar_data'].values
                        radar_max_np = np.max(radar_slice_np,axis=0)
                        convex_hull = convex_hull_image(temp_ltg_data)
                        if convex_hull is None:
                            convex_hull=np.zeros((64,64))     
                        ltg_target_one_sample.append(convex_hull*(radar_max_np>=30))
                    uno_sample = np.stack(ltg_target_one_sample)
                    final_target_list.append(uno_sample)
                    qcd_sample_times.append(sample_times[i])
                except Exception as e:
                    print(e)
                    print(i)
                    continue
            print(len(final_target_list))
            print(len(qcd_sample_times))

            ds = ds.sel(sample_time=qcd_sample_times)
            print('data downsampled fine')
            merlin_np = np.stack(final_target_list,axis=0)

            if args.use_MRMS==True:
                if args.use_Z==True:
                    for key in z_keys:
                        data_np = ds[key].values
                        data_norm_np = min_max_norm_2d(key,data_np)
                        for t,t_key in enumerate(time_keys):
                            x_dict.update({'MRMS_%s_%s'%(key,t_key):data_norm_np[:,t,:,:]})
                        del data_np, data_norm_np

                if args.use_VI==True:
                    for key in vi_keys:
                        data_np = ds[key].values
                        data_norm_np = min_max_norm_2d(key,data_np)
                        for t,t_key in enumerate(time_keys):
                            x_dict.update({'MRMS_%s_%s'%(key,t_key):data_norm_np[:,t,:,:]})
                        del data_np, data_norm_np

                if args.use_Zdr==True:
                    for key in zdr_keys:
                        data_np = ds[key].values
                        data_norm_np = min_max_norm_2d(key,data_np)
                        for t,t_key in enumerate(time_keys):
                            x_dict.update({'MRMS_%s_%s'%(key,t_key):data_norm_np[:,t,:,:]})
                        del data_np, data_norm_np
            
            if args.use_GLM==True:
                for key in glm_keys:
                    data_np = ds[key].values
                    data_norm_np = min_max_norm_2d(key,data_np)
                    for t,t_key in enumerate(time_keys):
                        x_dict.update({'GLM_%s_%s'%(key,t_key):data_norm_np[:,t,:,:]})
                    del data_np, data_norm_np

            if args.use_HRRR==True:
                hrrr_data_np = ds['hrrr_data'].values
                for f,ftr in enumerate(args.hrrr_features):
                    data_norm_np = min_max_norm_2d(key=ftr,data_np=hrrr_data_np[:,:,:,:,f])
                    for t,t_key in enumerate(time_keys):
                        x_dict.update({'HRRR_%s_%s'%(ftr,t_key):data_norm_np[:,t,:,:]})
                    del data_norm_np

            if args.use_EFM==True:
                for f in range(len(FM_keys)):
                    fm = FM_keys[f]

                    #extract the efm data into a 1-hour time series
                    #make into a 1-hour long vector
                    single_efm_data = ds[fm].values
                    single_efm_list = []
                    for pt in range(4):
                        single_efm_list.append(single_efm_data[:,pt,:])
                    single_efm_ts = np.concatenate(single_efm_list,axis=1)
                    stat_idxs=[0,-1]#median, std

                    #pre-allocate array
                    normed_efm_stats = np.zeros(single_efm_ts[:,:,stat_idxs].shape)
                    idx=0
                    for stat_idx in stat_idxs:
                        #get the stat label
                        stat = efm_stats[stat_idx]

                        #then get the time series of that stat label
                        single_efm_stat = single_efm_ts[:,:,stat_idx]

                        #then normalize the time series
                        normed_efm_stats[:,:,idx] = norm_efm_ts(data_np=single_efm_stat,key=fm,stat=stat)
                        idx+=1
                    x_dict.update({fm:normed_efm_stats})
                    del normed_efm_stats, single_efm_stat, single_efm_ts, single_efm_data

                efm_dist_file = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/88_LC_observed_datasets/6_final_xr_observed/LC_observed_2022_w_EFM_dist.nc'
                efm_distances_ds = xr.open_dataset(efm_dist_file,engine='netcdf4')
                efm_distances_np = np.squeeze(efm_distances_ds['efm_distances'].values[0,0,:,:,:])
                for f in range(len(FM_keys)):
                    
                    fm = FM_keys[f]#get the key

                    #normalize the distances to the grid cells
                    #these are fixed, so can do repeatedly without issue
                    
                    #one 64x64 grid for the distances for one EFM, across all samples
                    single_site_distances = np.squeeze(efm_distances_np[:,:,f])
                    
                    #get the max
                    dist_max = np.max(single_site_distances)

                    #get the min
                    dist_min = np.min(single_site_distances)

                    #min_max normalize
                    single_site_norm_distance = (single_site_distances-dist_min)/(dist_max-dist_min)
                    num_copies = merlin_np.shape[0]

                    #store the data
                    x_dict.update({'%s_distances'%fm:np.stack([single_site_norm_distance] * num_copies, axis=0)})
                    del single_site_norm_distance, dist_max, dist_min, single_site_distances

            print('the inputs in order: for verification with tf inputs, and their expected shape')
            for key in x_dict:
                print(key,x_dict[key].shape)
            data_tuple = (x_dict,merlin_np)
            tfds = tf.data.Dataset.from_tensor_slices(data_tuple)
            save_dir='/scratch/bmac87/'
            tfds.save(save_dir+'%s_%s.tfds'%(args.exp_pre,year))
            del data_tuple
            del x_dict, tfds
            
def tuple_2_model(args):
    rot=args.rotation
    data_dir=args.data_path
    # train_files, val_files, test_files = build_rotations(rot=rot)
    train_files = ['%s_2021_tuple.pkl'%(args.exp_pre),'%s_2022_tuple.pkl'%(args.exp_pre)]
    val_files = ['%s_2023_tuple.pkl'%(args.exp_pre)]
    test_files = ['%s_2024_tuple.pkl'%(args.exp_pre)]

    train_data_1 = pickle.load(open('%s%s'%(data_dir,train_files[0]),'rb'))
    train_data_2 = pickle.load(open('%s%s'%(data_dir,train_files[1]),'rb'))
    x_dict = OrderedDict()
    for key in train_data_1[0]:
        print(key)
        temp_data_1 = train_data_1[0][key]
        temp_data_2 = train_data_2[0][key]
        trn_data = np.concatenate([temp_data_1,temp_data_2],axis=0)
        x_dict.update({key:trn_data})
    y_true = np.concatenate([train_data_1[1],train_data_2[1]],axis=0)
    train_tfds = tf.data.Dataset.from_tensor_slices((x_dict,y_true))
    bs = tf.data.experimental.cardinality(train_tfds).numpy()
    print('training dataset shuffle size',bs)
    train_tfds = train_tfds.cache()
    # # train_tfds = train_tfds.shuffle(buffer_size=bs)
    train_tfds = train_tfds.batch(args.batch)
    print('train_tfds')
    print(train_tfds)
    # # train_tfds = train_tfds.prefetch(tf.data.AUTOTUNE)
    
    val_tuple = pickle.load(open('%s%s'%(data_dir,val_files[0]),'rb'))
    for key in val_tuple[0]:
        print('val',key)
    val_tfds = tf.data.Dataset.from_tensor_slices(val_tuple)
    bs = tf.data.experimental.cardinality(val_tfds).numpy()
    print('validation dataset shuffle size',bs)
    val_tfds = val_tfds.cache()
    # val_tfds = val_tfds.shuffle(buffer_size=bs)
    val_tfds = val_tfds.batch(args.batch)
    # # val_tfds = val_tfds.prefetch(tf.data.AUTOTUNE)
    print('val_tfds')
    print(val_tfds)

    test_tuple = pickle.load(open('%s%s'%(data_dir,test_files[0]),'rb'))
    for key in test_tuple[0]:
        print('test',key)
    test_tfds = tf.data.Dataset.from_tensor_slices(test_tuple)
    bs = tf.data.experimental.cardinality(test_tfds).numpy()
    print('test dataset shuffle buffer size',bs)
    test_tfds = test_tfds.cache()
    # test_tfds = test_tfds.shuffle(buffer_size=bs)
    test_tfds = test_tfds.batch(args.batch)
    # test_tfds = test_tfds.prefetch(tf.data.AUTOTUNE)
    print('test_tfds')
    print(test_tfds)
    return train_tfds, val_tfds, test_tfds

def tfds_2_model(args):
    print('loading the tfds-s')
    data_dir=args.data_path
    train_files = ['%s_2021.tfds'%(args.exp_pre),'%s_2022.tfds'%(args.exp_pre)]
    val_files = ['%s_2023.tfds'%(args.exp_pre)]
    test_files = ['%s_2024.tfds'%(args.exp_pre)]

    #shuffle, batch, prefetch
    tr1 = tf.data.Dataset.load(data_dir+train_files[0])
    tr2 = tf.data.Dataset.load(data_dir+train_files[1])
    train_tfds = tr1.concatenate(tr2)
    
    if args.cache==True:
        train_tfds=train_tfds.cache()
    if args.shuffle>0:
        card = tf.data.experimental.cardinality(train_tfds)
        if card==tf.data.INFINITE_CARDINALITY:
            bs=0
        else:
            bs=card
        train_tfds=train_tfds.shuffle(buffer_size=bs,reshuffle_each_iteration=True)
    train_tfds=train_tfds.batch(args.batch,drop_remainder=True)
    train_tfds=train_tfds.prefetch(tf.data.AUTOTUNE)

    val_tfds = tf.data.Dataset.load(data_dir+val_files[0])
    if args.cache==True:
        val_tfds=val_tfds.cache()
    val_tfds=val_tfds.batch(args.batch,drop_remainder=True)
    val_tfds=val_tfds.prefetch(tf.data.AUTOTUNE)

    test_tfds = tf.data.Dataset.load(data_dir+test_files[0])
    test_tfds=test_tfds.batch(args.batch)
    test_tfds=test_tfds.prefetch(tf.data.AUTOTUNE)
    return train_tfds, val_tfds, test_tfds

def visualize_tfds(args):
    year='2021'
    load_dir = '/scratch/bmac87/'
    fname = '%s_%s.tfds'%(args.exp_pre,year)
    test_tfds = tf.data.Dataset.load(load_dir+fname)
    takes = np.arange(1,20)
    
    for tk in takes:
        for x,y in test_tfds.take(tk):
            print(y.shape)
            fig,axes = plt.subplots(nrows=1,ncols=4)
            axes[0].imshow(y[0,:,:])
            axes[1].imshow(y[1,:,:])
            axes[2].imshow(y[2,:,:])
            axes[3].imshow(y[3,:,:])
            plt.savefig('./test_images/%s_test_y.png'%tk)
            plt.close()
    # FM_keys = ['FM01', 'FM02', 'FM04', 'FM05', 'FM06', 'FM07', 'FM08', 'FM09', 'FM10', 'FM11', 'FM12', 'FM14', 'FM15', 'FM16', 'FM17', 'FM18', 'FM19', 'FM20', 'FM21', 'FM22', 'FM24', 'FM25', 'FM26', 'FM27', 'FM28', 'FM29', 'FM30', 'FM31', 'FM32', 'FM34', 'FM35']
        # for fm in FM_keys:
        #     one_efm = x[fm].numpy()
        #     efm_dist = x[fm+'_distances'].numpy()
        #     print(one_efm.shape)
        #     fig,axes = plt.subplots(nrows=1,ncols=3)
        #     axes[0].plot(one_efm[:,0])
        #     axes[1].plot(one_efm[:,1])
        #     axes[2].imshow(efm_dist)
        #     plt.savefig('/scratch/bmac87/takes_test_efm_images/one_efm_tfds_test_%s_take_%s.png'%(fm,tk))
        #     plt.close()

if __name__=='__main__':
    parser = create_parser()
    args = parser.parse_args()
    norm_2_tfds(args=args)
    # visualize_tfds(args=args)#modified frequently