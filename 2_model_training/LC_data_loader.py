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
from collections import OrderedDict

def min_max_norm_2d(key='group_area',data_np=[]):
    print('normalizing',key)
    z_keys = ['Reflectivity_-10C_00.50', 'Reflectivity_-15C_00.50', 'Reflectivity_-20C_00.50', 'Reflectivity_-5C_00.50', 'Reflectivity_0C_00.50', 'VII_00.50', 'VIL_00.50']
    FM_keys = ['FM01', 'FM02', 'FM04', 'FM05', 'FM06', 'FM07', 'FM08', 'FM09', 'FM10', 'FM11', 'FM12', 'FM14', 'FM15', 'FM16', 'FM17', 'FM18', 'FM19', 'FM20', 'FM21', 'FM22', 'FM24', 'FM25', 'FM26', 'FM27', 'FM28', 'FM29', 'FM30', 'FM31', 'FM32', 'FM34', 'FM35']
    min_max_dict = pickle.load(open('overall_min_max_all_observed.pkl','rb'))
    data_max =  min_max_dict[key]['max']
    data_min = min_max_dict[key]['min']
    if key=='group_area' or key=='group_energy':
        data_norm = (data_np-data_min)/(data_max-data_min)
    if key in z_keys:
        data_np[data_np<0.0]=np.nan
        data_norm = (data_np-data_min)/(data_max-data_min)
        data_norm[np.isnan(data_norm)] = 0.0
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
    data_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/88_LC_observed_datasets/6_final_xr_observed/'
    
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
            ds = xr.open_dataset(data_dir+'LC_observed_%s_w_EFM_dist.nc'%year,engine='netcdf4')
            data_list = []
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
                    
            if args.use_EFM==True:
                efm_distances = ds['efm_distances'].values
                for f in range(efm_distances.shape[-1]):
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

                for f in range(efm_distances.shape[-1]):
                    fm = FM_keys[f]#get the key

                    #normalize the distances to the grid cells
                    #these are fixed, so can do repeatedly without issue
                    
                    #one 64x64 grid for the distances for one EFM, across all samples
                    single_site_distances = np.squeeze(efm_distances[:,0,:,:,f])
                    
                    #get the max
                    dist_max = np.max(np.max(np.max(single_site_distances)))

                    #get the min
                    dist_min = np.min(np.min(np.min(single_site_distances)))

                    #min_max normalize
                    single_site_norm_distance = (single_site_distances-dist_min)/(dist_max-dist_min)
                    
                    #store the data
                    x_dict.update({'%s_distances'%fm:single_site_norm_distance})
                    del single_site_norm_distance, dist_max, dist_min, single_site_distances

            print('the inputs in order: for verification with tf inputs, and their expected shape')
            for key in x_dict:
                print(key,x_dict[key].shape)
            
            #generate the targets
            fill_size = args.kernel_size
            merlin_list = []
            for key in merlin_keys:
                merlin_np = ds[key].values
                merlin_np[merlin_np>0] = 1.0
                merlin_np[merlin_np==0] = 0.0
                for i in range(merlin_np.shape[0]):
                    for t in range(4):
                        temp_ltg_data = merlin_np[i,t,:,:]
                        filled_ltg_data = binary_fill_holes(temp_ltg_data)
                        filled_ltg_data = binary_closing(filled_ltg_data,structure=np.ones((fill_size,fill_size)))
                        merlin_np[i,t,:,:] = filled_ltg_data
                        del temp_ltg_data, filled_ltg_data
                merlin_list.append(merlin_np)
            ltg_np = np.stack(merlin_list,axis=-1)
            data_tuple = (x_dict,ltg_np)
            tfds = tf.data.Dataset.from_tensor_slices(data_tuple)
            save_dir='/scratch/bmac87/'
            tfds.save(save_dir+'%s_%s.tfds'%(args.exp_pre,year))
            del merlin_list, merlin_np, ltg_np, data_tuple
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
    if args.shuffle>0:
        train_tfds=train_tfds.shuffle(buffer_size=args.shuffle)
    train_tfds=train_tfds.batch(args.batch,drop_remainder=True)
    if args.cache==True:
        train_tfds.cache()
    else:
        train_tfds=train_tfds.prefetch(tf.data.AUTOTUNE)

    val_tfds = tf.data.Dataset.load(data_dir+val_files[0])
    if args.shuffle>0:
        val_tfds=val_tfds.shuffle(buffer_size=args.shuffle)
    val_tfds=val_tfds.batch(args.batch,drop_remainder=True)
    if args.cache==True:
        val_tfds.cache()
    else:
        val_tfds=val_tfds.prefetch(tf.data.AUTOTUNE)

    test_tfds = tf.data.Dataset.load(data_dir+test_files[0])
    if args.shuffle>0:
        test_tfds=test_tfds.shuffle(buffer_size=args.shuffle)
    test_tfds=test_tfds.batch(args.batch)
    test_tfds=test_tfds.prefetch(tf.data.AUTOTUNE)
    return train_tfds, val_tfds, test_tfds

def visualize_tfds(args):
    year='2021'
    load_dir = '/scratch/bmac87/'
    fname = '%s_%s.tfds'%(args.exp_pre,year)
    test_tfds = tf.data.Dataset.load(load_dir+fname)
    takes = np.arange(1,2000)
    
    FM_keys = ['FM01', 'FM02', 'FM04', 'FM05', 'FM06', 'FM07', 'FM08', 'FM09', 'FM10', 'FM11', 'FM12', 'FM14', 'FM15', 'FM16', 'FM17', 'FM18', 'FM19', 'FM20', 'FM21', 'FM22', 'FM24', 'FM25', 'FM26', 'FM27', 'FM28', 'FM29', 'FM30', 'FM31', 'FM32', 'FM34', 'FM35']
    for tk in takes:
        for x,y in test_tfds.take(tk):
            pass
        for fm in FM_keys:
            one_efm = x[fm].numpy()
            efm_dist = x[fm+'_distances'].numpy()
            print(one_efm.shape)
            fig,axes = plt.subplots(nrows=1,ncols=3)
            axes[0].plot(one_efm[:,0])
            axes[1].plot(one_efm[:,1])
            axes[2].imshow(efm_dist)
            plt.savefig('/scratch/bmac87/takes_test_efm_images/one_efm_tfds_test_%s_take_%s.png'%(fm,tk))
            plt.close()

def determine_initial_bias(args):
    train_files = ['%s_2021.tfds'%(args.exp_pre),'%s_2022.tfds'%(args.exp_pre)]
    y_list=[]
    for file in train_files:
        tfds = tf.data.Dataset.load('/scratch/bmac87/'+file)
        tfds = tfds.batch(args.batch)
        y_true = tf.concat([y for _, y in tfds], axis=0).numpy()
        y_list.append(y_true)
    y_train = np.concatenate(y_list,axis=0)
    y_train_cc = y_train[:,:,:,:,0]
    cc_num = np.sum(y_train_cc)
    y_train_cg = y_train[:,:,:,:,1]
    cg_num = np.sum(y_train_cg)

    total_samples = y_train_cc.shape[0]*4*64*64

    p_ic = cc_num/total_samples
    p_cg = cg_num/total_samples

    print(p_ic,p_cg)

    b_ic = np.log(p_ic/(1-p_ic))
    b_cg = np.log(p_cg/(1-p_cg))
    print(b_ic,b_cg)

if __name__=='__main__':
    parser = create_parser()
    args = parser.parse_args()
    determine_initial_bias(args=args)
