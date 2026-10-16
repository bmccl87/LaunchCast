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

EFM_MASK_VALUE=-999999.0
ZDR_MASK_VALUE=-99
ZDR_FILL_VALUE=-5

def min_max_norm_2d(args,key='group_area',data_np=[]):
    print('normalizing',key)
    radar_keys = args.Z_keys + args.Zdr_keys + args.VI_keys
    min_max_dict = pickle.load(open('forecast_min_maxes.pkl','rb'))

    data_max =  min_max_dict[key]['max']
    data_min = min_max_dict[key]['min']

    if key=='group_area' or key=='group_energy':#GLM
        data_np[data_np==0.0] = np.nan
        data_norm = (data_np-data_min)/(data_max-data_min)
        data_norm[np.isnan(data_norm)] = 0.0

    elif key in radar_keys:#MRMS
        data_np[data_np<=0.0] = np.nan
        data_norm = (data_np-data_min)/(data_max-data_min)
        data_norm[np.isnan(data_norm)] = 0.0
    
    else:#HRRR
        data_norm=(data_np-data_min)/(data_max-data_min)

    return data_norm

def z_score_norm(args,key='divergence_10m',data_np=[]):
    print('Z-SCORE NORMALIZING')
    print(key)
    min_max_dict = pickle.load(open('forecast_min_maxes.pkl','rb'))
    data_mean =  min_max_dict[key]['mean']
    data_std = min_max_dict[key]['std']
    print(data_mean)
    print(data_std)
    data_norm = (data_np-data_mean)/data_std
    print(np.nanmin(data_norm))
    return data_norm

def norm_efm_ts(key='FM01',data_np=[],stat='mean'):

    #load the mins and maxes of the EFM data
    min_max_dict = pickle.load(open('efm_forecast_min_max.pkl','rb'))
    print(min_max_dict[key])
    if stat=='mean':
        p99 = min_max_dict[key][stat]['p99']
        data_np = np.clip(data_np, -p99, p99)
    if stat=='rms':
        p99 = min_max_dict[key][stat]['p99']
        data_np = np.clip(data_np, 0, p99)
    data_mean =  min_max_dict[key][stat]['mean']
    data_std = min_max_dict[key][stat]['std']
    data_norm = (data_np-data_mean)/data_std
    return data_norm

def log_norm(key='FM01',data_np=[],stat='mean'):
    log_x = np.log(data_np+1e-6)
    min_max_dict = pickle.load(open('efm_forecast_min_max.pkl','rb'))
    print(min_max_dict[key])
    log_mean = min_max_dict[key][stat]['log_mean']
    log_std = min_max_dict[key][stat]['log_std']
    z = (log_x-log_mean)/log_std
    return z


def build_rotations(rot=1):
    train_files = ['LC_2021.tfds','LC_2022.tfds']
    val_files = ['LC_2023.tfds']
    test_files = ['LC_2024.tfds']
    return train_files, val_files, test_files

def norm_2_tfds(args):
    nc_dir = '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3_data_processing/'

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
            ds = xr.open_dataset(nc_dir+'LC_Forecast_Final_log1p_GLM_VII_%s.nc'%year,engine='netcdf4')
            if args.use_MRMS==True:
                if args.use_Z==True:
                    for key in z_keys:
                        data_np = ds[key].values
                        data_np = np.maximum(data_np, 0.0)
                        data_norm_np = z_score_norm(args=args,key=key,data_np=data_np)
                        for t,t_key in enumerate(time_keys):
                            x_dict.update({'MRMS_%s_%s'%(key,t_key):data_norm_np[:,t,:,:]})
                        del data_np, data_norm_np

                if args.use_VI==True:
                    for key in vi_keys:
                        data_np = ds[key].values
                        data_norm_np = z_score_norm(args=args,key=key,data_np=data_np)
                        for t,t_key in enumerate(time_keys):
                            x_dict.update({'MRMS_%s_%s'%(key,t_key):data_norm_np[:,t,:,:]})
                        del data_np, data_norm_np

                if args.use_Zdr==True:
                    for key in zdr_keys:
                        print('ZDR:',key)
                        data_np = ds[key].values
                        data_np[data_np==ZDR_MASK_VALUE]=np.nan
                        data_norm_np = z_score_norm(args=args,key=key,data_np=data_np)
                        print(key)
                        print(np.nanmax(data_norm_np))
                        print(np.nanmin(data_norm_np))
                        print()
                        data_norm_np[np.isnan(data_norm_np)] = ZDR_FILL_VALUE
                        print(np.max(data_norm_np))
                        print(np.min(data_norm_np))
                        print(np.sum(np.isnan(data_norm_np)))
                        mask_np = (~(data_norm_np==ZDR_FILL_VALUE)).astype(np.float32)
                        for t,t_key in enumerate(time_keys):
                            x_dict.update({'MRMS_%s_%s'%(key,t_key):data_norm_np[:,t,:,:]})
                            x_dict.update({'MRMS_%s_mask_%s'%(key,t_key):mask_np[:,t,:,:]})
                        del data_np, data_norm_np, mask_np
                        print()
                        
            if args.use_GLM==True:
                for key in glm_keys:
                    data_np = ds[key].values
                    data_norm_np = z_score_norm(args=args,key=key,data_np=data_np)
                    for t,t_key in enumerate(time_keys):
                        x_dict.update({'GLM_%s_%s'%(key,t_key):data_norm_np[:,t,:,:]})
                    del data_np, data_norm_np

            if args.use_HRRR==True:
                for f,ftr in enumerate(args.hrrr_features):
                    hrrr_data_np = ds[ftr].values
                    data_norm_np = z_score_norm(args=args,key=ftr,data_np=hrrr_data_np)
                    for t,t_key in enumerate(time_keys):
                        x_dict.update({'HRRR_%s_%s'%(ftr,t_key):data_norm_np[:,t,:,:]})
                    del data_norm_np

            if args.use_EFM==True:
                data_store = np.zeros((len(ds['sample_time'])*120,len(args.efm_keys),len(ds['efm_stats'])))
                data_store2 = np.zeros((len(ds['sample_time'])*120,len(args.efm_keys),len(ds['efm_stats'])))
                for k,fm_key in enumerate(args.efm_keys):
                    data_np = ds[fm_key].values
                    data_norm_np = np.zeros((data_np.shape[0],120,data_np.shape[-1]))
                    print(data_np.shape)
                    for f,ftr in enumerate(args.efm_stats):
                        temp_data = data_np[:,0:120,f].copy()
                        temp_data[temp_data==EFM_MASK_VALUE]=np.nan
                        # ['count_gt_thresh','max_abs', 'mean', 'rms', 'std','var']
                        """
                        This was tuned by looking at the percentiles of each statistics. 95, 99, and 99.9. 
                        It was tuned to allow the non-linearities to not become continuously saturated.
                        Z-score, z-score with clipping, or log-transform then z-score. 
                        """
                        if ftr=='count_gt_thresh' or ftr=='mean' or ftr=='rms':#count greater than 1000
                            temp_norm = norm_efm_ts(key=fm_key,stat=ftr,data_np=temp_data)
                        elif ftr=='max_abs' or ftr=='std' or ftr=='var':
                            temp_norm = log_norm(key=fm_key,stat=ftr,data_np=temp_data)
                        data_store[:,k,f]=temp_norm.ravel()
                        data_store2[:,k,f]=temp_data.ravel()

                        temp_norm[~np.isfinite(temp_norm)]=EFM_MASK_VALUE
                        data_norm_np[:,:,f]=temp_norm
                        del temp_data, temp_norm
                    x_dict.update({fm_key:data_norm_np})

            dist_ds = xr.open_dataset('./efm_distance_proximity.nc',engine='netcdf4')
            factor = 5*1852 #5nm in meters
            dist_np = dist_ds['distances'].values#16,16,26
            dist_np = np.moveaxis(dist_np, 2, 0)#26,16,16
            dist_np = np.exp(-dist_np/factor)

            dist_4d = dist_np[None, :,:,:]          # (1,26,16,16)
            dist_4d = np.repeat(dist_4d, len(ds['sample_time'].values), axis=0)
            x_dict.update({'exp_weighted_distance_input':dist_4d})

            print('the inputs in order: for verification with tf inputs, and their expected shape')
            for key in x_dict:
                print(key,x_dict[key].shape)
            y_true = ds['targets'].values[:,:,24:40,24:40]
            print(y_true.shape)
            data_tuple = (x_dict,y_true)
            tfds = tf.data.Dataset.from_tensor_slices(data_tuple)
            save_dir='/scratch/bmac87/'
            tfds.save(save_dir+'%s_%s.tfds'%(args.exp_pre,year))
            del data_tuple
            del x_dict, tfds

def norm_2_tfds_hyper(args):

    nc_dir = '/scratch/bmac87/'

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

    x_dict = OrderedDict()
    ds = xr.open_dataset(nc_dir+'hyper_train_ds.nc',engine='netcdf4')
    if args.use_MRMS==True:
        if args.use_Z==True:
            for key in z_keys:
                data_np = ds[key].values
                data_norm_np = z_score_norm(args=args,key=key,data_np=data_np)
                for t,t_key in enumerate(time_keys):
                    x_dict.update({'MRMS_%s_%s'%(key,t_key):data_norm_np[:,t,:,:]})
                del data_np, data_norm_np

        if args.use_VI==True:
            for key in vi_keys:
                data_np = ds[key].values
                data_norm_np = z_score_norm(args=args,key=key,data_np=data_np)
                for t,t_key in enumerate(time_keys):
                    x_dict.update({'MRMS_%s_%s'%(key,t_key):data_norm_np[:,t,:,:]})
                del data_np, data_norm_np

        if args.use_Zdr==True:
            for key in zdr_keys:
                print('ZDR:',key)
                data_np = ds[key].values
                data_np[data_np==ZDR_MASK_VALUE]=np.nan
                data_norm_np = z_score_norm(args=args,key=key,data_np=data_np)
                print(key)
                print(np.nanmax(data_norm_np))
                print(np.nanmin(data_norm_np))
                print()
                data_norm_np[np.isnan(data_norm_np)] = ZDR_FILL_VALUE
                print(np.max(data_norm_np))
                print(np.min(data_norm_np))
                print(np.sum(np.isnan(data_norm_np)))
                mask_np = (~(data_norm_np==ZDR_FILL_VALUE)).astype(np.float32)
                for t,t_key in enumerate(time_keys):
                    x_dict.update({'MRMS_%s_%s'%(key,t_key):data_norm_np[:,t,:,:]})
                    x_dict.update({'MRMS_%s_mask_%s'%(key,t_key):mask_np[:,t,:,:]})
                del data_np, data_norm_np, mask_np
                print()
                
    if args.use_GLM==True:
        for key in glm_keys:
            data_np = ds[key].values
            data_norm_np = z_score_norm(args=args,key=key,data_np=data_np)
            for t,t_key in enumerate(time_keys):
                x_dict.update({'GLM_%s_%s'%(key,t_key):data_norm_np[:,t,:,:]})
            del data_np, data_norm_np

    if args.use_HRRR==True:
        for f,ftr in enumerate(args.hrrr_features):
            hrrr_data_np = ds[ftr].values
            data_norm_np = z_score_norm(args=args,key=ftr,data_np=hrrr_data_np)
            for t,t_key in enumerate(time_keys):
                x_dict.update({'HRRR_%s_%s'%(ftr,t_key):data_norm_np[:,t,:,:]})
            del data_norm_np

    if args.use_EFM==True:
        
        data_store = np.zeros((len(ds['sample_time'])*120,len(args.efm_keys),len(ds['efm_stats'])))
        data_store2 = np.zeros((len(ds['sample_time'])*120,len(args.efm_keys),len(ds['efm_stats'])))
        for k,fm_key in enumerate(args.efm_keys):
            data_np = ds[fm_key].values
            data_norm_np = np.zeros((data_np.shape[0],120,data_np.shape[-1]))
            print(data_np.shape)
            for f,ftr in enumerate(args.efm_stats):
                temp_data = data_np[:,0:120,f].copy()
                temp_data[temp_data==EFM_MASK_VALUE]=np.nan
                # ['count_gt_thresh','max_abs', 'mean', 'rms', 'std','var']
                """
                This was tuned by looking at the percentiles of each statistics. 95, 99, and 99.9. 
                It was tuned to allow the non-linearities to not become continuously saturated.
                Z-score, z-score with clipping, or log-transform then z-score. 
                """
                if ftr=='count_gt_thresh' or ftr=='mean' or ftr=='rms':#count greater than 1000
                    temp_norm = norm_efm_ts(key=fm_key,stat=ftr,data_np=temp_data)
                elif ftr=='max_abs' or ftr=='std' or ftr=='var':
                    temp_norm = log_norm(key=fm_key,stat=ftr,data_np=temp_data)
                data_store[:,k,f]=temp_norm.ravel()
                data_store2[:,k,f]=temp_data.ravel()

                temp_norm[~np.isfinite(temp_norm)]=EFM_MASK_VALUE
                data_norm_np[:,:,f]=temp_norm
                del temp_data, temp_norm
            x_dict.update({fm_key:data_norm_np})

    dist_ds = xr.open_dataset('./efm_distance_proximity.nc',engine='netcdf4')
    factor = 5*1852 #5nm in meters
    dist_np = dist_ds['distances'].values#16,16,26
    dist_np = np.moveaxis(dist_np, 2, 0)#26,16,16
    dist_np = np.exp(-dist_np/factor)

    dist_4d = dist_np[None, :,:,:]          # (1,26,16,16)
    dist_4d = np.repeat(dist_4d, len(ds['sample_time'].values), axis=0)
    x_dict.update({'exp_weighted_distance_input':dist_4d})

    print('the inputs in order: for verification with tf inputs, and their expected shape')
    for key in x_dict:
        print(key,x_dict[key].shape)
    y_true = ds['targets'].values[:,:,24:40,24:40]
    print(y_true.shape)
    data_tuple = (x_dict,y_true)
    tfds = tf.data.Dataset.from_tensor_slices(data_tuple)
    save_dir='/scratch/bmac87/'
    tfds.save(save_dir+'hyper_train.tfds')
    del data_tuple
    del x_dict, tfds

def tfds_2_model(args,is_hyper=False):

    print('loading the tfds-s')
    data_dir=args.data_path
    train_files = ['%s_2021.tfds'%(args.exp_pre),'%s_2022.tfds'%(args.exp_pre)]
    
    if is_hyper==True:
        train_tfds = tf.data.Dataset.load('/scratch/bmac87/hyper_train.tfds')
    else:
        #shuffle, batch, prefetch
        tr1 = tf.data.Dataset.load(data_dir+train_files[0])
        tr2 = tf.data.Dataset.load(data_dir+train_files[1])
        train_tfds = tr1.concatenate(tr2)
    
    val_files = ['%s_2023.tfds'%(args.exp_pre)]
    test_files = ['%s_2024.tfds'%(args.exp_pre)]

    if args.cache==True:
        if args.use_lscratch==True:
            lscratch_dir = os.environ.get("LSCRATCH")
            print('CACHE-ING TO LSCRATCH')
            print(lscratch_dir)
            cache_path = os.path.join(lscratch_dir, "lc_train_cache")
            os.makedirs(cache_path)
            print("Using cache path:", cache_path)
            train_tfds=train_tfds.cache(cache_path)
        else:
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

def build_hyper_ds(args):
    print('building the hyperparameter tfdss')
    nc_dir = '/ourdisk/hpc/ai2es/bmac87/scratch_dump/'
    fname1 = 'LC_Forecast_Final_log1p_GLM_VII_2021.nc'
    fname2 = 'LC_Forecast_Final_log1p_GLM_VII_2022.nc'
    ds1 = xr.open_dataset(nc_dir+fname1,engine='netcdf4')
    ds2 = xr.open_dataset(nc_dir+fname2,engine='netcdf4')
    hyper_ds = xr.concat([ds1,ds2],dim='sample_time')
    del ds1, ds2

    # hyper_ds = xr.open_dataset(nc_dir+'hyp_ds.nc',engine='netcdf4')
    sample_times = hyper_ds['sample_time'].values
    print('total number of samples')
    print(len(sample_times))
    sub_percent=.10
    n_sub = int(np.floor(len(sample_times)*sub_percent))

    glm = hyper_ds["group_area"].values
    target = hyper_ds["targets"].values

    # 1) previous hour GLM is zero everywhere across all 4 prior times and all grid points
    no_prior_glm = np.all(glm == 0, axis=(1, 2, 3))

    # 2) lightning exists somewhere in first prediction window
    # example: target dims (sample, forecast_time, y, x, flash_type)
    first_pred_has_ltng = np.any(target[:, 0, ...] > 0, axis=tuple(range(1, target[:, 0, ...].ndim)))

    # 3) first flash samples
    first_flash_mask = no_prior_glm & first_pred_has_ltng
    first_flash_idxs = np.where(first_flash_mask)[0]
    n_first_flash = len(first_flash_idxs)
    ff_sub = int(np.floor(n_first_flash*sub_percent))
    print(ff_sub,'num ff to hyper on')

    rng = np.random.default_rng(42)  # reproducible
    ff_subset = rng.choice(first_flash_idxs,
                        size=ff_sub,
                        replace=False)
    ff_vts = sample_times[ff_subset]
    
    
    all_idxs = np.arange(glm.shape[0])
    not_first_flash = np.setdiff1d(all_idxs, first_flash_idxs)

    t0_ltng = np.any(target[:, 0, ...] > 0,axis=tuple(range(1, target[:, 0, ...].ndim)))
    t_last_no_ltng = ~np.any(target[:, -1, ...] > 0,axis=tuple(range(1, target[:, -1, ...].ndim)))

    last_flash_mask = t0_ltng & t_last_no_ltng
    last_flash_idxs = np.where(last_flash_mask)[0]
    n_last_flash = len(last_flash_idxs)
    lf_sub = int(np.floor(n_last_flash*sub_percent))
    print(lf_sub,'last flashes to use')
    lf_subset = rng.choice(last_flash_idxs,
                        size=lf_sub,
                        replace=False)
    lf_vts = sample_times[lf_subset]

    not_last_or_first_flash_idxs = np.setdiff1d(not_first_flash,last_flash_idxs)
    n_remaining_sub = int(np.floor(n_sub - lf_sub-ff_sub))
    n_remaining_frac = int(np.floor(n_remaining_sub*sub_percent))

    print('total remaining to use')
    print(n_remaining_sub)
    cont_subset = rng.choice(not_last_or_first_flash_idxs,
                        size=n_remaining_sub,
                        replace=False)
    cont_vts = sample_times[cont_subset]
    hyper_vts = np.concatenate([cont_vts,lf_vts,ff_vts])

    print(hyper_vts)
    print(len(hyper_vts))
    print('saving the new training dataset for hyperband')
    hyper_ds.sel(sample_time=hyper_vts).to_netcdf('/scratch/bmac87/hyper_train_ds.nc',engine='netcdf4')
    print('SAVED SUCCESSFULLY')
    hyper_ds.close()

if __name__=='__main__':
    parser = create_parser()
    args = parser.parse_args()
    norm_2_tfds(args=args)
    # build_hyper_ds(args=args)