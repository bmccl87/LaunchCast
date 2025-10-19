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
import tensorflow as tf

def load_hrrr_only_no_prior_tfds(args):

    print('loading the hrrr only data for no prior lightning: rotation: ',args.rotation)

    train_data_dict = pickle.load(open('/scratch/bmac87/HRRR_only_xy_102pixels/rot_%s_train.pkl'%args.rotation,'rb'))
    print('training dictionary loaded successfully')
    train_tfds = tf.data.Dataset.from_tensor_slices((train_data_dict['x_norm'],train_data_dict['y']))
    train_tfds = train_tfds.cache()
    train_tfds = train_tfds.batch(args.batch)
    print('batched training tfds loaded successfully')

    val_data_dict = pickle.load(open('/scratch/bmac87/HRRR_only_xy_102pixels/rot_%s_val.pkl'%args.rotation,'rb'))
    print('validation dictionary loaded successfully')
    val_tfds = tf.data.Dataset.from_tensor_slices((val_data_dict['x_norm'],val_data_dict['y']))
    val_tfds = val_tfds.cache()
    val_tfds = val_tfds.batch(args.batch)
    print('batched val tfds loaded successfully')

    
    test_data_dict = pickle.load(open('/scratch/bmac87/HRRR_only_xy_102pixels/rot_%s_test.pkl'%args.rotation,'rb'))
    print('test dictionary loaded successfully')
    test_tfds = tf.data.Dataset.from_tensor_slices((test_data_dict['x_norm'],test_data_dict['y']))
    test_tfds = test_tfds.cache()
    test_tfds = test_tfds.batch(args.batch)
    print('batched test tfds loaded successfully')

    return train_tfds, val_tfds, test_tfds

def load_y2y(args):

    train_data_dict = pickle.load(open('/scratch/bmac87/LC_y2y_102_pixels/rot_%s_train.pkl'%args.rotation,'rb'))
    print('training dictionary loaded successfully')
    train_tfds = tf.data.Dataset.from_tensor_slices((train_data_dict['y'],train_data_dict['y']))
    train_tfds = train_tfds.cache()
    train_tfds = train_tfds.batch(args.batch)
    print('batched training tfds loaded successfully')

    val_data_dict = pickle.load(open('/scratch/bmac87/LC_y2y_102_pixels/rot_%s_val.pkl'%args.rotation,'rb'))
    print('validation dictionary loaded successfully')
    val_tfds = tf.data.Dataset.from_tensor_slices((val_data_dict['y'],val_data_dict['y']))
    val_tfds = val_tfds.cache()
    val_tfds = val_tfds.batch(args.batch)
    print('batched val tfds loaded successfully')

    test_data_dict = pickle.load(open('/scratch/bmac87/LC_y2y_102_pixels/rot_%s_test.pkl'%args.rotation,'rb'))
    print('test dictionary loaded successfully')
    test_tfds = tf.data.Dataset.from_tensor_slices((test_data_dict['y'],test_data_dict['y']))
    test_tfds = test_tfds.cache()
    test_tfds = test_tfds.batch(args.batch)
    print('batched test tfds loaded successfully')

    return train_tfds, val_tfds, test_tfds

def load_y2y_64_2_16(args):

    train_data_dict = pickle.load(open('/scratch/bmac87/LC_y2y_64_2_16/train_data.pkl','rb'))
    train_input = train_data_dict['y_input']
    train_target = train_data_dict['y_target']
    train_input_np = np.zeros((train_input.shape[0],train_input.shape[2],train_input.shape[3],train_input.shape[1]*train_input.shape[-1]))
    train_target_np = np.zeros((train_input.shape[0],train_input.shape[2],train_input.shape[3],train_input.shape[1]*train_input.shape[-1]))
    

    print('training dictionary loaded successfully')
    train_tfds = tf.data.Dataset.from_tensor_slices((train_input,train_target))
    train_tfds = train_tfds.cache()
    train_tfds = train_tfds.batch(args.batch)
    print('batched training tfds loaded successfully')

    val_data_dict = pickle.load(open('/scratch/bmac87/LC_y2y_64_2_16/val_data.pkl','rb'))
    val_input = val_data_dict['y_input']
    val_target = val_data_dict['y_target']
    val_input = val_input.reshape((val_input.shape[0],val_input.shape[2],val_input.shape[3],val_input.shape[1]*val_input.shape[-1]))
    val_target = val_target.reshape((val_target.shape[0],val_target.shape[2],val_target.shape[3],val_target.shape[1]*val_target.shape[-1]))
    print('validation dictionary loaded successfully')
    val_tfds = tf.data.Dataset.from_tensor_slices((val_input,val_target))
    val_tfds = val_tfds.cache()
    val_tfds = val_tfds.batch(args.batch)
    print('batched val tfds loaded successfully')

    test_data_dict = pickle.load(open('/scratch/bmac87/LC_y2y_64_2_16/test_data.pkl','rb'))
    test_input = test_data_dict['y_input']
    test_target = test_data_dict['y_target']
    test_input = test_input.reshape((test_input.shape[0],test_input.shape[2],test_input.shape[3],test_input.shape[1]*test_input.shape[-1]))
    test_target = test_target.reshape((test_target.shape[0],test_target.shape[2],test_target.shape[3],test_target.shape[1]*test_target.shape[-1]))
    print('test dictionary loaded successfully')
    test_tfds = tf.data.Dataset.from_tensor_slices((test_input,test_target))
    test_tfds = test_tfds.cache()
    test_tfds = test_tfds.batch(args.batch)

    print('batched test tfds loaded successfully')
    return train_tfds, val_tfds, test_tfds

def visualize_y2y():
    train_data_dict = pickle.load(open('/scratch/bmac87/LC_y2y/rot_1_train.pkl','rb'))
    x = train_data_dict['x']
    y = train_data_dict['y']
    for i in range(x.shape[0]):
        if i%50==0:
            plt.figure()
            plt.imshow(x[i,0,:,:,0])
            plt.savefig('./images/x_%s_0.png'%i)
            plt.close()
            plt.figure()
            plt.imshow(x[i,1,:,:,0])
            plt.savefig('./images/x_%s_1.png'%i)
            plt.close()
            plt.figure()
            plt.imshow(x[i,2,:,:,0])
            plt.savefig('./images/x_%s_2.png'%i)
            plt.close()
            plt.figure()
            plt.imshow(x[i,3,:,:,0])
            plt.savefig('./images/x_%s_3.png'%i)
            plt.close()

            plt.figure()
            plt.imshow(y[i,0,:,:,0])
            plt.savefig('./images/y_%s_0.png'%i)
            plt.close()
            plt.figure()
            plt.imshow(y[i,1,:,:,0])
            plt.savefig('./images/y_%s_1.png'%i)
            plt.close()
            plt.figure()
            plt.imshow(y[i,2,:,:,0])
            plt.savefig('./images/y_%s_2.png'%i)
            plt.close()
            plt.figure()
            plt.imshow(y[i,3,:,:,0])
            plt.savefig('./images/y_%s_3.png'%i)
            plt.close()

if __name__=='__main__':
    load_y2y_64_2_16()