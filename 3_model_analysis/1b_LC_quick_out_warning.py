import numpy as np
import os
import xarray as xr
import shutil
import cartopy.crs as ccrs
import pickle

from gewitter_functions import get_contingency_table,make_performance_diagram_axis,get_acc,get_pod,get_sr,csi_from_sr_and_pod
from sklearn.metrics import auc, precision_recall_curve

import tensorflow as tf

from LC_parser import *
from LC_models import Stack

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects

#outlines for text 
pe1 = [path_effects.withStroke(linewidth=1.5,
                            foreground="k")]
pe2 = [path_effects.withStroke(linewidth=1.5,
                            foreground="w")]

matplotlib.rcParams['axes.facecolor'] = [0.9,0.9,0.9] #makes a grey background to the axis face
matplotlib.rcParams['axes.labelsize'] = 24 #fontsize in pts
matplotlib.rcParams['axes.titlesize'] = 24 
matplotlib.rcParams['xtick.labelsize'] = 18 
matplotlib.rcParams['ytick.labelsize'] = 18 
matplotlib.rcParams['legend.fontsize'] = 18 
matplotlib.rcParams['legend.facecolor'] = '#f7f7f7'#light grey
matplotlib.rcParams['savefig.transparent'] = False

def save_outputs(args,save_dir='/scratch/bmac87/',from_logits=False,from_pickle=False,is_calibrated=False):
    """
    Returns the outputs in probability space. Not logits. 
    """
    model_dir = save_dir
    model_fname = 'model.keras'
    print(model_dir+model_fname)
    model = tf.keras.models.load_model(model_dir+model_fname,custom_objects={"Stack": Stack})

    data_dir = '/scratch/bmac87/'
    tfds_file = '%s_2024.tfds'%args.exp_pre
    test_tfds = tf.data.Dataset.load(data_dir+tfds_file)
    test_tfds = test_tfds.batch(args.batch)
    test_tfds = test_tfds.cache()
    y_true = tf.concat([y for _, y in test_tfds], axis=0).numpy()
    y_pred = np.zeros(y_true.shape)
    if from_logits==True:#in logit space
        if from_pickle==True:
            y_dict = pickle.load(open(save_dir+'labels_logit_outputs.pkl','rb'))
            y_pred_logits = y_dict['y_pred']
            if is_calibrated==True:
                calibration_dict = pickle.load(open(save_dir+'calibration_dict.pkl','rb'))
            else:
                T = 1
            for t,t_key in enumerate(['t0','t1','t2','t3']):
                if is_calibrated==True:
                    T = calibration_dict[t_key]
                y_pred[:,t,:,:] = np.squeeze(tf.nn.sigmoid(y_pred_logits[:,t,:,:]/T).numpy())
            y_dict = {'y_true':y_true,'y_pred':y_pred}
            pickle.dump(y_dict,open(save_dir+'labels_prob_outputs.pkl','wb'))
        else:
            y_pred_logits = model.predict(test_tfds)
            y_pred = tf.nn.sigmoid(y_pred_logits.numpy())

    else:#already in probability space
        if from_pickle==True:
            y_dict = pickle.load(open(save_dir+'labels_prob_outputs.pkl','rb'))
        else:
            y_pred = model.predict(test_tfds)
            y_dict = {'y_true':y_true,'y_pred':y_pred}
            pickle.dump(y_dict,open(save_dir+'labels_prob_outputs.pkl','wb'))

    print('Probabilistic Outs Saved and Returned Successfully')
    return y_dict

def generate_merlin_output_figures_with_cartopy_cc_cg(y_true,y_pred,save_dir):
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)

    #get 100 random images for output
    num_images = 100
    total_images = y_true.shape[0]
    idxs = np.random.choice(total_images, size=num_images, replace=False)
    
    ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2d_MERLIN_binary_fed_target/202206/MERLIN_hrrr_202206302230.nc',engine='netcdf4')
    lat = ds['input_lat'].values
    lon = ds['input_lon'].values

    # # Define the color segments and corresponding values
    # colors = ["gray", "slateblue", "blue", "darkgreen", "green", "lightgreen", "yellow", "peru", "brown"]#spc product
    bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Create a colormap and norm
    # cmap = mcolors.ListedColormap(colors)
    cmap = plt.get_cmap('viridis')
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    y_true = y_true[idxs,:,:,:,:]
    y_true_cc = np.squeeze(y_true[:,:,:,:,0])
    print(y_true_cc.shape)
    y_true_cg = np.squeeze(y_true[:,:,:,:,1])
    print(y_true_cg.shape)

    y_pred = y_pred[idxs,:,:,:,:]
    y_pred_cc = np.squeeze(y_pred[:,:,:,:,0])
    print(y_pred_cc.shape)

    y_pred_cg = np.squeeze(y_pred[:,:,:,:,1])
    print(y_pred_cg.shape)

    for i in range(len(idxs)):
        y_t_cc = y_true_cc[i,:,:,:]
        y_t_cc[y_t_cc<1] = np.nan
        y_t_cg = y_true_cg[i,:,:,:]
        y_t_cg[y_t_cg<1] =np.nan

        y_p_cc = y_pred_cc[i,:,:,:]
        y_p_cc[y_p_cc<.05] = np.nan
        y_p_cg = y_pred_cg[i,:,:,:]
        y_p_cg[y_p_cg<.05] = np.nan

        fig,axes = plt.subplots(nrows=4,ncols=4,figsize=(40,60),subplot_kw={'projection': ccrs.PlateCarree()})
        for t in range(4):
            im0 = axes[t,0].pcolormesh(lon,lat,y_p_cc[t,:,:]*100,cmap=cmap,norm=norm)
            axes[t,0].set_xticks([],[])
            axes[t,0].set_yticks([],[])
            axes[t,0].coastlines(color='black')
            plt.colorbar(im0,ax=axes[t,0],label='IC Prediction')

            im1 = axes[t,1].pcolormesh(lon,lat,y_t_cc[t,:,:],cmap='binary')
            axes[t,1].set_xticks([],[])
            axes[t,1].set_yticks([],[])
            axes[t,1].coastlines(color='black')
            plt.colorbar(im1,ax=axes[t,1],label='IC Labels')

            im2 = axes[t,2].pcolormesh(lon,lat,y_p_cg[t,:,:],cmap=cmap,norm=norm)
            axes[t,2].set_xticks([],[])
            axes[t,2].set_yticks([],[])
            axes[t,2].coastlines(color='black')
            plt.colorbar(im2,ax=axes[t,2],label='CG Prediction')

            im3 = axes[t,3].pcolormesh(lon,lat,y_t_cg[t,:,:]*100,cmap='binary')
            axes[t,3].set_xticks([],[])
            axes[t,3].set_yticks([],[])
            axes[t,3].coastlines(color='black')
            plt.colorbar(im3,ax=axes[t,3],label='CG Labels')

        axes[0,0].set_ylabel('00-15 mins',fontsize=24)
        axes[1,0].set_ylabel('15-30 mins',fontsize=24)
        axes[2,0].set_ylabel('30-45 mins',fontsize=24)
        axes[3,0].set_ylabel('45-60 mins',fontsize=24)

        plt.tight_layout()
        plt.savefig(save_dir+'cartopy_out_%s.png'%i)
        plt.close()

def make_warning_outputs(y_true,y_pred,save_dir):
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)

    #get 100 random images for output
    num_images = 100
    total_images = y_true.shape[0]
    idxs = np.random.choice(total_images, size=num_images, replace=False)
    
    ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2d_MERLIN_binary_fed_target/202206/MERLIN_hrrr_202206302230.nc',engine='netcdf4')
    lat = ds['input_lat'].values
    lon = ds['input_lon'].values

    # # Define the color segments and corresponding values
    # colors = ["gray", "slateblue", "blue", "darkgreen", "green", "lightgreen", "yellow", "peru", "brown"]#spc product
    bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Create a colormap and norm
    # cmap = mcolors.ListedColormap(colors)
    cmap = plt.get_cmap('viridis')
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    y_true = y_true[idxs,:,:,:]
    y_pred = y_pred[idxs,:,:,:]

    for i in range(len(idxs)):
        y_t = np.squeeze(y_true[i,:,:,:])
        y_t[y_t<1] = np.nan

        y_p = np.squeeze(y_pred[i,:,:,:])
        y_p[y_p<.05] = np.nan

        fig,axes = plt.subplots(nrows=4,ncols=2,figsize=(11,8.5),subplot_kw={'projection': ccrs.PlateCarree()})
        for t in range(4):
            im0 = axes[t,0].pcolormesh(lon,lat,y_p[t,:,:]*100,cmap=cmap,norm=norm)
            axes[t,0].set_xticks([],[])
            axes[t,0].set_yticks([],[])
            axes[t,0].coastlines(color='black')
            plt.colorbar(im0,ax=axes[t,0],label='Prediction')

            im1 = axes[t,1].pcolormesh(lon,lat,y_t[t,:,:],cmap='binary')
            axes[t,1].set_xticks([],[])
            axes[t,1].set_yticks([],[])
            axes[t,1].coastlines(color='black')
            plt.colorbar(im1,ax=axes[t,1],label='Labels')

        axes[0,0].set_ylabel('00-15 mins',fontsize=24)
        axes[1,0].set_ylabel('15-30 mins',fontsize=24)
        axes[2,0].set_ylabel('30-45 mins',fontsize=24)
        axes[3,0].set_ylabel('45-60 mins',fontsize=24)

        plt.tight_layout()
        plt.savefig(save_dir+'cartopy_out_%s.png'%i)
        plt.close()

if __name__=='__main__':

    parser = create_parser()
    args = parser.parse_args()
    args.batch = 4
    exp_name = 'HRRR_MRMS_GLM_no_EFM_no_Zdr_2_MERLIN_total_filled_9_rot_1_early_stop_True_conv_act_relu_L2_0.0__BN_False_SD_0.000_pool_max__2dkernel_8__last_act_sigmoid_epochs_250__no_noise_lrate_0.000000100__label_from_logits_focal_2_econders_bias_init_one_out'
    exp_dir = '/scratch/bmac87/LaunchCast_scratch/results/LC_Forecast_3DConv/'

    load_dir = exp_dir+exp_name+'/'
    from_pickle=True
    from_logits=True
    is_calibrated=True

    y_dict = save_outputs(args=args,
                            save_dir=load_dir,
                            from_logits=from_logits,
                            from_pickle=from_pickle,
                            is_calibrated=is_calibrated)

    img_dir = load_dir+'output_figures_cartopy/'
    if os.path.isdir(img_dir)==False:
        os.makedirs(img_dir)
    make_warning_outputs(y_true=y_dict['y_true'], y_pred=y_dict['y_pred'],save_dir=img_dir)
