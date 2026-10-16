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
from LC_model_64216 import Stack, FlattenStations, UnflattenStations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects

#outlines for text 
pe1 = [path_effects.withStroke(linewidth=1.5,foreground="k")]
pe2 = [path_effects.withStroke(linewidth=1.5,foreground="w")]

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
    model = tf.keras.models.load_model(model_dir+model_fname,custom_objects={"Stack": Stack,"FlattenStations":FlattenStations,"UnflattenStations":UnflattenStations},compile=False)

    data_dir = args.data_path
    tfds_file = '%s_2024.tfds'%args.exp_pre
    test_tfds = tf.data.Dataset.load(data_dir+tfds_file)
    test_tfds = test_tfds.batch(args.batch)
    test_tfds = test_tfds.cache()
    y_true = tf.concat([y for _, y in test_tfds], axis=0).numpy()
    y_pred = np.zeros(y_true.shape)
    if from_logits==True:#in logit space
        if from_pickle==True:#logits were saved during training
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

        else:#logits outputs weren't saved during training
            y_pred_logits = model.predict(test_tfds)
            pickle.dump(y_pred_logits,open(save_dir+'labels_logit_outputs.pkl','wb'))
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

    else:#already in probability space
        if from_pickle==True:
            y_dict = pickle.load(open(save_dir+'labels_prob_outputs.pkl','rb'))
        else:
            y_pred = model.predict(test_tfds)
            y_dict = {'y_true':y_true,'y_pred':y_pred}
            pickle.dump(y_dict,open(save_dir+'labels_prob_outputs.pkl','wb'))

    print('Probabilistic Outs Saved and Returned Successfully')
    return y_dict

def make_100_random_outputs(y_true,y_pred,save_dir):
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)

    #get 100 random images for output
    num_images = 100
    total_images = y_true.shape[0]
    idxs = np.random.choice(total_images, size=num_images, replace=False)
    
    ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/scratch_dump/LC_Forecast_Final_log1p_GLM_VII_2021.nc',engine='netcdf4')
    lat = ds['lat'].values[24:40,24:40]
    lon = ds['lon'].values[24:40,24:40]

    # # Define the color segments and corresponding values
    # colors = ["gray", "slateblue", "blue", "darkgreen", "green", "lightgreen", "yellow", "peru", "brown"]#spc product
    bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Create a colormap and norm
    # cmap = mcolors.ListedColormap(colors)
    cmap = plt.get_cmap('viridis')
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    y_true = y_true[idxs,:,:,:]
    y_pred = y_pred[idxs,:,:,:]
    print(y_true.shape)

    for i in range(len(idxs)):
        y_t = np.squeeze(y_true[i,:,:,:])
        plot_targets = np.ma.masked_where(y_t == 0, y_t)

        y_p = np.squeeze(y_pred[i,:,:,:])
        y_p[y_p<.05] = np.nan

        fig,axes = plt.subplots(nrows=4,ncols=2,figsize=(11,8.5),subplot_kw={'projection': ccrs.PlateCarree()})
        for t in range(4):
            im0 = axes[t,0].pcolormesh(lon,lat,y_p[t,:,:]*100,cmap=cmap,norm=norm)
            axes[t,0].set_xticks([],[])
            axes[t,0].set_yticks([],[])
            axes[t,0].coastlines(color='black')
            plt.colorbar(im0,ax=axes[t,0],label='Prediction')

            im1 = axes[t,1].pcolormesh(lon,lat,plot_targets[t,:,:],cmap='gray')
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
    exp_name = args.exp_name
    exp_dir = '%s%s/'%(args.results_path,args.project)

    load_dir = exp_dir+exp_name+'/'
    from_pickle=True
    from_logits=True
    is_calibrated=True

    y_dict = save_outputs(args=args,
                            save_dir=load_dir,
                            from_logits=from_logits,
                            from_pickle=from_pickle,
                            is_calibrated=is_calibrated)
    y_dict = pickle.load(open(load_dir+'labels_prob_outputs.pkl','rb'))

    img_dir = load_dir+'output_figures_cartopy/'
    if os.path.isdir(img_dir)==False:
        os.makedirs(img_dir)
    make_100_random_outputs(y_true=y_dict['y_true'], y_pred=y_dict['y_pred'],save_dir=img_dir)
