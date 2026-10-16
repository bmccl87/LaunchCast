import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

import os
import xarray as xr
from LC_models import Stack
from LC_data_loader_case_study import *
from LC_parser import *
import pandas as pd
import copy

def main(args):

    model_path = args.results_path+'/'+args.project+'/'+args.exp_name+'/'+'model.keras'
    print(model_path)

    print('loading the model')
    model = tf.keras.models.load_model(model_path,custom_objects={"Stack": Stack})
    print('model loaded successfully')

    print('loading the case_study.tfds')
    tfds = tf.data.Dataset.load('/scratch/bmac87/scenario_tfds_s/case_study.tfds')
    tfds = tfds.batch(1)
    print('tfds loaded successfully')

    #layer names from model_plot.png; post activation
    latent_layers = ["efm_ts_encoder_Conv1D_0","efm_distance_encoder_2","NWP_encoder_conv3d_5_2_2d_kernel_2_time_kernel_1","Obs_encoder_conv3d_5_2_2d_kernel_2_time_kernel_1"]
    latent_dict = {}

    for layer in latent_layers:
        print(layer)

        print('getting the specific layer from the model')
        probe_layer = model.get_layer(layer)  # example
        print('success')
        print(layer)
        probe_layer_weights, probe_layer_bias = probe_layer.get_weights()
        print(probe_layer_weights.shape)
        print('max weight:',np.max(probe_layer_weights),'min weight:',np.min(probe_layer_weights))
        print()
        print()
        print(probe_layer_bias.shape)
        print('max bias:',np.max(probe_layer_bias),'min bias:',np.min(probe_layer_bias))
        print()
        print()
        latent_dict.update({layer:{'weights':probe_layer_weights,'bias':probe_layer_bias}})

    print('dumping the dict')
    pickle.dump(latent_dict,open('/scratch/bmac87/latent_dict_weights.pkl','wb'))
    print('success')

def vis_probe():

    latent_dict = pickle.load(open('/scratch/bmac87/latent_dict.pkl','rb'))
    keys = []
    for key in latent_dict:
        keys.append(key)
    
    cmap = 'coolwarm'

    fig,axes = plt.subplots(len(keys),3,figsize=(21,21))
    for k,key in enumerate(keys):
        print(k,key)
        latent_output = latent_dict[key]
        print(latent_output.shape)
        print(np.min(latent_output),np.max(latent_output))
        if k>=2:
            latent_output=np.squeeze(latent_output)
        print(latent_output.shape)
        print(np.min(latent_output),np.max(latent_output))
        vmin = np.min(latent_output)
        vmax = np.max(latent_output)

        case_1_probe = np.expand_dims(latent_output[0,:],axis=0)
        im0 = axes[k,0].pcolormesh(case_1_probe,vmin=vmin,vmax=vmax,cmap=cmap)
        axes[k,0].set_yticks([],[])
        axes[k,0].set_xlabel('Latent Space')
        if k==0:
            axes[k,0].set_title('Case 1')
            axes[k,1].set_title('Case 2')
            axes[k,2].set_title('Delta')
        plt.colorbar(im0,ax=axes[k,0])

        case_2_probe = np.expand_dims(latent_output[1,:],axis=0)
        im1 = axes[k,1].pcolormesh(case_2_probe,vmin=vmin,vmax=vmax,cmap=cmap)
        axes[k,1].set_yticks([],[])
        axes[k,1].set_xlabel('Latent Space')
        plt.colorbar(im1,ax=axes[k,1])

        if k==1:
            axes[k,2].axis('off')
        else:
            delta = latent_output[1,:] - latent_output[0,:]
            delta=np.expand_dims(delta,axis=0)
            im2 = axes[k,2].pcolormesh(delta,vmin=vmin,vmax=vmax,cmap=cmap)
            plt.colorbar(im2,ax=axes[k,2])
            axes[k,2].set_yticks([],[])
            axes[k,2].set_xlabel('Latent Space')
            del delta, case_2_probe, case_1_probe, latent_output

        if k==0:
            ylabel='Latent EFM Time Series Encoder'
        if k==1:
            ylabel='Latent EFM Distances Encoder'
        if k==2:
            ylabel='Latent NWP Encoder'
        if k==3:
            ylabel='Latent Obs Encoder'
        axes[k,0].set_ylabel(ylabel)
    plt.savefig('./test_probe.png')
    plt.close()

if __name__=='__main__':

    parser = create_parser()
    args = parser.parse_args()
    main(args=args)
    # vis_probe()
    