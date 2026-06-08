import sys
import argparse
import pickle
import pandas as pd
import wandb
import socket
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shutil 
import cartopy.crs as ccrs
import xarray as xr
import os
import numpy as np

def generate_output_figures(args, y_true, y_pred, save_dir):
    
    #load the grid from the 2d_nc files for fancy plotting
    ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2d_MERLIN_binary_fed_target/202206/MERLIN_hrrr_202206302230.nc',engine='netcdf4')

    target_lat = ds['target_lat'].values
    target_lon = ds['target_lon'].values

    #load the test dataset from the pickle files
    test_dict = pickle.load(open('/scratch/bmac87/HRRR_only_xy_102pixels/rot_1_test.pkl','rb'))
    valid_times = test_dict['MERLIN_times']

    # # Define the color segments and corresponding values
    colors = ["gray", "slateblue", "blue", "darkgreen", "green", "lightgreen", "yellow", "peru", "brown"]#spc product
    bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    vmin = 10
    vmax = 100

    # Create a colormap and norm
    cmap = mcolors.ListedColormap(colors)
    # cmap = plt.get_cmap('viridis')
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for i in range(y_true.shape[0]):
            
        vt = valid_times[i]
        ts = pd.Timestamp(vt)
        minute = ts.minute
        hour = ts.hour
        day = ts.day
        month = ts.month
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        month = months[month-1]
        year = ts.year
        title_str = '%s:%s UTC %s %s %s'%(f"{hour:02}",f"{minute:02}",f"{day:02}",f"{month:02}",f"{year:04}")
        file_str = '%s%s%s%s%s.jpg'%(f"{year:04}",f"{ts.month:02}",f"{day:02}",f"{hour:02}",f"{minute:02}")
        
        fig,axes = plt.subplots(nrows=4,ncols=4,figsize=(40,40),subplot_kw={'projection': ccrs.PlateCarree()})
        for t in range(4):
            ltg_prob = y_true[i,t,:,:,0]*100
            ltg_prob[ltg_prob<=95] = np.nan
            im1 = axes[0,t].pcolormesh(target_lon,target_lat,ltg_prob,vmin=vmin,vmax=vmax,cmap=cmap)
            axes[0,t].coastlines()
            axes[0,t].set_title(title_str,fontsize=24)
            if t==0:
                axes[0,t].set_ylabel('CC Labels',fontsize=24)
            vt = vt+np.timedelta64(15,'m')
            ts = pd.Timestamp(vt)
            minute = ts.minute
            hour = ts.hour
            day = ts.day
            month = ts.month
            months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            month = months[month-1]
            year = ts.year
            title_str = '%s:%s UTC %s %s %s'%(f"{hour:02}",f"{minute:02}",f"{day:02}",f"{month:02}",f"{year:04}")

            ltg_prob = y_pred[i,t,:,:,0]*100
            ltg_prob[ltg_prob<=5] = np.nan
            im2 = axes[1,t].pcolormesh(target_lon,target_lat,ltg_prob,vmin=vmin,vmax=vmax,cmap=cmap)
            axes[1,t].coastlines()
            if t==0:
                axes[1,t].set_ylabel('CC Prediction',fontsize=24)
            
            ltg_prob = y_true[i,t,:,:,1]*100
            ltg_prob[ltg_prob<=95] = np.nan
            im3 = axes[2,t].pcolormesh(target_lon,target_lat,ltg_prob,vmin=vmin,vmax=vmax,cmap=cmap)
            axes[2,t].coastlines()
            if t==0:
                axes[2,t].set_ylabel('CG Labels',fontsize=24)

            ltg_prob = y_pred[i,t,:,:,1]*100
            ltg_prob[ltg_prob<=5] = np.nan
            im4 = axes[3,t].pcolormesh(target_lon,target_lat,ltg_prob,vmin=vmin,vmax=vmax,cmap=cmap)
            axes[3,t].coastlines()
            if t==0:
                axes[3,t].set_ylabel('CG Prediction',fontsize=24)

        cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])  # width = 50% of figure, height = 3%
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Lightning Probability',fontsize=24)
        cbar.set_ticks(bounds[:-1])
        cbar.ax.set_xticklabels([str(b) for b in bounds[:-1]],fontsize=24)
        # plt.tight_layout(rect=[0, 0.1, 1, 1])  # bottom margin = 10% of figure
        plt.savefig('%s%s'%(save_dir,file_str))
        plt.close()

def main():
    print('generating the output figures: LC_quick_out.py')
    load_dir = '/scratch/bmac87/LaunchCast_scratch/results/LC_HR3_16x16inputs/Conv3D__label_LC_HR3_16x16inputs_5000_epochs_rot_1_BC_focal_tanh_exp_4_no_early_stopping/'
    fname = 'labels_outputs.pkl'
    labels_outputs = pickle.load(open(load_dir+fname,'rb'))

    y_true = labels_outputs['y_true']
    y_pred = labels_outputs['y_pred']

    save_dir = load_dir+'output_images/'
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    generate_output_figures(args={}, y_true=y_true, y_pred=y_pred, save_dir=save_dir)

if __name__=='__main__':
    main()
    
    

