import numpy as np
import os
import xarray as xr
import shutil
import cartopy.crs as ccrs
import pickle
import pandas as pd

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

matplotlib.rcParams['axes.labelsize'] = 12 #fontsize in pts
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 12 
matplotlib.rcParams['ytick.labelsize'] = 12 
matplotlib.rcParams['legend.fontsize'] = 12 

def generate_LC_output_figures_with_cartopy(y_true,y_pred,save_dir):
    if os.path.isdir(save_dir)==False:
        os.makedirs(save_dir)
    ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/2_MERLIN_Data/2d_MERLIN_binary_fed_target/202206/MERLIN_hrrr_202206302230.nc',engine='netcdf4')
    lat = ds['input_lat'].values
    lon = ds['input_lon'].values
    del ds

    ds = xr.open_dataset('/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/6_final_xr/LC_2024.nc',engine='netcdf4')
    sample_times = ds['sample_time'].values
    merlin_cc = ds['cc_forecast'].values
    merlin_cg = ds['cg_forecast'].values

    # # Define the color segments and corresponding values
    # colors = ["gray", "slateblue", "blue", "darkgreen", "green", "lightgreen", "yellow", "peru", "brown"]#spc product
    bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Create a colormap and norm
    # cmap = mcolors.ListedColormap(colors)
    cmap = plt.get_cmap('viridis')
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for i in range(y_true.shape[0]):
        y_t = np.squeeze(y_true[i,:,:,:])
        y_t[y_t<1] = np.nan
        print(np.nanmax(y_t))

        m_cc = np.squeeze(merlin_cc[i,:,:,:])
        m_cc[m_cc>=1] = 1.0
        m_cc[m_cc<1] = np.nan

        m_cg = np.squeeze(merlin_cg[i,:,:,:])
        m_cg[m_cg>=1] = 1.0
        m_cg[m_cg<1] = np.nan

        y_p = np.squeeze(y_pred[i,:,:,:])
        y_p[y_p<.05] = np.nan

        vt = sample_times[i]
        ts = pd.Timestamp(vt)
        minute = f"{ts.minute:02}"
        hour = f"{ts.hour:02}"
        day = f"{ts.day:02}"
        month = f"{ts.month:02}"
        year = f"{ts.year:04}"  
        fsave = '%s%s%s%s%s.png'%(year,month,day,hour,minute)
        
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        month_int=int(month)-1
        title_str = '%s%s UTC %s %s %s'%(hour,minute,day,months[month_int],year)

        fig,axes = plt.subplots(nrows=4,ncols=4,figsize=(11,8.5),subplot_kw={'projection': ccrs.PlateCarree()})

        for t in range(4):
            im0 = axes[t,0].pcolormesh(lon,lat,y_p[t,:,:]*100,cmap=cmap,norm=norm)
            axes[t,0].set_xticks([],[])
            axes[t,0].set_yticks([],[])
            axes[t,0].coastlines(color='black')
            plt.colorbar(im0,ax=axes[t,0])

            axes[t,1].pcolormesh(lon,lat,y_t[t,:,:])
            axes[t,1].set_xticks([],[])
            axes[t,1].set_yticks([],[])
            axes[t,1].coastlines(color='black')

            axes[t,2].pcolormesh(lon,lat,m_cc[t,:,:])
            axes[t,2].set_xticks([],[])
            axes[t,2].set_yticks([],[])
            axes[t,2].coastlines(color='black')

            axes[t,3].pcolormesh(lon,lat,m_cg[t,:,:])
            axes[t,3].set_xticks([],[])
            axes[t,3].set_yticks([],[])
            axes[t,3].coastlines(color='black')

        axes[0,0].set_ylabel('00-15 mins',fontsize=12)
        axes[0,0].set_title('LC Forecast',fontsize=12)
        axes[0,1].set_title('Target',fontsize=12)
        axes[0,2].set_title('Obs. IC',fontsize=12)
        axes[0,3].set_title('Obs. CG',fontsize=12)

        axes[1,0].set_ylabel('15-30 mins',fontsize=12)
        axes[2,0].set_ylabel('30-45 mins',fontsize=12)
        axes[3,0].set_ylabel('45-60 mins',fontsize=12)
        plt.suptitle(title_str)
        plt.tight_layout()
        plt.savefig(save_dir+fsave)
        plt.close()

if __name__=='__main__':
    
    parser = create_parser()
    args = parser.parse_args()
    load_dir = '%s%s/%s/'%(args.results_path,args.project,args.exp_name)
    output_fname = 'labels_prob_outputs.pkl'
    y_dict = pickle.load(open(load_dir+output_fname,'rb'))
    save_dir=load_dir+'all_output_images/'
    generate_LC_output_figures_with_cartopy(y_true=y_dict['y_true'], y_pred=y_dict['y_pred'],save_dir=save_dir)