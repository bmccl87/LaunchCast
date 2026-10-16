import numpy as np
import os
import xarray as xr
import shutil
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pickle
from gewitter_functions import get_contingency_table,make_performance_diagram_axis,get_acc,get_pod,get_sr,csi_from_sr_and_pod
import matplotlib
import matplotlib.patheffects as path_effects
import tensorflow as tf
from sklearn.metrics import auc, precision_recall_curve
from LC_parser import *

#outlines for text 
pe1 = [path_effects.withStroke(linewidth=1.5,
                            foreground="k")]
pe2 = [path_effects.withStroke(linewidth=1.5,
                            foreground="w")]

matplotlib.rcParams['axes.facecolor'] = [0.9,0.9,0.9] #makes a grey background to the axis face
matplotlib.rcParams['axes.labelsize'] = 18 #fontsize in pts
matplotlib.rcParams['axes.titlesize'] = 18 
matplotlib.rcParams['xtick.labelsize'] = 18 
matplotlib.rcParams['ytick.labelsize'] = 18 
matplotlib.rcParams['legend.fontsize'] = 18 
matplotlib.rcParams['legend.facecolor'] = '#f7f7f7'#light grey
matplotlib.rcParams['savefig.transparent'] = False

def forecast_PD(y_true=[],y_pred=[],save_dir='/scratch/bmac87/'):

    fig, axes = plt.subplots(1,1,figsize=(11,9.5))
    axes = make_performance_diagram_axis(axes,csi_cmap='Greys')
    axes.text(0.85, 0.9,' High Skill ',fontsize=18,fontweight='normal',
            ha='center',va='center',bbox=dict(facecolor='white', 
            edgecolor='black', boxstyle='round',pad=.25),transform=axes.transAxes)

    axes.text(0.2, 0.2,' Low Skill ',fontsize=18,fontweight='normal',
            ha='center',va='center',bbox=dict(facecolor='white', 
            edgecolor='black', boxstyle='round',pad=.25),transform=axes.transAxes)

    axes.set_ylabel('Recall (POD)')
    axes.set_xlabel('Precision (SR)')

    plt.savefig('./dissertation_blank_PD.png')
    plt.savefig('./dissertation_blank_PD.pdf')
    plt.close()

def main():
    forecast_PD()

if __name__=='__main__':
    main()