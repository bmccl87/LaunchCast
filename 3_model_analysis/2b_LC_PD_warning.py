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
from LC_models import Stack

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

def forecast_PD(y_true=[],y_pred=[],save_dir='/scratch/bmac87/'):

    colors = ['#ca0020','#f4a582','#92c5de','#0571b0']  # Colorblind-friendly, same as BC
    linestyles = ['dashed','dashdot','dotted','solid'] #np.flip(['-', '--', '-.', ':'])
    markers = ['s', 'o', '^', 'D']
    fcst_times = ['00-15m','15-30m','30-45m','45-60m']
    t_keys = ['t0','t1','t2','t3']
    grey_color = ['#f7f7f7']
    thresh_bins = np.arange(0.0,1.05,0.05)
    thresh = np.arange(0.05,1.05,0.05)

    print(y_true.shape)
    print(y_pred.shape)

    fig, axes = plt.subplots(1,1,figsize=(20,20))

    axes = make_performance_diagram_axis(axes,csi_cmap='Greys')

    for t,t_key in enumerate(t_keys):
        print(t,t_key)
        y_true_1d = np.ravel(y_true[:,t,:,:])
        y_pred_1d = np.ravel(y_pred[:,t,:,:])

        #statistics we need for performance diagram 
        tp = tf.keras.metrics.TruePositives(thresholds=thresh.tolist())
        fp = tf.keras.metrics.FalsePositives(thresholds=thresh.tolist())
        fn = tf.keras.metrics.FalseNegatives(thresholds=thresh.tolist())

        # get performance diagram line by getting tp,fp and fn 
        tp.reset_state()
        fp.reset_state()
        fn.reset_state()

        tps = tp(y_true_1d,y_pred_1d)
        fps = fp(y_true_1d,y_pred_1d)
        fns = fn(y_true_1d,y_pred_1d)
        
        pods = tps/(tps + fns)
        srs = tps/(tps + fps)
        csis = tps/(tps + fns + fps)

        precision, recall, thresholds = precision_recall_curve(y_true_1d, y_pred_1d)
        auc_value = auc(recall, precision)

        label = '%s Max CSI: %s, AUC: %s'%(fcst_times[t],f"{max(csis):.2f}",f"{auc_value:.2f}")
        print('Max Threshold %s:'%t_key,thresh[np.argmax(csis)])
        
        axes.plot(np.asarray(srs),np.asarray(pods),
                color=colors[t],
                markerfacecolor=colors[t],
                marker=markers[t],
                markeredgecolor='black',
                label=label,
                linewidth=3,
                linestyle=linestyles[t])
    
    axes.legend(loc='upper right',fontsize=18)
    axes.set_title('Total Lightning')

    plt.savefig('./temp_PD_forecast_warning.png')
    plt.savefig('%sPD_forecast_warning.png'%save_dir)
    plt.savefig('%sPD_forecast_warning.pdf'%save_dir)
    plt.close()

def main():

    print('Breaking out the AUC curves by forecast increment')
    print('parsing args')
    parser = create_parser()
    args = parser.parse_args()
    print('args parsed')

    exp_name = args.exp_name
    exp_dir = '%s%s/%s/'%(args.results_path,args.project,exp_name)
    fname = 'labels_prob_outputs.pkl'
    data_dict = pickle.load(open(exp_dir+fname,'rb'))
    
    y_true = data_dict['y_true']
    y_pred = data_dict['y_pred']

    forecast_PD(y_true=y_true,y_pred=y_pred,save_dir=exp_dir)

if __name__=='__main__':
    main()