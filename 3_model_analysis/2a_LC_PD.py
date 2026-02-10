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

    fig, axes = plt.subplots(1,2,figsize=(20,10))

    axes[0] = make_performance_diagram_axis(axes[0],csi_cmap='Greys')
    axes[1] = make_performance_diagram_axis(axes[1],csi_cmap='Greys')

    for t,t_key in enumerate(t_keys):
        print(t,t_key)
        cc_labels_1d = np.ravel(y_true[:,t,:,:,0])
        cg_labels_1d = np.ravel(y_true[:,t,:,:,1])

        cc_pred_1d = np.ravel(y_pred[:,t,:,:,0])
        cg_pred_1d = np.ravel(y_pred[:,t,:,:,1])

        #statistics we need for performance diagram 
        tp = tf.keras.metrics.TruePositives(thresholds=thresh.tolist())
        fp = tf.keras.metrics.FalsePositives(thresholds=thresh.tolist())
        fn = tf.keras.metrics.FalseNegatives(thresholds=thresh.tolist())

        # get performance diagram line by getting tp,fp and fn 
        tp.reset_state()
        fp.reset_state()
        fn.reset_state()

        cc_tps = tp(cc_labels_1d,cc_pred_1d)
        cc_fps = fp(cc_labels_1d,cc_pred_1d)
        cc_fns = fn(cc_labels_1d,cc_pred_1d)

        cg_tps = tp(cg_labels_1d,cg_pred_1d)
        cg_fps = fp(cg_labels_1d,cg_pred_1d)
        cg_fns = fn(cg_labels_1d,cg_pred_1d)
        
        cc_pods = cc_tps/(cc_tps + cc_fns)
        cc_srs = cc_tps/(cc_tps + cc_fps)
        cc_csis = cc_tps/(cc_tps + cc_fns + cc_fps)

        cg_pods = cg_tps/(cg_tps + cg_fns)
        cg_srs = cg_tps/(cg_tps + cg_fps)
        cg_csis = cg_tps/(cg_tps + cg_fns + cg_fps)

        cc_precision, cc_recall, cc_thresholds = precision_recall_curve(cc_labels_1d, cc_pred_1d)
        cc_auc = auc(cc_recall, cc_precision)

        cg_precision, cg_recall, cg_threshold = precision_recall_curve(cg_labels_1d, cg_pred_1d)
        cg_auc = auc(cg_recall, cg_precision)

        cc_label = '%s Max CSI: %s, AUC: %s'%(fcst_times[t],f"{max(cc_csis):.2f}",f"{cc_auc:.2f}")
        cg_label = '%s Max CSI: %s, AUC: %s'%(fcst_times[t],f"{max(cg_csis):.2f}",f"{cg_auc:.2f}")
        
        axes[0].plot(np.asarray(cc_srs),np.asarray(cc_pods),
                color=colors[t],
                markerfacecolor=colors[t],
                marker=markers[t],
                markeredgecolor='black',
                label=cc_label,
                linewidth=3,
                linestyle=linestyles[t])
        
        axes[1].plot(np.asarray(cg_srs),np.asarray(cg_pods),
                color=colors[t],
                markerfacecolor=colors[t],
                marker=markers[t],
                markeredgecolor='black',
                label=cg_label,
                linewidth=3,
                linestyle=linestyles[t])
    
    axes[0].legend(loc='upper right',fontsize=18)
    axes[0].set_title('IC')
    axes[1].legend(loc='upper right',fontsize=18)
    axes[1].set_title('CG')

    plt.savefig('./temp_PD_forecast.png')
    plt.savefig('%sPD_forecast.png'%save_dir)
    plt.savefig('%sPD_forecast.pdf'%save_dir)
    plt.close()

def main():

    print('Breaking out the AUC curves by forecast increment')
    print('parsing args')
    parser = create_parser()
    args = parser.parse_args()
    print('args parsed')

    exp_name = 'LC_Forecast_HRRR_MRMS_noZdr_GLM_no_EFM_2_MERLIN_rot_1_early_stop_True_conv_act_relu_L2_0.0__BN_False_SD_0.000_pool_max__2dkernel_8__last_act_sigmoid_epochs_250__no_noise_lrate_0.000000100__label_from_logits_focal_2_econders_bias_init'
    exp_dir = '/scratch/bmac87/LaunchCast_scratch/results/%s/%s/'%(args.project,exp_name)
    fname = 'labels_prob_outputs.pkl'
    data_dict = pickle.load(open(exp_dir+fname,'rb'))
    
    y_true = data_dict['y_true']
    y_pred = data_dict['y_pred']

    forecast_PD(y_true=y_true,y_pred=y_pred,save_dir=exp_dir)

if __name__=='__main__':
    main()