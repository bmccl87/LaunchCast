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
from LC_model_64216 import Stack

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
    fcst_times = ['00-15 min','15-30 min','30-45 min','45-60 min']
    t_keys = ['t0','t1','t2','t3']
    grey_color = ['#f7f7f7']

    thresh_bins = np.arange(0.0,1.05,0.05)
    thresh = np.arange(0.05,1.05,0.05)

    print(y_true.shape)
    print(y_pred.shape)

    fig, axes = plt.subplots(1,1,figsize=(10.3,9))

    axes = make_performance_diagram_axis(axes,csi_cmap='Greys')
    p_dict = pickle.load(open('persistence_prec_recall.pkl','rb'))
    h_dict = pickle.load(open('hrrr_stat_dict.pkl','rb'))
    handles=[]
    labels =[]
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

        label = fcst_times[t]
        # print('Max Threshold %s:'%t_key,thresh[np.argmax(csis)])
        label1 = 'Persistence'
        
        lch = axes.plot(np.asarray(srs),np.asarray(pods),
                color=colors[t],
                markerfacecolor=colors[t],
                marker=markers[t],
                markeredgecolor='black',
                markersize=8,
                label=label,
                linewidth=5,
                linestyle=linestyles[t])[0]
        handles.append(lch)
        labels.append(label)

        ph = axes.scatter(p_dict[t_key]['precision'][1],p_dict[t_key]['recall'][1],
                    c=colors[t],
                    marker=markers[t],
                    edgecolors='black',
                    linewidths=2.5,
                    s=120,   # marker area in points^2
                    label=label1
                    )
        if t==0:
            handles.append(ph)
            labels.append(label1)

        recall_f01 = h_dict[t_key]['f01_precision'][1]
        precision_f01 = h_dict[t_key]['f01_recall'][1]

        f1h = axes.scatter(recall_f01,precision_f01,
                    marker='p',
                    s=120,
                    color=colors[t],
                    edgecolor='black',
                    linewidth=2.5,
                    label='HRRR F01')
        if t==0:
            handles.append(f1h)
            labels.append('HRRR F01')

        recall_f02 = h_dict[t_key]['f02_precision'][1]
        precision_f02 = h_dict[t_key]['f02_recall'][1]

        f2h=axes.scatter(recall_f02,precision_f02,
            marker='h',
            s=120,
            color=colors[t],
            edgecolor='black',
            linewidth=2.5,
            label='HRRR F02')
        if t==0:
            handles.append(f2h)
            labels.append('HRRR F02')
        
        if t==0:
            for i,thr in enumerate(thresh):
                if i%2==0:
                    text = np.char.ljust(str(np.round(thr+.05,2)),width=4,fillchar='0')
                    axes.text(srs[i+1]+0.02,pods[i+1]+0.01,text,path_effects=pe1,fontsize=12,color=colors[t])
        if t==3:
            for i,thr in enumerate(thresh):
                if i==0:
                    text = np.char.ljust(str(np.round(thr,2)),width=4,fillchar='0')
                    axes.text(srs[i]-0.056,pods[i]-0.05,text,path_effects=pe1,fontsize=12,color=colors[t])
                
                if i==1:
                    text = np.char.ljust(str(np.round(thr+.05,2)),width=4,fillchar='0')
                    axes.text(srs[i+1]-0.056,pods[i+1]-0.04,text,path_effects=pe1,fontsize=12,color=colors[t])
                
                if i==3:
                    text = np.char.ljust(str(np.round(thr+.05,2)),width=4,fillchar='0')
                    axes.text(srs[i+1]-0.056,pods[i+1]-0.04,text,path_effects=pe1,fontsize=12,color=colors[t])

                if i==5:
                    text = np.char.ljust(str(np.round(thr+.05,2)),width=4,fillchar='0')
                    axes.text(srs[i+1]-0.056,pods[i+1]-0.04,text,path_effects=pe1,fontsize=12,color=colors[t])
                
                if i==7:
                    text = np.char.ljust(str(np.round(thr+.05,2)),width=4,fillchar='0')
                    axes.text(srs[i+1]-0.056,pods[i+1]-0.04,text,path_effects=pe1,fontsize=12,color=colors[t])

    axes.text(
            0.5,0.98,
            "Lead   CSI   AUC\n"
            "0-15  0.34  0.47\n"
            "15-30 0.30  0.41\n"
            "30-45 0.26  0.33\n"
            "45-60 0.22  0.26",
            transform=axes.transAxes,
            va='top',
            family='monospace',
            bbox=dict(
            facecolor='white',
            edgecolor='black',
            boxstyle='round',
            pad=0.4,
            alpha=1.0),zorder=12)

    axes.set_ylabel('Recall (POD)')
    axes.set_xlabel('Precision (SR)')

    axes.legend(handles=handles,labels=labels,loc='upper right',fontsize=12)
    plt.savefig('./temp_PD.png')
    plt.savefig('./temp_PD.pdf')
    plt.savefig('%sPD.png'%save_dir)
    plt.savefig('%sPD.pdf'%save_dir)
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