import numpy as np
import os
import shutil
import pandas as pd
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects

#outlines for text 
pe1 = [path_effects.withStroke(linewidth=1.5,
                            foreground="k")]

pe2 = [path_effects.withStroke(linewidth=1.5,
                            foreground="w")]

matplotlib.rcParams['axes.labelsize'] = 18 #fontsize in pts
matplotlib.rcParams['axes.titlesize'] = 24
matplotlib.rcParams['xtick.labelsize'] = 12 
matplotlib.rcParams['ytick.labelsize'] = 12 
matplotlib.rcParams['legend.fontsize'] = 12 

def plot_bar():

    df = pickle.load(open('../3_model_analysis/ablation_table_df.pkl','rb'))
    df_columns = df.columns
    for col in df_columns:
        print(col)

    exp_nums = np.arange(1,16)
    x_label_dict = {}
    x_label_dict.update({1:'MRMS \n GLM \n EFM'})
    x_label_dict.update({2:'MRMS \n GLM \n HRRR'})
    x_label_dict.update({3:'MRMS \n GLM'})
    x_label_dict.update({4:'MRMS \n HRRR \n EFM '})
    x_label_dict.update({5:'MRMS \n EFM'})
    x_label_dict.update({6:'MRMS \n HRRR'})
    x_label_dict.update({7:'MRMS'})
    #--------------------------------------------
    x_label_dict.update({8:'EFM \n GLM \n HRRR'})
    x_label_dict.update({9:'EFM \n GLM'})
    x_label_dict.update({10:'GLM \n HRRR'})
    x_label_dict.update({11:'GLM'})
    x_label_dict.update({12:'EFM \n HRRR'})
    x_label_dict.update({13:'EFM'})
    x_label_dict.update({14:'HRRR'})
    x_label_dict.update({15:'None'})
    x_tick_labels = []

    exp_labels = []
    for i in range(1,16):
        exp_labels.append('Exp. %s'%(i))

    for i in range(1,16):
        x_tick_labels.append(x_label_dict[i])

    colors = ['#ca0020','#f4a582','#92c5de','#0571b0']
    width=.2
    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(20,8))#figsize=(width,height)
    auc_dict = {'t0':[],'t1':[],'t2':[],'t3':[]}

    for t,t_key in enumerate(['t0','t1','t2','t3']):
        stat_key = 'auc_pr_%s'%(t_key)
        ref_auc = df.loc[df['exp']==0][stat_key].values
        print(stat_key)
        print(ref_auc)
        auc_dict[t_key] = (ref_auc-df[stat_key].values[1:])/ref_auc*100
        print(auc_dict)
    
    for i in range(1,16):
        ax.text(i-.25,-4,exp_labels[i-1],fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.6,zorder=0)
    ax.bar(exp_nums - 1.5*width, auc_dict['t0'], width, color=colors[0], label='00-15 min',zorder=2)
    ax.bar(exp_nums - 0.5*width, auc_dict['t1'], width, color=colors[1], label='15-30 min',zorder=2)
    ax.bar(exp_nums + 0.5*width, auc_dict['t2'], width, color=colors[2], label='30-45 min',zorder=2)
    ax.bar(exp_nums + 1.5*width, auc_dict['t3'], width, color=colors[3], label='45-60 min',zorder=2)
    ax.vlines(x=7.5,ymin=0,ymax=100,linewidth=3,color='black')
    ax.hlines(y=0,xmin=.25,xmax=15.5,linewidth=3,color='black')
    
    ax.set_ylabel('Reduction in AUC-PR (%)')
    # # ax.set_ylim([0,.6])
    # # ax.set_yticks([0,.1,.2,.3,.4,.5,.6],['0.0','0.1','0.2','0.3','0.4','0.5','0.6'])
    ax.set_xlabel('Input Data Sets')
    ax.set_xticks(exp_nums, x_tick_labels)
    ax.set_xlim([0.5,15.5])
    ax.set_ylim([-5,100])
    ax.legend()
    ax.set_facecolor('#f0f0f0')
    plt.tight_layout()
    plt.savefig('./figure_files/Figure_X_ablation_bar_16.png')
    plt.savefig('./figure_files/Figure_X_ablation_bar_16.pdf')
    plt.savefig('./temp.png')
    plt.close()

def main():
    print('plotting the ablation bar graph')
    plot_bar()

if __name__=='__main__':
    main()