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

    exp_nums = [7,11,13,14]
    x_label_dict = {}
    x_label_dict.update({0:'MRMS Only'})
    x_label_dict.update({1:'GLM Only'})
    x_label_dict.update({2:'EFM Only'})
    x_label_dict.update({3:'HRRR Only'})
    #--------------------------------------------
    x_tick_labels = []
    for i in range(4):
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
        print(df.loc[df['exp'].isin(exp_nums)][stat_key])
        auc_dict[t_key] = (ref_auc-df.loc[df['exp'].isin(exp_nums)][stat_key].values)/ref_auc*100
        print(auc_dict)

    exp_idxs=np.arange(0,4)
    ax.grid(axis='y', linestyle='--', alpha=0.6,zorder=0)
    ax.bar(exp_idxs - 1.5*width, auc_dict['t0'], width, color=colors[0], label='00-15 min',zorder=2)
    ax.bar(exp_idxs - 0.5*width, auc_dict['t1'], width, color=colors[1], label='15-30 min',zorder=2)
    ax.bar(exp_idxs + 0.5*width, auc_dict['t2'], width, color=colors[2], label='30-45 min',zorder=2)
    ax.bar(exp_idxs + 1.5*width, auc_dict['t3'], width, color=colors[3], label='45-60 min',zorder=2)
    # ax.vlines(x=7.5,ymin=0,ymax=100,linewidth=3,color='black')
    # ax.hlines(y=0,xmin=.25,xmax=15.5,linewidth=3,color='black')
    
    ax.set_ylabel('Reduction in AUC-PR (%)')
    # # # ax.set_ylim([0,.6])
    # # # ax.set_yticks([0,.1,.2,.3,.4,.5,.6],['0.0','0.1','0.2','0.3','0.4','0.5','0.6'])
    ax.set_xlabel('Input Data Sets')
    ax.set_xticks(exp_idxs, x_tick_labels)
    # ax.set_xlim([0.5,15.5])
    # ax.set_ylim([-5,100])
    ax.legend()
    ax.set_facecolor('#f0f0f0')
    plt.tight_layout()
    plt.savefig('./figure_files/Figure_X_ablation_bar.png')
    plt.savefig('./figure_files/Figure_X_ablation_bar.pdf')
    plt.savefig('./temp.png')
    plt.close()

def main():
    print('plotting the ablation bar graph')
    plot_bar()

if __name__=='__main__':
    main()