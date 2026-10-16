import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pickle
import xarray as xr
from LC_parser import create_parser
import matplotlib
import matplotlib.patheffects as path_effects

def LaunchCast_attributes_plot(load_dir='/scratch/bmac87/',wandb={}):
    
    perfect = np.linspace(0,1,100)
    pe1 = [path_effects.withStroke(linewidth=1.5,foreground="k")]

    fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(11,11))
    climo =.09
    no_skill = (climo+perfect)/2
    axes.plot(perfect,perfect,linestyle='--',color='grey',linewidth=3.0,label='Perfect Reliability')
    axes.set_xlim([0,1])
    axes.set_xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],['0','10','20','30','40','50','60','70','80','90','100'],fontsize=18)
    axes.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'],fontsize=18)
    axes.set_ylim([0,1])
    axes.set_ylabel('Observed Relative Frequency',fontsize=18)
    axes.set_xlabel('Forecast Probability (%)',fontsize=18)
    axes.text(.6,.52,'Over Forecast',fontsize=18,fontweight='heavy')
    axes.text(.3,.72,'Under Forecast',fontsize=18,fontweight='heavy')
    axes.text(.65,.22,'No Skill',fontsize=18,fontweight='heavy')
    axes.grid(True)
    axes.plot(climo*np.ones(100),np.linspace(0,1,100),linestyle='dashdot',linewidth=3.0,color='grey',label='Climatology',zorder=0)
    axes.plot(np.linspace(0,1,100),climo*np.ones(100),linestyle='dashdot',linewidth=3.0,color='grey',zorder=0)
    axes.plot(perfect,no_skill,linestyle='solid',linewidth=5.0,color='grey',label='No Skill',zorder=0)
    axes.legend(fontsize=18,loc='upper left', bbox_to_anchor=(0.11, 0.999))
    plt.savefig('./reliability_attr_diss.png',dpi=300)
    plt.savefig('./reliability_attr_diss.pdf',dpi=300)
    plt.close()

def main():
    LaunchCast_attributes_plot()

if __name__=='__main__':
    main()
    print('END OF LC_eval_attributes.py')