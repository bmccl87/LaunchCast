import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pickle
import xarray as xr
from LC_parser import create_parser
import matplotlib
import matplotlib.patheffects as path_effects

def LaunchCast_pred_hist_calc_np(save_dir = '/scratch/bmac87/', y_true=[],y_pred=[]):
    print('generating the LC attributes diagram histogram with numpy')
    thresh = np.arange(0.00,1.05,0.05)
    thresh_centers =  0.5 * (thresh[:-1] + thresh[1:])
    t_keys = ['t0','t1','t2','t3']
    hist_dict = {}

    for t,t_key in enumerate(t_keys):
        y_t_true = np.ravel(y_true[:,t,:,:])
        y_t_pred = np.ravel(y_pred[:,t,:,:])

        #bin the predictions into the thresholds. inclusive to the right
        pred_indices = np.digitize(y_t_pred,bins=thresh,right=True)

        # Initialize lists
        thresh_counts = np.zeros(len(thresh))

        # Compute stats for each bin
        for i in range(1, len(thresh)):
            idx = pred_indices == i
            if np.sum(idx)>0:
                thresh_counts[i] = np.sum(idx)
        hist_dict.update({'thresh_counts_%s'%t_key:thresh_counts})

    pickle.dump(hist_dict,open('%sforecast_hist_dict_warning.pkl'%save_dir,'wb'))

def LaunchCast_attributes_calc_np(save_dir = '/scratch/bmac87/',y_true=[],y_pred=[]):
    
    print('calculating the LC attributes with numpy for better control')
    thresh = np.arange(0.00,1.05,0.05)
    thresh_centers =  0.5 * (thresh[:-1] + thresh[1:])
    t_keys = ['t0','t1','t2','t3']

    climatology = np.mean(y_true)
    attr_dict = {}
    for t,t_key in enumerate(t_keys):
        y_t_true = np.ravel(y_true[:,t,:,:])
        y_t_pred = np.ravel(y_pred[:,t,:,:])

        #bin the predictions into the thresholds. inclusive to the right
        indices = np.digitize(y_t_pred,bins=thresh,right=True)

        # Initialize lists
        thresh_true = np.zeros(len(thresh))
        thresh_pred = np.zeros(len(thresh))
        thresh_counts = np.zeros(len(thresh))

        # Compute stats for each bin
        for i in range(1, len(thresh)):
            idx = indices == i
            if np.sum(idx)>0:
                thresh_true[i] = np.mean(y_t_true[idx])
                thresh_pred[i] = np.mean(y_t_pred[idx])
                thresh_counts[i] = np.sum(idx)

        brier_score = np.mean((y_t_pred-y_t_true)**2)
        brier_score_climatology = np.mean((y_t_true-climatology)**2)
        brier_skill_score = 1 - (brier_score / brier_score_climatology)
        
        stat_dict = {'thresh_true_%s'%t_key:thresh_true,
                    'thresh_pred_%s'%t_key:thresh_pred,
                    'thresh_counts_%s'%t_key:thresh_counts,
                    'bs_%s'%t_key:brier_score,
                    'bss_%s'%t_key:brier_skill_score,
                    'climo':climatology}

        attr_dict.update({'%s'%t_key:stat_dict})
        del stat_dict
        del y_t_true, y_t_pred

    pickle.dump(attr_dict,open('%sattr_dict_forecast_warning.pkl'%save_dir,'wb'))

def LaunchCast_attributes_plot(load_dir='/scratch/bmac87/',wandb={}):
    colors = ['#ca0020','#f4a582','#92c5de','#0571b0']  # Colorblind-friendly, same as BC
    linestyles = ['dashed','dashdot','dotted','solid'] #np.flip(['-', '--', '-.', ':'])
    markers = ['s', 'o', '^', 'D']
    fcst_times = ['00-15 min','15-30 min','30-45 min','45-60 min']
    t_keys = ['t0','t1','t2','t3']
    grey_color = ['#f7f7f7']

    attr_data = pickle.load(open('%sattr_dict_forecast_warning.pkl'%load_dir,'rb'))
    perfect = np.linspace(0,1,100)
    pe1 = [path_effects.withStroke(linewidth=1.5,foreground="k")]

    fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(17,22))
    for t, t_key in enumerate(t_keys):
        climos = []
        data = attr_data['%s'%t_key]

        x = data['thresh_pred_%s'%t_key]#x_axis
        y = data['thresh_true_%s'%t_key]#y_axis
        bs = data['bs_%s'%t_key]
        plot_idxs=[]
        for idx in range(len(x)):
            if x[idx]>0 and y[idx]>0:
                plot_idxs.append(idx)
        axes[0].plot(x[plot_idxs],y[plot_idxs],linewidth=5.0,marker=markers[0],linestyle=linestyles[t],markeredgecolor='black',markersize=10.0,color=colors[t],label='BS: %s = %s'%(fcst_times[t],f"{bs:.03f}"),zorder=3)
    
    climo =data['climo']
    no_skill = (climo+perfect)/2
    axes[0].plot(perfect,perfect,linestyle='--',color='grey',linewidth=3.0,label='Perfect')
    axes[0].set_xlim([0,1])
    axes[0].set_xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],['0','10','20','30','40','50','60','70','80','90','100'],fontsize=18)
    axes[0].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'],fontsize=18)
    axes[0].set_ylim([0,1])
    axes[0].set_ylabel('Observed Relative Frequency',fontsize=18)
    axes[0].grid(True)
    axes[0].set_title('Total Lightning',fontsize=24)
    axes[0].plot(np.mean(climo)*np.ones(100),np.linspace(0,1,100),linestyle='dashdot',linewidth=3.0,color='grey',label='Climatology',zorder=0)
    axes[0].plot(np.linspace(0,1,100),np.mean(climo)*np.ones(100),linestyle='dashdot',linewidth=3.0,color='grey',zorder=0)
    axes[0].plot(perfect,no_skill,linestyle='solid',linewidth=5.0,color='grey',label='No Skill',zorder=0)
    axes[0].legend(fontsize=18,loc='upper left', bbox_to_anchor=(0.11, 0.999))
    axes[0].text(.015,.93,'(a)',fontsize=24,weight='heavy',path_effects=pe1)
    
    thresh = np.arange(0.00,1.05,0.05)
    thresh_centers =  0.5 * (thresh[:-1] + thresh[1:])
    counts = pickle.load(open('%sforecast_hist_dict_warning.pkl'%(load_dir),'rb'))
    for t,t_key in enumerate(t_keys):
        thresh_counts = counts['thresh_counts_%s'%t_key]
        axes[1].bar(thresh-.025,thresh_counts,width=0.05, color=colors[t],alpha=.4, edgecolor='black',zorder=3,label='Prob Counts - %s'%(fcst_times[t]))
    
    axes[1].grid(True,zorder=0)    
    axes[1].set_yscale('log')
    axes[1].set_xlim([0,1])
    axes[1].set_ylabel('Count',fontsize=18)
    axes[1].set_xticks(thresh[::2],['0','10','20','30','40','50','60','70','80','90','100'],fontsize=18)
    axes[1].set_yticks([10,10**(2),10**(3),10**(4),10.0**(5),10.0**(6),10.0**(7)],['10','$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$','$10^{6}$','$10^{7}$'],fontsize=18)
    axes[1].set_ylim([10,10**7.2])
    axes[1].set_xlabel('Forecast Probability (%)',fontsize=18)
    axes[1].text(.915,10**6.7,'(b)',fontsize=24,weight='heavy',path_effects=pe1)
    axes[1].legend(fontsize=18,loc='upper center')

    plt.show()
    plt.savefig('temp_attr_warning.png')
    plt.savefig('%sReliability_Diagram_forecast_lead_times_warning.png'%(load_dir))
    plt.savefig('%sReliability_Diagram_forecast_lead_times_warning.pdf'%(load_dir))
    plt.close()

def main():

    #command to run
    #python 1_LC_reliability.py @txt_exp_eval.txt 
    print('LC_eval_attributes.py main function')
    
    print('parsing args')
    parser = create_parser()
    args = parser.parse_args()
    print('args parsed')
    
    exp_name = args.exp_name
    exp_dir = '%s/%s/%s/'%(args.results_path,args.project,exp_name)
    fname = 'labels_prob_outputs.pkl'
    data_dict = pickle.load(open(exp_dir+fname,'rb'))
    
    y_true = data_dict['y_true']
    y_pred = data_dict['y_pred']

    LaunchCast_pred_hist_calc_np(save_dir=exp_dir, y_true=y_true,y_pred=y_pred)
    LaunchCast_attributes_calc_np(save_dir=exp_dir,y_true=y_true,y_pred=y_pred)
    LaunchCast_attributes_plot(load_dir=exp_dir)

if __name__=='__main__':
    main()
    print('END OF LC_eval_attributes.py')