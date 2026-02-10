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
        y_cc_true = np.ravel(y_true[:,t,:,:,0])
        y_cg_true = np.ravel(y_true[:,t,:,:,1])

        y_cc_pred = np.ravel(y_pred[:,t,:,:,0])
        y_cg_pred = np.ravel(y_pred[:,t,:,:,1])

        #bin the predictions into the thresholds. inclusive to the right
        cc_indices = np.digitize(y_cc_pred,bins=thresh,right=True)
        cg_indices = np.digitize(y_cg_pred,bins=thresh,right=True)

        # Initialize lists
        thresh_cc_counts = np.zeros(len(thresh))
        thresh_cg_counts = np.zeros(len(thresh))

        # Compute stats for each bin
        for i in range(1, len(thresh)):
            cc_idx = cc_indices == i
            cg_idx = cg_indices == i
            if np.sum(cc_idx)>0:
                thresh_cc_counts[i] = np.sum(cc_idx)
            if np.sum(cg_idx)>0:
                thresh_cg_counts[i] = np.sum(cg_idx)
        hist_dict.update({'thresh_cc_counts_%s'%t_key:thresh_cc_counts,'thresh_cg_counts_%s'%t_key:thresh_cg_counts})
    pickle.dump(hist_dict,open('%sforecast_hist_dict.pkl'%save_dir,'wb'))

def LaunchCast_attributes_calc_np(save_dir = '/scratch/bmac87/',y_true=[],y_pred=[]):
    
    print('calculating the LC attributes with numpy for better control')
    thresh = np.arange(0.00,1.05,0.05)
    thresh_centers =  0.5 * (thresh[:-1] + thresh[1:])
    t_keys = ['t0','t1','t2','t3']

    cc_climatology = np.mean(np.mean(np.mean(np.mean(y_true[:,:,:,:,0]))))
    cg_climatology = np.mean(np.mean(np.mean(np.mean(y_true[:,:,:,:,1]))))
    attr_dict = {}
    for t,t_key in enumerate(t_keys):
        y_cc_true = np.ravel(y_true[:,t,:,:,0])
        y_cg_true = np.ravel(y_true[:,t,:,:,1])
        

        y_cc_pred = np.ravel(y_pred[:,t,:,:,0])
        y_cg_pred = np.ravel(y_pred[:,t,:,:,1])

        #bin the predictions into the thresholds. inclusive to the right
        cc_indices = np.digitize(y_cc_pred,bins=thresh,right=True)
        cg_indices = np.digitize(y_cg_pred,bins=thresh,right=True)

        # Initialize lists
        thresh_cc_true = np.zeros(len(thresh))
        thresh_cc_pred = np.zeros(len(thresh))
        thresh_cc_counts = np.zeros(len(thresh))

        thresh_cg_true = np.zeros(len(thresh))
        thresh_cg_pred = np.zeros(len(thresh))
        thresh_cg_counts = np.zeros(len(thresh))

        # Compute stats for each bin
        for i in range(1, len(thresh)):
            cc_idx = cc_indices == i
            cg_idx = cg_indices == i
            if np.sum(cc_idx)>0:
                thresh_cc_true[i] = np.mean(y_cc_true[cc_idx])
                thresh_cc_pred[i] = np.mean(y_cc_pred[cc_idx])
                thresh_cc_counts[i] = np.sum(cc_idx)

            if np.sum(cg_idx)>0:
                thresh_cg_true[i] = np.mean(y_cg_true[cg_idx])
                thresh_cg_pred[i] = np.mean(y_cg_pred[cg_idx])
                thresh_cg_counts[i] = np.sum(cg_idx)
        
        cc_brier_score = np.mean((y_cc_pred-y_cc_true)**2)
        cg_brier_score = np.mean((y_cg_pred-y_cg_true)**2)
        
        brier_score_climatology_cc = np.mean((y_cc_true-cc_climatology)**2)
        brier_score_climatology_cg = np.mean((y_cg_true-cg_climatology)**2)

        bss_cc = 1 - (cc_brier_score / brier_score_climatology_cc)
        bss_cg = 1 - (cg_brier_score / brier_score_climatology_cg)
        
        cc_dict = {'thresh_cc_true_%s'%t_key:thresh_cc_true,
                    'thresh_cc_pred_%s'%t_key:thresh_cc_pred,
                    'thresh_cc_counts_%s'%t_key:thresh_cc_counts,
                    'cc_bs_%s'%t_key:cc_brier_score,
                    'cc_bss_%s'%t_key:bss_cc,
                    'climo':cc_climatology}
        
        cg_dict = {'thresh_cg_true_%s'%t_key:thresh_cg_true,
                    'thresh_cg_pred_%s'%t_key:thresh_cg_pred,
                    'thresh_cg_counts_%s'%t_key:thresh_cg_counts,
                    'cg_bs_%s'%t_key:cg_brier_score,
                    'cg_bss_%s'%t_key:bss_cc,
                    'climo':cg_climatology}
        attr_dict.update({'cc_%s'%t_key:cc_dict,'cg_%s'%t_key:cg_dict})
        del cc_dict, cg_dict
        del y_cc_true, y_cg_true

    pickle.dump(attr_dict,open('%sattr_dict_forecast.pkl'%save_dir,'wb'))

def LaunchCast_attributes_plot(load_dir='/scratch/bmac87/',wandb={}):
    colors = ['#ca0020','#f4a582','#92c5de','#0571b0']  # Colorblind-friendly, same as BC
    linestyles = ['dashed','dashdot','dotted','solid'] #np.flip(['-', '--', '-.', ':'])
    markers = ['s', 'o', '^', 'D']
    fcst_times = ['00-15 min','15-30 min','30-45 min','45-60 min']
    t_keys = ['t0','t1','t2','t3']
    grey_color = ['#f7f7f7']

    attr_data = pickle.load(open('%sattr_dict_forecast.pkl'%load_dir,'rb'))
    perfect = np.linspace(0,1,100)
    pe1 = [path_effects.withStroke(linewidth=1.5,foreground="k")]

    fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(40,40))
    for t, t_key in enumerate(t_keys):
        climos = []
        cc_data = attr_data['cc_%s'%t_key]
        cg_data = attr_data['cg_%s'%t_key]

        x_cc = cc_data['thresh_cc_pred_%s'%t_key]#x_axis
        y_cc = cc_data['thresh_cc_true_%s'%t_key]#y_axis
        bs_cc = cc_data['cc_bs_%s'%t_key]
        plot_idxs=[]
        for idx in range(len(x_cc)):
            if x_cc[idx]>0 and y_cc[idx]>0:
                plot_idxs.append(idx)
        axes[0,0].plot(x_cc[plot_idxs],y_cc[plot_idxs],linewidth=5.0,marker=markers[0],linestyle=linestyles[t],markeredgecolor='black',markersize=10.0,color=colors[t],label='CC - BS: %s = %s'%(fcst_times[t],f"{bs_cc:.03f}"),zorder=3)
        
        x_cg = cg_data['thresh_cg_pred_%s'%t_key]#x_axis
        y_cg = cg_data['thresh_cg_true_%s'%t_key]#y_axis
        bs_cg = cg_data['cg_bs_%s'%t_key]
        plot_idxs = []
        for idx in range(len(x_cg)):
            if x_cg[idx]>0 and y_cg[idx]>0:
                plot_idxs.append(idx)
        axes[0,1].plot(x_cg[plot_idxs],y_cg[plot_idxs],linewidth=5.0,marker=markers[1],linestyle=linestyles[t],markeredgecolor='black',markersize=10.0,color=colors[t],label='CG - BS: %s = %s'%(fcst_times[t],f"{bs_cg:.03f}"),zorder=3)    
    
    cc_climo =cc_data['climo']
    no_skill_cc = (cc_climo+perfect)/2
    axes[0,0].plot(perfect,perfect,linestyle='--',color='grey',linewidth=3.0,label='Perfect')
    axes[0,0].set_xlim([0,1])
    axes[0,0].set_xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],['0','10','20','30','40','50','60','70','80','90','100'],fontsize=18)
    axes[0,0].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'],fontsize=18)
    axes[0,0].set_ylim([0,1])
    axes[0,0].set_ylabel('Observed Relative Frequency',fontsize=18)
    axes[0,0].grid(True)
    axes[0,0].set_title('IC/CC Flashes',fontsize=24)
    axes[0,0].plot(np.mean(cc_climo)*np.ones(100),np.linspace(0,1,100),linestyle='dashdot',linewidth=3.0,color='grey',label='IC/CC Climatology',zorder=0)
    axes[0,0].plot(np.linspace(0,1,100),np.mean(cc_climo)*np.ones(100),linestyle='dashdot',linewidth=3.0,color='grey',zorder=0)
    axes[0,0].plot(perfect,no_skill_cc,linestyle='solid',linewidth=5.0,color='grey',label='No Skill',zorder=0)
    axes[0,0].legend(fontsize=18,loc='upper left', bbox_to_anchor=(0.11, 0.999))
    axes[0,0].text(.015,.93,'(a)',fontsize=24,weight='heavy',path_effects=pe1)

    cg_climo =cg_data['climo']
    no_skill_cg = (cg_climo+perfect)/2
    axes[0,1].plot(perfect,perfect,linestyle='--',color='grey',linewidth=3.0,label='Perfect')
    axes[0,1].set_xlim([0,1])
    axes[0,1].set_xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],['0','10','20','30','40','50','60','70','80','90','100'],fontsize=18)
    axes[0,1].set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'],fontsize=18)
    axes[0,1].set_ylim([0,1])
    axes[0,1].plot(np.mean(cg_climo)*np.ones(100),np.linspace(0,1,100),linestyle='dashdot',linewidth=3.0,color='grey',zorder=0,label='CG Climatology')
    axes[0,1].plot(np.linspace(0,1,100),np.mean(cg_climo)*np.ones(100),linestyle='dashdot',linewidth=3.0,color='grey',zorder=0)
    axes[0,1].plot(perfect,no_skill_cg,linestyle='solid',linewidth=5.0,color='grey',zorder=0,label='CG - No Skill')
    axes[0,1].grid(True)
    axes[0,1].legend(fontsize=18,loc='upper left', bbox_to_anchor=(0.11, 0.999))
    axes[0,1].set_title('CG Flashes',fontsize=24)
    axes[0,1].text(.015,.93,'(b)',fontsize=24,weight='heavy',path_effects=pe1)
    

    thresh = np.arange(0.00,1.05,0.05)
    thresh_centers =  0.5 * (thresh[:-1] + thresh[1:])
    counts = pickle.load(open('%sforecast_hist_dict.pkl'%(load_dir),'rb'))
    for t,t_key in enumerate(t_keys):
        cc_counts = counts['thresh_cc_counts_%s'%t_key]
        cg_counts = counts['thresh_cg_counts_%s'%t_key]
        axes[1,0].bar(thresh-.025,cc_counts,width=0.05, color=colors[t],alpha=.4, edgecolor='black',zorder=3,label='IC Counts - %s'%(fcst_times[t]))
        axes[1,1].bar(thresh-.025,cg_counts,width=0.05, color=colors[t],alpha=.4, edgecolor='black',zorder=3,label='CG Counts - %s'%(fcst_times[t]))
    
    axes[1,0].grid(True,zorder=0)    
    axes[1,0].set_yscale('log')
    axes[1,0].set_xlim([0,1])
    axes[1,0].set_ylabel('Count',fontsize=18)
    axes[1,0].set_xticks(thresh[::2],['0','10','20','30','40','50','60','70','80','90','100'],fontsize=18)
    axes[1,0].set_yticks([10,10**(2),10**(3),10**(4),10.0**(5),10.0**(6),10.0**(7)],['10','$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$','$10^{6}$','$10^{7}$'],fontsize=18)
    axes[1,0].set_ylim([10,10**7.2])
    axes[1,0].set_xlabel('Forecast Probability (%)',fontsize=18)
    axes[1,0].text(.915,10**7.0,'(c)',fontsize=24,weight='heavy',path_effects=pe1)
    axes[1,0].legend(fontsize=18,loc='upper center')

    axes[1,1].grid(True,zorder=0)
    axes[1,1].set_yscale('log')
    axes[1,1].set_xlim([0,1])
    axes[1,1].set_xticks(thresh[::2],['0','10','20','30','40','50','60','70','80','90','100'],fontsize=18)
    axes[1,1].set_yticks([10**(1),10**(2),10**(3),10**(4),10.0**(5),10.0**(6),10.0**(7)],['10','$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$','$10^{6}$','$10^{7}$'],fontsize=18)
    axes[1,1].set_ylim([10,10**7.2])
    axes[1,1].set_xlabel('Forecast Probability (%)',fontsize=18)
    axes[1,1].text(.915,10**7.0,'(d)',fontsize=24,weight='heavy',path_effects=pe1)
    axes[1,1].legend(fontsize=18,loc='upper center')

    plt.show()
    plt.savefig('temp_attr.png')
    plt.savefig('%sReliability_Diagram_forecast_lead_times.png'%(load_dir))
    plt.savefig('%sReliability_Diagram_forecast_lead_times.pdf'%(load_dir))
    plt.close()

def main():
    #command to run
    #python 1_LC_reliability.py @txt_exp_eval.txt 
    print('LC_eval_attributes.py main function')
    
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

    LaunchCast_pred_hist_calc_np(save_dir=exp_dir, y_true=y_true,y_pred=y_pred)
    LaunchCast_attributes_calc_np(save_dir=exp_dir,y_true=y_true,y_pred=y_pred)
    LaunchCast_attributes_plot(load_dir=exp_dir)

if __name__=='__main__':
    main()
    print('END OF LC_eval_attributes.py')