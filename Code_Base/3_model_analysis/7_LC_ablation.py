"""
This script will run the ablation for LaunchCast.
"""
from LC_parser import create_parser
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pickle
import keras
import pandas as pd
import time
from scipy.ndimage import uniform_filter
from LC_model_64216 import Stack, FlattenStations, UnflattenStations

def calculated_ablated_normalized_values(args):
    print('generating the ablation values')
    efm_stat_nothings = {'mean':250,'std':0.0,'var':0.0,'count_gt_thresh':0.0,'rms':0.0,'max_abs':250}
    min_max_dict = pickle.load(open('./forecast_min_maxes.pkl','rb'))
    for key in min_max_dict:
        print(key)

    data_dir = args.data_path
    fname = '%s_2024.tfds'%args.exp_pre
    tfds = tf.data.Dataset.load(data_dir+fname)
    for X, y in tfds.take(1):
        for key in X:
            print('from_tfds',key)
    del X,y
    
    ablation_dict = {}
    for t_key in args.t_keys:
        ablation_dict.update({'MRMS_low_zdr_dB_%s'%t_key:np.float32(-5.0),'MRMS_low_zdr_dB_mask_%s'%t_key:np.float32(0.0)})
        ablation_dict.update({'MRMS_mid_zdr_dB_%s'%t_key:np.float32(-5.0),'MRMS_mid_zdr_dB_mask_%s'%t_key:np.float32(0.0)})
        ablation_dict.update({'MRMS_high_zdr_dB_%s'%t_key:np.float32(-5.0),'MRMS_high_zdr_dB_mask_%s'%t_key:np.float32(0.0)})
        
        Z_nothing = 0.0
        for z_key in args.Z_keys:
            mean = min_max_dict[z_key]['mean']
            std = min_max_dict[z_key]['std']
            norm_value = (Z_nothing-mean)/std
            ablation_dict.update({'MRMS_%s_%s'%(z_key,t_key):norm_value})
            
        VI_nothing = np.log1p(0.0)
        VI_mean = min_max_dict['VII_00.50_log']['mean']
        VI_std = min_max_dict['VII_00.50_log']['std']
        VI_abl_norm = (VI_nothing-VI_mean)/VI_std
        ablation_dict.update({'MRMS_VII_00.50_log_%s'%t_key:VI_abl_norm})

        for hrrr_feature in args.hrrr_features:
            norm_value = min_max_dict[z_key]['mean']
            print('hrrr_feature','HRRR_%s_%s'%(hrrr_feature,t_key))
            ablation_dict.update({'HRRR_%s_%s'%(hrrr_feature,t_key):norm_value})
        
        glm_nothing=np.log1p(0.0)
        glm_mean = min_max_dict['group_area_log']['mean'] 
        glm_std = min_max_dict['group_area_log']['std']
        glm_abl_norm = (glm_nothing-glm_mean)/(glm_std)
        ablation_dict.update({'GLM_group_area_log_%s'%t_key:glm_abl_norm})
    pickle.dump(ablation_dict,open('./gridded_ablation_dict.pkl','wb'))

    for key in ablation_dict:
        print('ablation_dict')
        print(key)
    efm_norm_stats = pickle.load(open('./efm_forecast_min_max.pkl','rb'))
    fm_abl_values = {}
    for key in args.efm_keys:
        print(key)
        temp_dict = efm_norm_stats[key]
        efm_norm_dict = {}
        for ftr in efm_stat_nothings:
            efm_norm_dict.update({'mean':0.0})
            if ftr=='count_gt_thresh':
                mn = temp_dict[ftr]['mean']
                std = temp_dict[ftr]['std']
                abl_value = (0.0-mn)/std
                efm_norm_dict.update({'count_gt_thresh':abl_value})
            elif ftr=='std' or ftr=='var':
                mn = temp_dict[ftr]['log_mean']
                std = temp_dict[ftr]['log_std']
                abl_value = (np.log(0.0+1e-6)-mn)/(std)
                efm_norm_dict.update({ftr:abl_value})
            elif ftr=='rms' or ftr=='mean':#count greater than 1000
                abl_value=0.0
                efm_norm_dict.update({ftr:abl_value})
                pass
            elif ftr=='max_abs':
                value = temp_dict['mean']['mean']
                mn = temp_dict[ftr]['mean']
                std = temp_dict[ftr]['std']
                abl_value = (value-mn)/std
                efm_norm_dict.update({ftr:abl_value})
            else:
                pass
        
        fm_abl_values.update({key:efm_norm_dict})
    print(fm_abl_values)
    pickle.dump(fm_abl_values,open('./efm_ablation_dict.pkl','wb'))
    

def calculate_fss(forecast, observation, window_size):
    """
    Calculates the Fractions Skill Score (FSS) for binary fields.  This is from gemini. 
    
    Parameters:
    - forecast: a thresholded 3D numpy array (binary: 0 or 1), generated from the model predictions
    - observation: 3D numpy array (binary: 0 or 1)
    - window_size: Integer, the size of the neighborhood (e.g., 3, 5, 11)
    
    Returns:
    - fss: The Fractions Skill Score (float)
    """

    # Ensure inputs are floats for calculation
    f = np.squeeze(forecast.astype(float))
    o = np.squeeze(observation.astype(float))
    window_shape = (1,window_size,window_size)
    
    # 1. Generate fractional coverage using a uniform filter (neighborhood mean)
    # The 'constant' mode with cval=0 treats edges as if no events occurred outside
    f_frac = uniform_filter(f, size=window_shape, mode='constant', cval=0.0)
    o_frac = uniform_filter(o, size=window_shape, mode='constant', cval=0.0)
    
    # 2. Calculate the Mean Squared Error of the fractions
    mse = np.mean((f_frac - o_frac)**2)
    
    # 3. Calculate the reference MSE (the worst possible MSE)
    mse_ref = np.mean(f_frac**2) + np.mean(o_frac**2)

    # 4. Calculate FSS
    if mse_ref == 0:
        return 1.0 if np.array_equal(f, o) else 0.0
        
    fss = 1 - (mse / mse_ref)
    return fss

gridded_abl_values = pickle.load(open('./gridded_ablation_dict.pkl','rb'))
efm_abl_values = pickle.load(open('./efm_ablation_dict.pkl','rb'))
def ablate_selected_keys(features, labels, keys_to_ablate):
    
    for key in keys_to_ablate:
        if key.startswith("FM"):
            temp_values = []
            for stat in ['count_gt_thresh','max_abs', 'mean', 'rms', 'std','var']:
                temp_values.append(efm_abl_values[key][stat])
            abl_vec = tf.constant(temp_values, dtype=features[key].dtype)  # (6,)
            mask = tf.ones_like(features[key])  # (120, 6)
            features[key] = mask * abl_vec
        else:
            abl_value = tf.cast(gridded_abl_values[key], features[key].dtype)
            features[key] = tf.ones_like(features[key]) * abl_value       
    return features, labels

def build_exp_list():
    exp_dict = {}
    exp_id = 0
    
    mrms_flag = [True,False]
    glm_flag =  [True,False]
    efm_flag =  [True,False]
    hrrr_flag = [True,False]

    for mrms in mrms_flag:
        for glm in glm_flag:
            for efm in efm_flag:
                for hrrr in hrrr_flag:
                    temp_dict = {'mrms':mrms,
                                    'glm':glm,
                                    'efm':efm,
                                    'hrrr':hrrr}
                    exp_dict.update({exp_id:temp_dict})
                    exp_id+=1
    print(exp_id,' ablation experiments')
    return exp_dict

def create_ablated_tfds(args,exp_dict={}):
    time_keys = ['t0','t1','t2','t3']
    data_dir = args.data_path
    fname = '%s_2024.tfds'%args.exp_pre
    tfds = tf.data.Dataset.load(data_dir+fname) 
    for exp in exp_dict:
        
        keys_to_ablate = []
    
        mrms = exp_dict[exp]['mrms']
        glm = exp_dict[exp]['glm']
        efm = exp_dict[exp]['efm']
        hrrr = exp_dict[exp]['hrrr']

        #false means do not use the data
        if mrms==False:
            radar_keys = args.Z_keys+args.Zdr_keys+args.VI_keys
            for r,radar_key in enumerate(radar_keys):
                for t,t_key in enumerate(time_keys):
                    keys_to_ablate.append('MRMS_%s_%s'%(radar_key,t_key))

        if hrrr==False:
            hrrr_features = args.hrrr_features
            for h,hrrr_feature in enumerate(hrrr_features):
                for t,t_key in enumerate(time_keys):
                    keys_to_ablate.append('HRRR_%s_%s'%(hrrr_feature,t_key))

        if glm==False:
            glm_keys = args.GLM_keys
            for g,glm_key in enumerate(glm_keys):
                for t,t_key in enumerate(time_keys):
                    keys_to_ablate.append('GLM_%s_%s'%(glm_key,t_key))

        if efm==False:
            efm_keys = args.efm_keys
            for e,efm_key in enumerate(efm_keys):
                keys_to_ablate.append(efm_key)

        ablated_tfds = tfds.map(lambda features, labels, k=keys_to_ablate:
                                ablate_selected_keys(features, labels, k),
                                num_parallel_calls=tf.data.AUTOTUNE)
        ablated_tfds.save('/scratch/bmac87/LC_ablation_datasets_v3.2/exp_%s_ablation.tfds'%(exp))
        del  ablated_tfds,keys_to_ablate, mrms, glm, hrrr, efm

def test_ablated_tfds(args,exp_dict={}):
    zero_tfds = zero_tfds.batch(args.batch)
    tfds = tfds.batch(args.batch)
    x, y = next(iter(zero_tfds.take(1)))
    for k in keys_to_zero:
        v = x[k]
        print(k, "shape:", v.shape,
            "min:", tf.reduce_min(v).numpy(),
            "max:", tf.reduce_max(v).numpy(),
            "sum:", tf.reduce_sum(v).numpy())

    x, y = next(iter(tfds.take(1)))
    for k in keys_to_zero:
        v = x[k]
        print(k, "shape:", v.shape,
            "min:", tf.reduce_min(v).numpy(),
            "max:", tf.reduce_max(v).numpy(),
            "sum:", tf.reduce_sum(v).numpy())

def ablation_outputs(exp=0,exp_dict={}):
    
    print('loading the ablation experiment dictionary')
    abl_dir = '/scratch/bmac87/LC_ablation_datasets_v3.2/'
    # exp_dict = pickle.load(open(abl_dir+'ablation_exp_dict.pkl','rb'))

    print('loading the calibration coefficients')
    exp_path = '%s%s/%s/'%(args.results_path,args.project,args.exp_name)
    cal_dict = pickle.load(open(exp_path+'calibration_dict.pkl','rb'))
    t_keys = ['t0','t1','t2','t3']
    
    print('loading the model')
    model_fname = 'model.keras'
    model = tf.keras.models.load_model(exp_path+model_fname,custom_objects={"Stack": Stack,"FlattenStations":FlattenStations,"UnflattenStations":UnflattenStations})
    losses = [tf.keras.losses.BinaryCrossentropy(from_logits=True)]
    opt = keras.optimizers.Adam(learning_rate=args.lrate, amsgrad=False)
    model.compile(optimizer=opt,loss=losses)
    print('model compiled successfully')
    exp = args.exp
    print('exp:',exp)

    print('loading the ablation dataset for exp: %s'%exp)
    abl_tfds = tf.data.Dataset.load(abl_dir+'exp_%s_ablation.tfds'%(exp))
    abl_tfds = abl_tfds.batch(args.batch)
    y_true = tf.concat([y for _, y in abl_tfds], axis=0).numpy()
    y_logits = np.squeeze(model.predict(abl_tfds))
    y_dict = {'y_true':y_true,'y_logits':y_logits}
    y_probs = np.zeros(y_logits.shape)
    for t,t_key in enumerate(t_keys):
        cal_T = cal_dict[t_key]
        y_probs[:,t,:,:] = tf.nn.sigmoid(y_logits[:,t,:,:]/cal_T)
    y_dict.update({'y_pred':y_probs})
    pickle.dump(y_dict,open('/scratch/bmac87/ablation_outputs_v3.2/exp_%s.pkl'%exp,'wb'))

def generate_stats(exp=0):

    print('running the stats... just keep runnin, runnin...')
    data_dir='/scratch/bmac87/ablation_outputs_v3.2/'
    pkl_name = 'exp_%s.pkl'%(exp)
    y_dict = pickle.load(open(data_dir+pkl_name,'rb'))
    
    y_true = y_dict['y_true']
    print('y_true.shape',y_true.shape)

    y_pred = y_dict['y_pred']
    print('y_pred.shape',y_pred.shape)

    print('generating the thresholds')
    thresh = np.arange(0.05,1.05,0.05)

    print('initializing the metrics')
    fn = tf.keras.metrics.FalseNegatives(thresholds=thresh.tolist(), name='false_neg', dtype=tf.dtypes.float32)
    fp = tf.keras.metrics.FalsePositives(thresholds=thresh.tolist(), name='false_pos', dtype=tf.dtypes.float32)
    tp = tf.keras.metrics.TruePositives(thresholds=thresh.tolist(), name='true_pos', dtype=tf.dtypes.float32)
    tn = tf.keras.metrics.TrueNegatives(thresholds=thresh.tolist(), name='true_neg', dtype=tf.dtypes.float32)

    auc_roc = tf.keras.metrics.AUC(curve='ROC',name='auc_roc',dtype=tf.dtypes.float32,thresholds=thresh.tolist(),from_logits=False)
    auc_pr = tf.keras.metrics.AUC(curve='PR',name='auc_pr',dtype=tf.dtypes.float32,thresholds=thresh.tolist(),from_logits=False)

    precision = tf.keras.metrics.Precision(thresholds=thresh.tolist(), name='precision', dtype=tf.dtypes.float32)
    recall = tf.keras.metrics.Recall(thresholds=thresh.tolist(), name='pod_recall', dtype=tf.dtypes.float32)
    print('metrics initialized correctly')

    store_dict = {'thresh':thresh}
    for t, t_key in enumerate(['t0','t1','t2','t3']):
        y_true_1d = np.squeeze(np.ravel(y_true[:,t,:,:]))
        y_pred_1d = np.squeeze(np.ravel(y_pred[:,t,:,:]))
        store_dict.update({t_key:{}})
        print('resetting the metric states')
        fn.reset_state()
        fp.reset_state()
        tp.reset_state()
        tn.reset_state()

        auc_roc.reset_state()
        auc_pr.reset_state()
        precision.reset_state()
        recall.reset_state()
        print('metrics reset successfully')

        print('tps:',t_key,'exp',exp)
        tps = tp(y_true_1d,y_pred_1d)        
        print(tps)
        store_dict[t_key].update({'tps':tps})

        print('fps:',t_key,'exp',exp)
        fps = fp(y_true_1d,y_pred_1d)
        print(fps)
        store_dict[t_key].update({'fps':fps})

        print('fns',t_key,'exp',exp)
        fns = fn(y_true_1d,y_pred_1d)
        print(fns)
        store_dict[t_key].update({'fns':fns})

        print('auc_roc_curve',t_key,'exp',exp)
        auc_roc_curve = auc_roc(y_true_1d,y_pred_1d)
        print(auc_roc_curve)
        store_dict[t_key].update({'auc_roc':auc_roc_curve})

        print('auc_pr_curve',t_key,'exp',exp)
        auc_pr_curve = auc_pr(y_true_1d,y_pred_1d)
        print(auc_pr_curve)
        store_dict[t_key].update({'auc_pr':auc_pr_curve})

        print('precision_values',t_key,'exp',exp)
        precision_values = precision(y_true_1d,y_pred_1d)
        print(precision_values)
        store_dict[t_key].update({'precision':precision_values})

        print('recall values',t_key,'exp',exp)
        recall_values = recall(y_true_1d,y_pred_1d)
        print(recall_values)
        store_dict[t_key].update({'recall':recall_values})

        print('calculating the pods:',t_key)
        pods = tps/(tps + fns)
        print(pods)
        store_dict[t_key].update({'pods':pods})

        print('calculating the srs:',t_key)
        srs = tps/(tps + fps)
        print(srs)
        store_dict[t_key].update({'srs':srs})

        print('calculating the far_ratio',t_key,'exp',exp)
        eps = tf.keras.backend.epsilon()
        far_ratio = fps / (tps + fps + eps)
        print(far_ratio)
        store_dict[t_key].update({'far_ratio':far_ratio})

        print('calculating the far_rate',t_key,'exp',exp)
        far_rate = 1-srs
        print(far_rate)
        store_dict[t_key].update({'far_rate':far_rate})
        
        print('calculcating the csi values:',t_key)
        csis = tps/(tps + fns + fps)
        if exp==0:
            max_thresh = thresh[np.argmax(csis)]
        else:#load the max threshold for the fss calculation
            temp_dict = pickle.load(open('/scratch/bmac87/ablation_stats_v3.2/abl_results_exp_0.pkl','rb'))
            max_thresh = temp_dict[t_key]['csi_max_thresh']
        print(csis)
        print('max csi:',t_key,np.max(csis))
        print('max csi threshold',t_key,thresh[np.argmax(csis)])
        store_dict[t_key].update({'csis':csis})
        store_dict[t_key].update({'csi_max_thresh':thresh[np.argmax(csis)]})
        store_dict[t_key].update({'max_csi':np.max(csis)})
        store_dict[t_key].update({'max_idx':np.argmax(csis)})

        #calculate fss for a window size of 7, that corresponds with the 5-nautical mile lightning
        y_pred_thresholded = np.zeros(y_pred.shape)
        y_pred_thresholded[y_pred>=max_thresh] = 1.0
        f0 = np.mean(y_true[:,t,:,:])
        fss_5 = calculate_fss(forecast=y_pred_thresholded[:,t,:,:], observation=y_true[:,t,:,:], window_size=7)
        print(t,t_key,'fss_5nm',f"{fss_5:.03f}",'fss to beat',f"{.5+f0/2:.03f}")
        store_dict[t_key].update({'FSS_5nm':fss_5})

        #calculate fss for a window size of 13, that corresponds with the 10-nautical mile lightning
        fss_10 = calculate_fss(forecast=y_pred_thresholded[:,t,:,:],observation=y_true[:,t,:,:],window_size=13)
        print(t,t_key,'fss_10nm',f"{fss_10:.03f}",'fss to beat',f"{.5+f0/2:.03f}")
        store_dict[t_key].update({'FSS_10nm':fss_10})

    pickle.dump(store_dict,open('/scratch/bmac87/ablation_stats_v3.2/abl_results_exp_%s.pkl'%exp,'wb'))
    
def build_df(exp_dict):
    t_keys = ['t0','t1','t2','t3']
    print('initializing the df to store the metrics')
    df_columns = ['EFM','MRMS','GLM','HRRR']
    tf_stats = ['csi_max_thresh','max_csi','max_idx','pod','far_rate','far_ratio','auc_roc','auc_pr','FSS_5nm','FSS_10nm']
    
    for tf_stat in tf_stats:
        for t_key in t_keys:
            df_columns.append('%s_%s'%(tf_stat,t_key))
    df_indexes = range(16)
    df = pd.DataFrame(index=df_indexes, columns=df_columns)
    df['exp'] = df_indexes
    for i in df_indexes:
        booleans = exp_dict[i]
        df.loc[df.index[i],'EFM'] = booleans['efm']
        df.loc[df.index[i],'MRMS'] = booleans['mrms']
        df.loc[df.index[i], 'GLM'] = booleans['glm']
        df.loc[df.index[i], 'HRRR'] = booleans['hrrr']
    df.to_pickle(open('/scratch/bmac87/ablation_stats_df.pkl','wb'))
    print('metrics df built and saved successfully')
    return df

def build_table(args,exp_dict={},exp_df=pd.DataFrame()):
    
    print('storing the experiment results in a data frame')
    print(exp_df.columns)
    t_keys = args.t_keys
    df_columns = ['EFM','MRMS','GLM','HRRR']
    tf_stats = ['csi_max_thresh','max_csi','max_idx','pod','far_rate','far_ratio','auc_roc','auc_pr','FSS_5nm','FSS_10nm']
    results_dir = '/scratch/bmac87/ablation_stats_v3.2/'
    
    for exp in exp_dict:
        print(exp)
        fname = 'abl_results_exp_%s.pkl'%(exp)
        temp_dict = pickle.load(open(results_dir+fname,'rb'))
        for stat_key in temp_dict:
            thresh = temp_dict['thresh']
            for t_key in t_keys:
                for tf_stat in tf_stats:
                    column_idx = '%s_%s'%(tf_stat,t_key)
                    print(t_key,tf_stat,column_idx)
                    if tf_stat=='pod':
                        pods = temp_dict[t_key]['pods']
                        pod = pods[temp_dict[t_key]['max_idx']]
                        exp_df.loc[exp,column_idx] = pod.numpy()

                    elif tf_stat=='far_rate':
                        far_rates = temp_dict[t_key]['far_rate']
                        far_rate = far_rates[temp_dict[t_key]['max_idx']]
                        exp_df.loc[exp,column_idx] = far_rate.numpy()

                    elif tf_stat=='far_ratio':
                        far_ratios = temp_dict[t_key]['far_ratio']
                        far_ratio = far_ratios[temp_dict[t_key]['max_idx']]
                        exp_df.loc[exp,column_idx] = far_ratio.numpy()

                    elif tf_stat=='auc_pr':
                        exp_df.loc[exp,column_idx] = temp_dict[t_key][tf_stat].numpy()

                    elif tf_stat=='auc_roc':
                        exp_df.loc[exp,column_idx] = temp_dict[t_key][tf_stat].numpy()

                    else:
                        exp_df.loc[exp,column_idx] = temp_dict[t_key][tf_stat]

    exp_df.to_pickle(open('ablation_table_df.pkl','wb'))
    exp_df.to_excel('ablation_table_df.xlsx')
    print(exp_df['auc_pr_t0'])
    latex_code = exp_df.style.format(formatter="{:.3g}").to_latex() # index=False prevents writing the pandas index as a column
    print(latex_code)

if __name__=='__main__':
    start_clock = time.time()
    
    print('parsing args')
    parser = create_parser()
    args = parser.parse_args()
    print('args parsed')

    visible_devices = tf.config.get_visible_devices('GPU') 
    print('visible_devices')
    print(visible_devices)

    # Set number of threads, if it is specified
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)

    exp_dict = build_exp_list()
    # calculated_ablated_normalized_values(args=args)
    # create_ablated_tfds(args,exp_dict=exp_dict)

    generate_output = False
    if generate_output==True:
        print('generating the ablation outputs')
        ablation_outputs(exp=args.exp,exp_dict=exp_dict)
        print('ablation outputs generated successfully')

    run_stats = False
    if run_stats==True:
        generate_stats(exp=args.exp)
    
    exp_df = build_df(exp_dict=exp_dict)
    exp_df = build_table(args=args,exp_dict=exp_dict,exp_df=exp_df)
    
    # end_clock = time.time()
    # min2run = (end_clock-start_clock)/60
    # print('time to ablate, in minutes')
    # print(min2run)
    # print('END OF 7_LC_ABLATION.py for exp',args.exp)