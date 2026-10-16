import tensorflow as tf
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from LC_parser import create_parser
from LC_model_64216 import Stack, FlattenStations, UnflattenStations
import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from scipy.ndimage import uniform_filter
from sklearn.metrics import average_precision_score, precision_recall_curve,roc_auc_score,confusion_matrix

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


def main(args):
    print('calculating the hrrr stats')
    load_dir = '/scratch/bmac87/'
    ds = xr.open_dataset(load_dir+'LC_2024_test_ds_w_hrrr_ltg_f01_f02.nc')

    hrrr_f01 = ds['hrrr_ltg_f001'].values[:,24:40,24:40]
    hrrr_f01 = np.expand_dims(hrrr_f01,axis=1)
    print(hrrr_f01.shape)

    hrrr_f02 = ds['hrrr_ltg_f002'].values[:,24:40,24:40]
    hrrr_f02 = np.expand_dims(hrrr_f02,axis=1)
    print(hrrr_f02.shape)
    
    y_true = ds['targets'].values[:,:,24:40,24:40]
    print(y_true.shape)

    bl_binary = .04
    f01_binary = np.where(hrrr_f01>=bl_binary,1.0,0.0)
    f01_binary = np.tile(f01_binary,(1, 4, 1, 1))

    f02_binary = np.where(hrrr_f02>=bl_binary,1.0,0.0)
    f02_binary = np.tile(f02_binary,(1, 4, 1, 1))
    stat_dict={}
    for t,t_key in enumerate(['t0','t1','t2','t3']):
        f0 = np.mean(y_true[:,t,:,:])
        fss_5h1 = calculate_fss(forecast=f01_binary[:,t,:,:], observation=y_true[:,t,:,:], window_size=7)
        # print(t,t_key,'fss_5nm_h1',f"{fss_5h1:.03f}",'fss to beat',f"{.5+f0/2:.03f}")
        fss_10h1 = calculate_fss(forecast=f01_binary[:,t,:,:], observation=y_true[:,t,:,:], window_size=13)
        # print(t,t_key,'fss_10nm_h1',f"{fss_10h1:.03f}",'fss to beat',f"{.5+f0/2:.03f}")
        
        y_true_1d = np.ravel(y_true[:,t,:,:])
        y_pred_1d = np.ravel(f01_binary[:,t,:,:])

        precision, recall, thresholds = precision_recall_curve(y_true_1d, y_pred_1d)
        auc_pr = average_precision_score(y_true_1d,y_pred_1d)
        auc_roc = roc_auc_score(y_true_1d,y_pred_1d)

        tn, fp, fn, tp = confusion_matrix(y_true_1d, y_pred_1d).ravel()
        far1 = fp / (tp + fp)
        fpr1 = fp/(fp+tn)

        print(t,t_key,'f01','precision',f"{precision[1]:.02f}",'recall',f"{recall[1]:.02f}", 
                'auc_pr',f"{auc_pr:.02f}",'auc_roc',f"{auc_roc:.02f}",'far1',f"{far1:.02f}",'fpr1',f"{fpr1:.02f}")
        stat_dict.update({t_key:{'f01_precision':precision,'f01_recall':recall,
                            'auc_pr_f01':auc_pr,'auc_roc_f01':auc_roc,
                            'fss5h1':fss_5h1,'fss10h1':fss_10h1,'far1':far1,'fpr1':fpr1}})
    pickle.dump(stat_dict,open('hrrrf01_stats.pkl','wb'))

    stat_dict={}
    for t,t_key in enumerate(['t0','t1','t2','t3']):
        y_true_1d = np.ravel(y_true[:,t,:,:])
        y_pred_1d2 = np.ravel(f02_binary[:,t,:,:])

        fss_5h2 = calculate_fss(forecast=f02_binary[:,t,:,:], observation=y_true[:,t,:,:], window_size=7)
        # print(t,t_key,'fss_5nm_h2',f"{fss_5h2:.03f}",'fss to beat',f"{.5+f0/2:.03f}")

        fss_10h2 = calculate_fss(forecast=f02_binary[:,t,:,:], observation=y_true[:,t,:,:], window_size=13)
        # print(t,t_key,'fss_10nm_h2',f"{fss_10h2:.03f}",'fss to beat',f"{.5+f0/2:.03f}")

        precision2, recall2, thresholds2 = precision_recall_curve(y_true_1d, y_pred_1d2)
        auc_pr2 = average_precision_score(y_true_1d,y_pred_1d2)
        auc_roc2 = roc_auc_score(y_true_1d,y_pred_1d2)
        tn, fp, fn, tp = confusion_matrix(y_true_1d, y_pred_1d2).ravel()
        far2 = fp / (tp + fp)
        fpr2 = fp/(fp+tn)

        print(t,t_key,'f02','precision',f"{precision2[1]:.02f}",'recall',f"{recall2[1]:.02f}",
                'auc_pr',f"{auc_pr2:.02f}",'auc_roc',f"{auc_roc2:.02f}",'far2',f"{far2:.02f}",'fpr2',f"{fpr2:.02f}")
        stat_dict.update({t_key:{'f02_precision':precision2,'f02_recall':recall2,'auc_pr_f02':auc_pr2,'auc_roc_f02':auc_roc2,
                        'fss5h2':fss_5h2,'fss10h2':fss_10h2,'far2':far2,'fpr2':fpr2}})
    pickle.dump(stat_dict,open('hrrrf02_stats.pkl','wb'))

if __name__=='__main__':
    parser=create_parser()
    args=parser.parse_args()
    main(args=args)