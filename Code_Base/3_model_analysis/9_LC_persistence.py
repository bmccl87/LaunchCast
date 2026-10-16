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
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

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

def main():
    print('generating the persistence forecast')
    data_dir= '/ourdisk/hpc/ai2es/bmac87/LaunchCast/data/v3_data_processing/'
    ds = xr.open_dataset('%sLC_Forecast_Final_log1p_GLM_VII_2024.nc'%data_dir,engine='netcdf4')
    for key in ds:
        print(key)

    #rule 6 requires shall wait for a flash within 30 minutes
    glm_llccr6_inputs = np.expand_dims(((ds['group_area_log'].values[:,2,:,:]>0).astype(np.float32)+
                                        (ds['group_area_log'].values[:,3,:,:]>0).astype(np.float32)),axis=1)
    glm_persistence = np.tile(glm_llccr6_inputs,(1, 4, 1, 1))

    glm_persistence[glm_persistence>0] = 1.0
    glm_persistence[glm_persistence<=0] = 0.0

    glm_persistence = glm_persistence[:,:,24:40,24:40]
    targets = ds['targets'].values[:,:,24:40,24:40]
    stat_dict = {}
    for t,t_key in enumerate(['t0','t1','t2','t3']):

        f0 = np.mean(targets[:,t,:,:])
        fss_5p = calculate_fss(forecast=glm_persistence[:,t,:,:], observation=targets[:,t,:,:], window_size=7)
        print(t,t_key,'fss_5nm_p',f"{fss_5p:.03f}",'fss to beat',f"{.5+f0/2:.03f}")

        fss_10p = calculate_fss(forecast=glm_persistence[:,t,:,:], observation=targets[:,t,:,:], window_size=13)
        print(t,t_key,'fss_10nm_p',f"{fss_10p:.03f}",'fss to beat',f"{.5+f0/2:.03f}")

        y_true_1d = np.ravel(targets[:,t,:,:])
        y_pred_1d = np.ravel(glm_persistence[:,t,:,:])

        precision, recall, thresholds = precision_recall_curve(y_true_1d, y_pred_1d)
        auc_pr = average_precision_score(y_true_1d,y_pred_1d)
        auc_roc = roc_auc_score(y_true_1d,y_pred_1d)

        tn, fp, fn, tp = confusion_matrix(y_true_1d, y_pred_1d).ravel()
        far = fp / (tp + fp)
        fpr = fp/(fp+tn)

        print(t,t_key,'precision',precision[1],'recall',recall[1],'auc_pr',auc_pr,'auc_roc',auc_roc,'faratio',far,'fpratio',fpr)
        stat_dict.update({t_key:{'precision':precision,'recall':recall,'auc_pr':auc_pr,'fss5p':fss_5p,'fss10p':fss_10p,
        'fss2beat':(.5+f0/2),'auc_roc':auc_roc,'faratio':far,'fpratio':fpr}})

    pickle.dump(stat_dict,open('./persistence_prec_recall.pkl','wb'))



if __name__=='__main__':
    main()