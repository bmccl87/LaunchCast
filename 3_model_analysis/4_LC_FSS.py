from scipy.ndimage import uniform_filter
import numpy as np
import os
import pickle
import matplotlib
import matplotlib.patheffects as path_effects
from LC_parser import *

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

    print('calculating FSS by forecast increment')
    print('parsing args')
    parser = create_parser()
    args = parser.parse_args()
    print('args parsed')

    exp_name=args.exp_name
    fname = 'labels_prob_outputs.pkl'
    exp_dir = '%s/%s/%s/'%(args.results_path,args.project,exp_name)
    data_dict = pickle.load(open(exp_dir+fname,'rb'))
    
    y_true = data_dict['y_true']
    y_pred = data_dict['y_pred']

    print(y_true.shape)

    window_size=5
    threshold=.30
    y_pred_thresholded = np.zeros(y_pred.shape)
    y_pred_thresholded[y_pred>=threshold] = 1.0
    for t,t_key in enumerate(['t0','t1','t2','t3']):
        if t>=0:
            print(t,t_key)
            f0 = np.mean(y_true[:,t,:,:])
            fss = calculate_fss(forecast=y_pred_thresholded[:,t,:,:], observation=y_true[:,t,:,:], window_size=window_size)
            print(t,t_key,'fss',f"{fss:.03f}",'fss to beat',f"{.5+f0/2:.03f}")

if __name__=='__main__':
    main()