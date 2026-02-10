import numpy as np
import os
import xarray as xr
import shutil
import cartopy.crs as ccrs
import pickle

from gewitter_functions import get_contingency_table,make_performance_diagram_axis,get_acc,get_pod,get_sr,csi_from_sr_and_pod
from sklearn.metrics import auc, precision_recall_curve

import tensorflow as tf

from LC_parser import *
from LC_models import Stack

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects

def main():
    
    parser = create_parser()
    args = parser.parse_args()

    print('0_LC_calibrate.py')
    exp_name = args.exp_name
    exp_dir = '%s%s/'%(args.results_path,args.project)

    model_dir = exp_dir+exp_name+'/'
    model_fname = 'model.keras'
    print(model_dir+model_fname)
    model = tf.keras.models.load_model(model_dir+model_fname,custom_objects={"Stack": Stack})
    model.trainable = False

    #make the prediction on the validation dataset and get the true labels
    val_tfds = tf.data.Dataset.load('/scratch/bmac87/%s_2023.tfds'%args.exp_pre)
    val_tfds = val_tfds.batch(4)
    val_logits = model.predict(val_tfds)

    y_val = tf.concat([y for _, y in val_tfds], axis=0).numpy()#true labels
    
    #convert to tf format
    val_logits = tf.cast(val_logits, tf.float32)
    val_logits = tf.squeeze(val_logits)

    y_val  = tf.cast(y_val, tf.float32)
    y_val = tf.squeeze(y_val)

    #temperature calibration
    calibration_dict = {}
    # for l,ltg in enumerate(['cc','cg']):
    for t,t_key in enumerate(['t0','t1','t2','t3']):
        T = tf.Variable(1.0, trainable=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        for _ in range(200):
            with tf.GradientTape() as tape:
                labels_sub = y_val[:,t,:,:]
                logits_sub = val_logits[:,t,:,:]
                calibrated_logits = logits_sub / T
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_sub,logits=calibrated_logits))
            grads = tape.gradient(loss, [T])
            optimizer.apply_gradients(zip(grads, [T]))
        print(t_key,'T',T.numpy())
        calibration_dict.update({'%s'%(t_key):T.numpy()})
        del T, optimizer, calibrated_logits
        
    pickle.dump(calibration_dict,open(model_dir+'calibration_dict.pkl','wb'))
    print('CALIBRATION COMPLETE')

if __name__=='__main__':
    main()