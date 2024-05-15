import sys
import argparse
import pickle
import pandas as pd
import wandb
import socket
import matplotlib.pyplot as plt
import shutil 

import tensorflow as tf
from tensorflow import keras


#import homework 8 specific code
from LC_data_loader import *
from LC_parser import *
from LC_unet_classifier import *



#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=18
plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
#################################################################

#################################################################
def check_args(args):
    '''
    Check that the input arguments are rational

    '''
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    assert (args.cpus_per_task is None or args.cpus_per_task > 1), "cpus_per_task must be positive or None"
    
 
    
#################################################################

def generate_fname(args):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.
    '''


    # Label
    if args.label is None:
        label_str = ""
    else:
        label_str = "%s_"%args.label

    # Put it all together, including #of training folds and the experiment rotation
    return 'LC_'+label_str

def execute_exp(args=None, multi_gpus=False):

    #Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])

    # Scale the batch size with the number of GPUs
    if multi_gpus > 1:
        args.batch = args.batch*multi_gpus

    print('Batch size', args.batch)

    ####################################################
    # Create the TF datasets for training, validation, testing

    if args.verbose >= 3:
        print('Starting data flow')

    image_size=args.image_size[0:2][0]
    nchannels = args.image_size[2]
    print('image_size')
    print(image_size)
    print('n_channels')
    print(nchannels)
    

    if args.load_data:
        #load the data
        print('loading the data')

    ####################################################

    

    # Output file base and pkl file
    fbase = generate_fname(args)
    print(fbase)
    fname_out = "%s_results.pkl"%fbase
    print(fname_out)

   

    

    
    if args.build_model:
        print('building the model')
        model = create_unet(image_size=image_size,
                        hrrr=args.hrrr,
                        n_hrrr_params=args.n_hrrr_params,
                        ksc_wx_twr=args.ksc_wx_twr,
                        n_ksc_twr_params=args.n_wxtwr_params,
                        ksc_efm = args.ksc_efm,
                        filters=args.conv_nfilters,
                        conv_size=args.conv_size,
                        pool_size=args.pool,
                        deep=args.deep,
                        n_conv_per_step=args.n_conv_per_step,
                        nsteps=args.n_time_steps,#time
                        lrate=args.lrate,
                        n_classes=3,#int, number of predictions
                        loss=tf.keras.losses.MeanSquaredError(),#tensor flow loss function
                        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),#tensor flow metrics
                        padding=args.padding,#string, same,valid,etc.
                        strides=args.stride,#int, pixel stride
                        conv_activation=args.activation_conv,
                        last_activation=args.activation_last,
                        batch_normalization=args.batch_normalization,
                        dropout=args.dropout,
                        skip=args.skip)

        print(model.summary())
      
    # Plot the model if the model is built
    if args.render and args.build_model:
        render_fname = './results/%s_model_plot.png'%fbase
        plot_model(model, to_file=render_fname, show_shapes=True, show_layer_names=True)

     # Perform the experiment?
    if args.nogo:
        # No!
        print("NO GO")
        return

    # Check if output file already exists
    if not args.force and os.path.exists(fname_out):
        # Results file does exist: exit
        print("File %s already exists"%fname_out)
        return

    #####
    # Start wandb
    run = wandb.init(project=args.project, name=args.label, notes=fbase, config=vars(args))

    # Log hostname
    wandb.log({'hostname': socket.gethostname()})

    # Log model design image
    if args.render:
        wandb.log({'model architecture': wandb.Image(render_fname)})

            
    #####
    # Callbacks
    cbs = []

    if args.early_stopping:
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True,
                                                        min_delta=args.min_delta, monitor=args.monitor)
        cbs.append(early_stopping_cb)
    
    cbs.append(tf.keras.callbacks.ModelCheckpoint(filepath='checkpoint.model.keras',
                                                    monitor='val_loss',
                                                    mode='auto',
                                                    save_best_only=False,
                                                    save_freq='epoch'))

    # Weights and Biases
    wandb_metrics_cb = wandb.keras.WandbMetricsLogger()
    cbs.append(wandb_metrics_cb)

    if args.verbose >= 3:
        print('Fitting model')
    
    #train the model
    history = model.fit(ds_train,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        use_multiprocessing=True, 
                        verbose=args.verbose>=2,
                        validation_data=ds_valid,
                        validation_steps=None,
                        callbacks=cbs)

    # Done training
    print('Done Training')
    # Generate results data
    results = {}
    results['history'] = history.history

    # Save results
    fbase = generate_fname(args)
    results['fname_base'] = fbase
    with open("./results/%s_results.pkl"%(fbase), "wb") as fp:
        pickle.dump(results, fp)
    
    # Save model
    if args.save_model:
        print('saving the model')
        model.save("./results/%s_model"%(fbase))

    wandb.finish()

    return model



if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    #n_physical_devices = 0

    if args.verbose >= 3:
        print('Arguments parsed')

    # Turn off GPU?
    if not args.gpu or "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        tf.config.set_visible_devices([], 'GPU')
        print('NO VISIBLE DEVICES!!!!')


    # GPU check
    # visible_devices = tf.config.get_visible_devices('GPU') 
    # n_visible_devices = len(visible_devices)
    # print('GPUS:', visible_devices)
    # if n_visible_devices > 0:
    #     for device in visible_devices:
    #         tf.config.experimental.set_memory_growth(device, True)
    #     print('We have %d GPUs\n'%n_visible_devices)
    # else:
    #     print('NO GPU')

    # Set number of threads, if it is specified
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)

    execute_exp(args, multi_gpus=n_visible_devices)