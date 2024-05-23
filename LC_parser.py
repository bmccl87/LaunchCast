'''
Advanced Machine Learning, 2024

Argument parser needed by multiple programs.

Author: Andrew H. Fagg (andrewhfagg@gmail.com)
'''

import argparse

def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='LaunchCast', fromfile_prefix_chars='@')

    # High-level info for WandB
    parser.add_argument('--project', type=str, default='LaunchCast', help='WandB project name')

    # High-level commands
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--force', action='store_true', help='Perform the experiment even if the it was completed previously')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    parser.add_argument('--load_data',action='store_true',default=False,help='Flag to load the data')
    parser.add_argument('--build_model',action='store_true',default=False,help='Flag to build the model')

    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu', help='Do not use the GPU')
    

    # High-level experiment configuration
    parser.add_argument('--label', type=str, default=None, help="Extra label to add to output files")
    parser.add_argument('--dataset', type=str, default='/home/fagg/datasets/radiant_earth/pa/', help='Data set directory')
    parser.add_argument('--image_size', nargs=3, type=int, default=[64,84,1], help="Size of input images (rows, cols, channels)")
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')
    parser.add_argument('--fold',type=int,default=0,help='Fold Number')

    # Specific experiment configuration
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lrate', type=float, default=0.00001, help="Learning rate")
    parser.add_argument('--end_digit',type=str,default='*0',help='The end number pertaining to the image file')
    parser.add_argument('--deep',type=int,default=2,help='How deep to build the model')
    parser.add_argument('--n_conv_per_step',type=int,default=2,help='The number of convolutions per deep step')
    
    # U-Net
    parser.add_argument('--conv_size', type=int, default=3, help='Convolution filter size per layer')
    parser.add_argument('--conv_nfilters', nargs='+', type=int, default=[10,15], help='Convolution filters per layer (sequence of ints)')
    parser.add_argument('--pool', type=int, default=2, help='Max pooling size (1=None)')
    parser.add_argument('--stride',type=int,default=1,help='Stride pixels')
    parser.add_argument('--padding', type=str, default='same', help='Padding type for convolutional layers')
    parser.add_argument('--activation_conv', type=str, default='elu', help='Activation function for convolutional layers')
    parser.add_argument('--activation_last',type=str,default='softmax',help='Last activation function')
    parser.add_argument('--skip',action='store_true',default=False,help='Build skip connections in the UNet')
    parser.add_argument('--no-skip',action='store_false',dest='skip',help='Do no use skip connections in the UNet')

    #LaunchCast parameters
    parser.add_argument('--hrrr',action='store_true',default=True,help='Use the HRRR data and grid the lightning data to the HRRR grid')
    parser.add_argument('--n_hrrr_params',type=int,default=5,help='The number of HRRR parameters like Temp, RH, etc.')
    parser.add_argument('--mrms',action='store_true',default=False,help='Use the MRMS data and grid the lightning data to the MRMS grid')
    parser.add_argument('--ksc_wx_twr',action='store_true',default=False,help='Load and grid the KSC data to the HRRR grid')
    parser.add_argument('--n_wxtwr_params',type=int,default=0,help='The number of KSC params like wind, temp, rh, etc.')
    parser.add_argument('--ksc_efm',action='store_true',default=False,help='Load and grid the EFM data for training')
    parser.add_argument('--int_2_hrrr_grid',action='store_true',default=False,help='Interpolate the data to the HRRR grid')
    parser.add_argument('--int_2_mrms_time',action='store_false',default=False,help='Temporally interpolate to the mrms times')

    #Regularization
    parser.add_argument('--dropout',type=float,default=0.0,help='Amount of dropout to use')
    parser.add_argument('--spatial_dropout',type=float,default=0.0,help='Amount of spatial dropout')
    parser.add_argument('--batch_normalization', action='store_true', help='Turn on batch normalization')

    # Early stopping
    parser.add_argument('--early_stopping',action='store_true',default=False,help='Use Early Stopping')
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('--monitor', type=str, default="val_loss", help="Metric to monitor for early termination")

    # Training parameters
    parser.add_argument('--batch', type=int, default=10, help="Training set batch size")
    parser.add_argument('--prefetch', type=int, default=3, help="Number of batches to prefetch")
    parser.add_argument('--num_parallel_calls', type=int, default=4, help="Number of threads to use during batch construction")
    parser.add_argument('--cache', type=str, default=None, help="Cache (default: none; RAM: specify empty string; else specify file")
    parser.add_argument('--shuffle', type=int, default=0, help="Size of the shuffle buffer (0 = no shuffle")
    parser.add_argument('--repeat', action='store_true', help='Continually repeat training set')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help="Number of training batches per epoch (must use --repeat if you are using this)")

    # Post
    parser.add_argument('--render', action='store_true', default=True, help='Write model image')
    parser.add_argument('--save_model', action='store_true', default=True, help='Save a model file')
    parser.add_argument('--no-save_model', action='store_false', dest='save_model', help='Do not save a model file')

    return parser

