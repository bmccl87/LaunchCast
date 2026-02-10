import argparse
import pickle

def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='LaunchCast', fromfile_prefix_chars='@')

    # High-level info for WandB
    parser.add_argument('--project', type=str, default='LaunchCast', help='WandB project name')
    parser.add_argument('--exp_pre',type=str,default='LC_',help='Prefix for the jobs')
    parser.add_argument('--exp_name',type=str,default='LC_default_exp')
    
    # High-level commands
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--force', action='store_true', default=False,help='Perform the experiment even if the it was completed previously')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    parser.add_argument('--load_data',action='store_true',default=False,help='Flag to load the data')
    parser.add_argument('--build_model',action='store_true',default=False,help='Flag to build the model')
    parser.add_argument('--train',action='store_true',default=False,help='Flag to train the model')
    parser.add_argument('--from_checkpt',action='store_true',default=False,help='Flag to train from a checkpoint')
    parser.add_argument('--years',nargs=4,type=str,default=['2021','2022','2023','2024'])

    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu', help='Do not use the GPU')
    
    # High-level experiment configuration
    parser.add_argument('--label', type=str, default=None, help="Extra label to add to output files")
    parser.add_argument('--input_image_size', nargs=3, type=int, default=[64,64,2], help="Size of input images for one time step (lat, lon, channels)")
    parser.add_argument('--output_image_size',nargs=4, type=int, default=[4,64,64,2], help="Size of output images (time, lat, lon, channels)")
    parser.add_argument('--rotation',type=int,default=1)
    parser.add_argument('--fbase_data',type=str,default='LC_parser')

    # Specific experiment configuration
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--lrate', type=float, default=0.00001, help="Learning rate")
    parser.add_argument('--kernel_size',type=int, default=4, help='2D Convolution filter size per layer')
    parser.add_argument('--pool_size', type=int, default=2, help='Avg/Max pooling size (1=None)')
    parser.add_argument('--pool_type',type=str,default='max',help='Flag for max or avg pooling')
    parser.add_argument('--strides',type=int,default=1,help='Stride pixels')
    parser.add_argument('--padding', type=str, default='same', help='Padding type for convolutional layers')
    parser.add_argument('--activation_conv', type=str, default='relu', help='Activation function for convolutional layers')
    parser.add_argument('--activation_last',type=str,default='linear',help='Last activation function')
    parser.add_argument('--encoder_deep',type=int,default=5,help='The number of layers in the UNet encoder. 5 - 2x2 at the bottom')
    parser.add_argument('--num_layers_per_block',type=int,default=1,help='The number of layers in each depth of the UNet')
    parser.add_argument('--skip',action='store_true',default=False,help='Flag for skip connections')
    parser.add_argument('--rotate',action='store_true',default=False,help='Flag for data augmentation, adding rotation')
    parser.add_argument('--rotate_dec',type=float,default=.1)
    parser.add_argument('--noise',action='store_true',default=False,help='Flag for data augmentation, adding gaussian noise')
    parser.add_argument('--noise_std',type=float,default=0.005)

    #efm_arguments
    parser.add_argument('--use_EFM',action='store_true',default=False)
    parser.add_argument('--efm_keys',nargs=31,type=str,default=['FM01', 'FM02', 'FM04', 'FM05', 'FM06', 'FM07', 'FM08', 'FM09', 'FM10', 'FM11', 'FM12', 'FM14', 'FM15', 'FM16', 'FM17', 'FM18', 'FM19', 'FM20', 'FM21', 'FM22', 'FM24', 'FM25', 'FM26', 'FM27', 'FM28', 'FM29', 'FM30', 'FM31', 'FM32', 'FM34', 'FM35'])
    parser.add_argument('--efm_ts_keys',nargs=31,type=str,default=['FM01', 'FM02', 'FM04', 'FM05', 'FM06', 'FM07', 'FM08', 'FM09', 'FM10', 'FM11', 'FM12', 'FM14', 'FM15', 'FM16', 'FM17', 'FM18', 'FM19', 'FM20', 'FM21', 'FM22', 'FM24', 'FM25', 'FM26', 'FM27', 'FM28', 'FM29', 'FM30', 'FM31', 'FM32', 'FM34', 'FM35'])
    parser.add_argument('--efm_dist_keys',nargs=31,type=str,default=['FM01_distances', 'FM02_distances', 'FM04_distances', 'FM05_distances', 'FM06_distances', 'FM07_distances', 'FM08_distances', 'FM09_distances', 'FM10_distances', 'FM11_distances', 'FM12_distances', 'FM14_distances', 'FM15_distances', 'FM16_distances', 'FM17_distances', 'FM18_distances', 'FM19_distances', 'FM20_distances', 'FM21_distances', 'FM22_distances', 'FM24_distances', 'FM25_distances', 'FM26_distances', 'FM27_distances', 'FM28_distances', 'FM29_distances', 'FM30_distances', 'FM31_distances', 'FM32_distances', 'FM34_distances', 'FM35_distances'])
    parser.add_argument('--efm_ts_encoder_deep',type=int,default=1)
    parser.add_argument('--efm_ts_encoder_filters',nargs=1,type=int, default=[32])
    parser.add_argument('--efm_ts_input_shape',nargs=2,type=int, default=[724,2])
    parser.add_argument('--efm_dist_encoder_deep',type=int,default=6)
    parser.add_argument('--efm_dist_encoder_filters',nargs=5,type=int,default=[32,64,128,256,512,1024])
    parser.add_argument('--efm_dist_input_shape',nargs=2,type=int,default=[64,64])
    parser.add_argument('--efm_stats',nargs=4,type=str,default=['median','min','max','std'])

    #HRRR_arguments
    parser.add_argument('--HRRR_input_shape',nargs=3,type=int,default=[64,64,9])
    parser.add_argument('--use_HRRR',action='store_true',default=False)
    parser.add_argument('--hrrr_features',nargs=9,type=str,default=['10m_divergence','temp_2m','td_2m','mslp','precip_rate','comp_reflectivity','vil','z_1000m','z_4000m'])

    #MRMS_arguments
    parser.add_argument('--MRMS_input_shape',nargs=3,type=int,default=[64,64,1])
    parser.add_argument('--use_MRMS',action='store_true',default=False)
    parser.add_argument('--use_VI',action='store_true',default=False)
    parser.add_argument('--use_Z',action='store_true',default=False)
    parser.add_argument('--use_Zdr',action='store_true',default=False)
    parser.add_argument('--Z_keys',nargs=5,type=str,default=['Reflectivity_0C_00.50','Reflectivity_-5C_00.50','Reflectivity_-10C_00.50', 'Reflectivity_-15C_00.50','Reflectivity_-20C_00.50'])
    parser.add_argument('--Zdr_keys',nargs=3,type=str,default=['low_zdr','mid_zdr','high_zdr'])
    parser.add_argument('--VI_keys',nargs=2,type=str,default=['VII_00.50', 'VIL_00.50'])

    #GLM_arguments
    parser.add_argument('--GLM_input_shape',nargs=3,type=int,default=[64,64,1])
    parser.add_argument('--GLM_keys',nargs=2,type=str,default=['group_area','group_energy'])
    parser.add_argument('--use_GLM',action='store_true',default=False)

    #MERLIN_arguments
    parser.add_argument('--merlin_keys',nargs=2,type=str,default=['cc_forecast','cg_forecast'])
    
    #Regularization
    parser.add_argument('--spatial_dropout',type=float,default=0.0,help='Amount of spatial dropout for the HRRR Unet')
    parser.add_argument('--batch_normalization', action='store_true', help='Turn on batch normalization')
    parser.add_argument('--L2_reg',type=float,default=0.0)
    
    # Early stopping
    parser.add_argument('--early_stopping',action='store_true',default=False,help='Use Early Stopping')
    parser.add_argument('--min_delta', type=float, default=0.0000001, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('--monitor', type=str, default="val_loss", help="Metric to monitor for early termination")

    # Training parameters
    parser.add_argument('--batch', type=int, default=4, help="Training set batch size")
    parser.add_argument('--prefetch', type=int, default=3, help="Number of batches to prefetch")
    parser.add_argument('--num_parallel_calls', type=int, default=4, help="Number of threads to use during batch construction")
    parser.add_argument('--cache', action='store_true', default=False, help="Cache if true put on tfds else use prefetch")
    parser.add_argument('--shuffle', type=int, default=0, help="Size of the shuffle buffer (0 = no shuffle")
    parser.add_argument('--repeat', action='store_true', help='Continually repeat training set')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help="Number of training batches per epoch (must use --repeat if you are using this)")

    #Post
    parser.add_argument('--render', action='store_true', default=True, help='Write model image')
    parser.add_argument('--no-render',action='store_false', dest='render',help='Do not write the model image')
    parser.add_argument('--save_model', action='store_true', default=True, help='Save a model file')
    parser.add_argument('--no-save_model', action='store_false', dest='save_model', help='Do not save a model file')
    parser.add_argument('--results_path',type=str,default='/scratch/bmac87/LaunchCast_scratch/',help='Where to save the results')
    parser.add_argument('--checkpt_path',type=str,default='/scratch/bmac87/LaunchCast_scratch/',help='Where to save the checkpoints')
    parser.add_argument('--save_output_labels',action='store_true',default=False,help='Flag on whether or not to save off the labels and the model predictions.')
    parser.add_argument('--make_PD',action='store_true',default=False,help='Flag on whether or not to make the performance diagram')
    parser.add_argument('--make_loss_fig',action='store_true',default=False,help='Flag on whether or not to make the loss function diagram')
    parser.add_argument('--make_output_figs',action='store_true',default=False,help='Flag on whether or not to test dataset output images, with formatting')
    parser.add_argument('--make_reliability_diagram',action='store_true',default=False)
    parser.add_argument('--data_load_exp',type=int,default=6,help='This is a number to help process the data quickly.')
    parser.add_argument('--data_path',type=str,default='/scratch/bmac87/',help='The path to the training, validation, and test tensorflow datasets.')
    parser.add_argument('--exp',type=int,default=1,help='This is an experiment number, to help in hyperparameter tuning. The parameters are often set inside the training script.')
    return parser
