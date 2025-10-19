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

    # High-level commands
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--force', action='store_true', default=False,help='Perform the experiment even if the it was completed previously')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    parser.add_argument('--load_data',action='store_true',default=False,help='Flag to load the data')
    parser.add_argument('--build_model',action='store_true',default=False,help='Flag to build the model')
    parser.add_argument('--train',action='store_true',default=False,help='Flag to train the model')

    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu', help='Do not use the GPU')
    
    # High-level experiment configuration
    parser.add_argument('--label', type=str, default=None, help="Extra label to add to output files")
    parser.add_argument('--hrrr_image_size', nargs=4, type=int, default=[4,16,16,4], help="Size of input images (rows, cols, filters)")
    parser.add_argument('--rotation',type=int,default=1,help='Rotation Number 1-7')

    # Specific experiment configuration
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--lrate', type=float, default=0.00001, help="Learning rate")
    parser.add_argument('--kernel_shape',type=int, default=4, help='2D Convolution filter size per layer')
    parser.add_argument('--pool_size', type=int, default=2, help='Avg/Max pooling size (1=None)')
    parser.add_argument('--pool_type',type=str,default='max',help='Flag for max or avg pooling')
    parser.add_argument('--stride',type=int,default=1,help='Stride pixels')
    parser.add_argument('--padding', type=str, default='same', help='Padding type for convolutional layers')
    parser.add_argument('--activation_conv', type=str, default='relu', help='Activation function for convolutional layers')
    parser.add_argument('--activation_last',type=str,default='linear',help='Last activation function')
    parser.add_argument('--encoder_deep',type=int,default=5,help='The number of layers in the UNet encoder. 5 - 2x2 at the bottom')
    parser.add_argument('--num_layers_per_block',type=int,default=3,help='The number of layers in each depth of the UNet')

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
    parser.add_argument('--cache', type=str, default=None, help="Cache (default: none; RAM: specify empty string; else specify file")
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
    parser.add_argument('--data_load_exp',type=int,default=6,help='This is a number to help process the data quickly.')
    parser.add_argument('--data_path',type=str,default='/scratch/bmac87/',help='The path to the training, validation, and test tensorflow datasets.')
    parser.add_argument('--exp',type=int,default=1,help='This is an experiment number, to help in hyperparameter tuning. The parameters are often set inside the training script.')
    return parser

def build_exp_dict():
    exp_count=0
    hyper_dict = {}
    for activation in ['elu','gelu','tanh','relu']:
        for bn in [True,False]:
            for kernel_size in [2,4]:
                for l2 in [0.0,1e-6]:
                    for sd in [0.0,0.05,0.10]:
                        for pt in ['max','avg']:
                            print(exp_count,activation,bn,kernel_size, l2, sd)
                            exp_count+=1
                            hyper_dict.update({exp_count:{'activation':activation,'bn':bn,'kernel_size':kernel_size,'l2':l2,'sd':sd,'pt':pt}})
    pickle.dump(hyper_dict,open('hyper_dict3.pkl','wb'))

if __name__=='__main__':
    build_exp_dict()