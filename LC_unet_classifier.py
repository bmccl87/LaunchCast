import argparse
import numpy as np
import tensorflow as tf
import keras
from keras.layers import InputLayer, Dense, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model

from LC_parser import *

def create_unet(image_size=[64,84,1],
                hrrr=True,
                n_hrrr_params=5,
                ksc_wx_twr=False,
                n_ksc_twr_params=4,
                ksc_efm = False,
                filters=[8,16,32],
                conv_size=2,
                pool_size=2,
                deep=4,
                n_conv_per_step=2,
                lrate=.0001,
                n_types=2,
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=tf.keras.metrics.SparseCategoricalAccuracy(),
                padding='same',
                strides=1,
                conv_activation='elu',
                last_activation='linear',
                batch_normalization=False,
                dropout=0.0,
                skip=False): 

    '''
    This functions builds a U-Net for LaunchCast.  It builds down the number of layers 
    determined by the deep parameter.  It performs MaxPooling after each convolution block 
    described by the n_conv_step parameter.  Activation functions throughout the U are 
    provided, along with the activation function for the last layer.  Strides, padding, 
    loss functions, and metrics are provided too.  The filters is a list of the number 
    of filters to use, for every convolution down.  The filters are flipped along the way 
    up.  Flags/booleans for KSC data sets and regularization
    are also provided.  
    '''
    
    #build the input layers for each hrrr sfc parameter
    hrrr_u = tf.keras.Input(shape=(image_size[0],image_size[1],1),
                            dtype=tf.dtypes.float64,
                            name='hrrr_sfc_u')

    hrrr_v = tf.keras.Input(shape=(image_size[0],image_size[1],1),
                            dtype=tf.dtypes.float64,
                            name='hrrr_sfc_v')

    hrrr_temp = tf.keras.Input(shape=(image_size[0],image_size[1],1),
                            dtype=tf.dtypes.float64,
                            name='hrrr_sfc_temp')

    hrrr_moist = tf.keras.Input(shape=(image_size[0],image_size[1],1),
                                dtype=tf.dtypes.float64,
                                name='hrrr_sfc_moist')

    hrrr_sfc_pres = tf.keras.Input(shape=(image_size[0],image_size[1],1),
                                    dtype=tf.dtypes.float64,
                                    name='hrrr_sfc_pres')


    #concatenate the layers before doing the convolutions
    input_tensor=tf.concat([hrrr_u, hrrr_v, hrrr_temp, hrrr_moist, hrrr_sfc_pres],axis=3)
    tensor = input_tensor

    #go down the U-Net
    for i,f in enumerate(filters):
        
        #build the convolution layer
        tensor = Conv2D(filters=f,padding=padding,strides=1,kernel_size=(conv_size,conv_size),use_bias=True,activation=conv_activation, name='Down_Conv_f'+str(f)+'_'+conv_activation)(tensor)

        #conduct pooling
        tensor = MaxPooling2D(pool_size=(pool_size,pool_size),strides=(pool_size,pool_size),name='Down_Pool_'+str(i))(tensor)
    
    #learn at the bottom
    tensor = Conv2D(filters=f,padding=padding,strides=1,kernel_size=(conv_size,conv_size),use_bias=True,activation=conv_activation, name='Bottom_Conv_f'+str(f)+'_'+conv_activation)(tensor)

    #flip the filters to build back up the U
    filters = np.flip(filters)
    print(filters)

    for i,f in enumerate(filters):

        #build the convolution layer
        tensor = Conv2D(filters=f,padding=padding,strides=1,kernel_size=(conv_size,conv_size),use_bias=True,activation=conv_activation,name='Up_Conv_f'+str(f)+'_'+conv_activation)(tensor)

        #upsample to higher resolution
        tensor = UpSampling2D(size=(pool_size,pool_size),name='Up_UpSample_'+str(i))(tensor)
    
    #one last learning block
    tensor = Conv2D(filters=f,padding=padding,strides=1,kernel_size=(conv_size,conv_size),use_bias=True,activation=conv_activation,name='Top_Conv_f'+str(f)+'_'+conv_activation)(tensor)

    #build the output layer. Use softmax if you want to predict the probability 
    #of CC and CG lightning.  Use linear or something greater than 0 to predict the 
    #amount of CC and CG lightning. 
    output_tensor = Conv2D(filters=n_types,padding=padding,strides=1,kernel_size=(conv_size,conv_size),use_bias=True,name='MERLIN_CC_or_CG_'+last_activation,activation=last_activation)(tensor)

    #compile the model 
    model = Model(inputs=[hrrr_u, hrrr_v, hrrr_temp, hrrr_moist, hrrr_sfc_pres],outputs=output_tensor)
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
    model.compile(optimizer=opt,loss=loss)

    return model

if __name__ == "__main__":

    print('LC_unet_classifier.py main function')
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    print(args)

    image_size=[64,64,1]

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
                        lrate=args.lrate,
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
        render_fname = 'LC_model_test.png'
        plot_model(model, to_file=render_fname, show_shapes=True, show_layer_names=True)

    


