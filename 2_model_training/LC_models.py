import tensorflow as tf
import keras
from keras.layers import SpatialDropout2D, SpatialDropout3D, AveragePooling3D, Dense, Concatenate, Masking, Conv2D, Conv3D, UpSampling2D, ConvLSTM2D, MaxPooling2D, MaxPooling3D, UpSampling3D, ConvLSTM1D, LSTM, Reshape, Conv1D 
from keras.models import Model
import numpy as np
from tensorflow.keras.utils import plot_model
from LC_parser import *
import math

def act_function(tensor,conv_activation):

    if conv_activation=='tanh':
        tensor = tf.keras.activations.tanh(tensor)
    if conv_activation=='relu':
        tensor = tf.keras.activations.relu(tensor)
    if conv_activation=='leaky_relu':
        tensor = tf.keras.activations.leaky_relu(tensor)
    if conv_activation=='elu':
        tensor = tf.keras.activations.elu(tensor)
    if conv_activation=='gelu':
        tensor = tf.keras.activations.gelu(tensor)
    return tensor

def build_conv3d(args):

    conv_activation = args.activation_conv
    last_activation = args.activation_last
    drop_rate = args.spatial_dropout
    L2_reg = tf.keras.regularizers.l2(args.L2_reg)
    batch_norm = args.batch_normalization
    padding = args.padding
    kernel_2d = args.kernel_shape
    pool_type = args.pool_type

    input_shape = args.hrrr_image_size
    input_tensor = tf.keras.Input(shape=input_shape,dtype=tf.dtypes.float32,name='hrrr_input')
    tensor = input_tensor
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)

    kernel_size = (4,kernel_2d,kernel_2d)
    tensor = Conv3D(input_shape=tensor.shape,
                            filters=32,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='conv3d_1_0')(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    tensor = Conv3D(input_shape=tensor.shape,
                            filters=32,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='conv3d_1_1')(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    tensor = Conv3D(input_shape=tensor.shape,
                            filters=32,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='conv3d_1_2')(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)
    tensor = SpatialDropout3D(rate=drop_rate,data_format='channels_last')(tensor)
    skip1 = tensor

    if pool_type=='avg':
        tensor = AveragePooling3D(pool_size=(2,2,2),name='AvgPooling3D_1')(tensor)#2x8x8
    else:
        tensor = MaxPooling3D(pool_size=(2,2,2),name='MaxPooling3D_1')(tensor)#2x8x8

    kernel_size = (2,kernel_2d,kernel_2d)
    tensor = Conv3D(input_shape=tensor.shape,
                            filters=64,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='conv3d_2_0')(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    tensor = Conv3D(input_shape=tensor.shape,
                            filters=64,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='conv3d_2_1')(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    tensor = Conv3D(input_shape=tensor.shape,
                            filters=64,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='conv3d_2_2')(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    tensor = SpatialDropout3D(rate=drop_rate,data_format='channels_last')(tensor)
    skip2 = tensor
    if args.pool_type=='avg':
        tensor = AveragePooling3D(pool_size=(2,2,2),name='AveragePooling3D_2')(tensor)#1x4x4
    else:
        tensor = MaxPooling3D(pool_size=(2,2,2),name='MaxPooling3D_2')(tensor)#1x4x4
    
    kernel_size=(1,kernel_2d,kernel_2d)
    tensor = Conv3D(input_shape=tensor.shape,
                            filters=128,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='conv3d_bottom_1')(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    tensor = Conv3D(input_shape=tensor.shape,
                            filters=128,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='conv3d_bottom_2')(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    tensor = Conv3D(input_shape=tensor.shape,
                            filters=128,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='conv3d_bottom_3')(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    tensor = SpatialDropout3D(rate=drop_rate,data_format='channels_last')(tensor)
    tensor = UpSampling3D(size=(2,2,2),name='Up_1')(tensor)#2x8x8
    tensor = Concatenate(axis=-1,name='Skip2')([skip2,tensor])

    kernel_size=(2,kernel_2d,kernel_2d)
    tensor = Conv3D(input_shape=tensor.shape,
                            filters=64,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='conv3d_3_0')(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    tensor = Conv3D(input_shape=tensor.shape,
                            filters=64,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='conv3d_3_1')(tensor)

    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    tensor = Conv3D(input_shape=tensor.shape,
                            filters=64,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='conv3d_3_2')(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    tensor = SpatialDropout3D(rate=drop_rate,data_format='channels_last')(tensor)
    tensor = UpSampling3D(size=(2,2,2),name='Up_2')(tensor)#4x16x16
    tensor = Concatenate(axis=-1,name='Skip1')([tensor,skip1])

    kernel_size = (4,kernel_2d,kernel_2d)
    tensor = Conv3D(input_shape=tensor.shape,
                            filters=32,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='conv3d_6_0')(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    tensor = Conv3D(input_shape=tensor.shape,
                            filters=32,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='conv3d_6_1')(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    tensor = Conv3D(input_shape=tensor.shape,
                            filters=32,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='conv3d_6_2')(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)
    tensor = SpatialDropout3D(rate=drop_rate,data_format='channels_last')(tensor)

    tensor = Conv3D(input_shape=tensor.shape,
                            filters=16,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='output_block')(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    kernel_size=(2,kernel_2d,kernel_2d)
    tensor = Conv3D(input_shape=tensor.shape,
                            filters=2,
                            kernel_size=kernel_size,
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            activation=last_activation,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='output_tensor')(tensor)
    output_tensor = tensor
    model = Model(inputs=input_tensor,outputs=output_tensor)
    return model

def next_power_of_two(n: int) -> int:
    if n < 1:
        raise ValueError("Input must be a positive integer.")
    return 1 if n == 1 else 2 ** math.ceil(math.log2(n))

def build_conv2d(args):

    print('building the conv2d model')
    conv_activation = args.activation_conv
    last_activation = args.activation_last
    drop_rate = args.spatial_dropout
    L2_reg = tf.keras.regularizers.l2(args.L2_reg)
    batch_norm = args.batch_normalization
    padding = args.padding
    kernel_2d = args.kernel_shape
    pool_type = args.pool_type
    pool_size = args.pool_size
    encoder_deep = args.encoder_deep
    num_per_block = args.num_layers_per_block

    input_shape = args.hrrr_image_size
    input_shape = [64,64,8]
    num_channels = next_power_of_two(n=input_shape[-1])
    input_tensor = tf.keras.Input(shape=input_shape,dtype=tf.dtypes.float32,name='hrrr_input')
    tensor = input_tensor

    skip_list = []
    channels_list = []
    #build the encoder
    for d in range(encoder_deep):
        channels_list.append(num_channels)
        for n in range(num_per_block):
            tensor = Conv2D(input_shape=tensor.shape,
                            filters=num_channels,
                            kernel_size=(kernel_2d,kernel_2d),
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            name='encoder_conv2d_%s_%s'%(d,n))(tensor)
            if batch_norm==True:
                tensor = tf.keras.layers.BatchNormalization(axis=-1,name='encoder_BN_%s_%s'%(d,n))(tensor)
            tensor = act_function(tensor=tensor,conv_activation=conv_activation)
        skip_list.append(tensor)
        tensor = SpatialDropout2D(rate=drop_rate,data_format='channels_last')(tensor)
        if pool_type=='max':
            tensor = MaxPooling2D(pool_size=(pool_size,pool_size),name='encoder_MaxPool2D_%s'%d)(tensor)
        if pool_type=='avg':
            tensor = AvgPooling2D(pool_size=(pool_size,pool_size),name='encoder_AvgPool2D_%s'%d)(tensor)
        num_channels=num_channels*2


    #build the bottom block
    for n in range(num_per_block):
        tensor = Conv2D(input_shape=tensor.shape,
                        filters=num_channels,
                        kernel_size=(kernel_2d,kernel_2d),
                        padding=padding,
                        kernel_regularizer=L2_reg,
                        data_format='channels_last',
                        dtype=tf.dtypes.float32,
                        name='conv2d_bottom_%s'%(n))(tensor)
        if batch_norm==True:
            tensor = tf.keras.layers.BatchNormalization(axis=-1,name='BN_bottom_%s'%(n))(tensor)
        tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    tensor = SpatialDropout2D(rate=drop_rate,data_format='channels_last')(tensor)
    tensor = UpSampling2D(size=(pool_size,pool_size),name='US2D_bottom_0')(tensor)

    #build the decoder (two less than the encoder)
    channels_list = np.flip(channels_list)
    for d in range(encoder_deep-2):
        num_channels = channels_list[d]
        for n in range(num_per_block):
            tensor = Conv2D(input_shape=tensor.shape,
                        filters=num_channels,
                        kernel_size=(kernel_2d,kernel_2d),
                        padding=padding,
                        kernel_regularizer=L2_reg,
                        data_format='channels_last',
                        dtype=tf.dtypes.float32,
                        name='conv2d_decoder_%s_%s'%(d,n))(tensor)
            if batch_norm==True:
                tensor = tf.keras.layers.BatchNormalization(axis=-1,name='BN_bottom_%s_%s'%(d,n))(tensor)
            tensor = act_function(tensor=tensor,conv_activation=conv_activation)

        tensor = SpatialDropout2D(rate=drop_rate,data_format='channels_last')(tensor)
        if d<(encoder_deep-3):
            tensor = UpSampling2D(size=(pool_size,pool_size),name='decoder_US2D_%s'%d)(tensor)
    
    tensor = Conv2D(input_shape=input_shape,
                        filters=num_channels/2,
                        kernel_size=(kernel_2d,kernel_2d),
                        padding=padding,
                        kernel_regularizer=L2_reg,
                        data_format='channels_last',
                        dtype=tf.dtypes.float32,
                        name='conv2d_out_1')(tensor)
    if batch_norm==True:
                tensor = tf.keras.layers.BatchNormalization(axis=-1,name='BN_out_1')(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation)

    output_tensor = Conv2D(input_shape=input_shape,
                        filters=8,#CC/CG lightning for 4 different times
                        kernel_size=(kernel_2d,kernel_2d),
                        padding=padding,
                        kernel_regularizer=L2_reg,
                        data_format='channels_last',
                        dtype=tf.dtypes.float32,
                        activation=last_activation,
                        name='conv2d_out_final')(tensor)
    model = Model(inputs=input_tensor,outputs=output_tensor)
    return model

if __name__=='__main__':

    parser = create_parser()
    args = parser.parse_args([])
    model = build_conv2d(args=args)
    plot_model(model,to_file='Conv2d_y2y.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
