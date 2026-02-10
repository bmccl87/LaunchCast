import tensorflow as tf
import keras
from keras.layers import SpatialDropout2D, SpatialDropout3D, AveragePooling3D, Dense, Concatenate, Masking, Conv2D, Conv3D, UpSampling2D, ConvLSTM2D, MaxPooling2D, MaxPooling3D, UpSampling3D, ConvLSTM1D, LSTM, Reshape, Conv1D, Lambda, MultiHeadAttention, LayerNormalization, Flatten
from keras.models import Model
import numpy as np
from tensorflow.keras.utils import plot_model
from LC_parser import *
import math

class Stack(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.stack(inputs, axis=self.axis)

def next_power_of_two(n: int) -> int:
    if n < 1:
        raise ValueError("Input must be a positive integer.")
    return 1 if n == 1 else 2 ** math.ceil(math.log2(n))

def act_function(tensor,conv_activation,name):
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

def build_efm_dist_encoder(args):
    tf.print('building the efm distance encoder')
    dist_efm_keys = args.efm_dist_keys
    input_shape = args.efm_dist_input_shape
    L2_reg = tf.keras.regularizers.l2(args.L2_reg)
    conv_activation=args.activation_conv
    pool_size=args.pool_size

    dist_input_tensor_list = []
    for dist_key in dist_efm_keys:
        tf.print(dist_key,'building the model')
        dist_input_tensor=tf.keras.Input(shape=input_shape,
                                    dtype=tf.dtypes.float32,
                                    name=dist_key)
        dist_input_tensor_list.append(dist_input_tensor)
    dist_stacked_tensor = Stack(axis=-1,name='Stacked_dist_inputs')(dist_input_tensor_list)
    dist_tensor = dist_stacked_tensor
    tf.print(dist_stacked_tensor.shape)
    tf.print('efm_dist tensor stacked correctly')
    filters = args.efm_dist_encoder_filters
    for d in range(args.efm_dist_encoder_deep):
        dist_tensor = Conv2D(filters=filters[d],
                        kernel_size=(args.kernel_size,args.kernel_size),
                        strides=args.strides,
                        padding=args.padding,
                        data_format='channels_last',
                        activation=None,
                        use_bias=True,
                        kernel_regularizer=L2_reg,
                        name='efm_distance_encoder_%s'%d)(dist_tensor)
        dist_tensor = act_function(tensor=dist_tensor,conv_activation=conv_activation,name='efm_distance_encoder_%s'%d)
        dist_tensor = MaxPooling2D(pool_size=(pool_size,pool_size),name='MaxPooling2D_efm_distance_%s'%d)(dist_tensor)
    dist_tensor = Flatten(name='EFM_distance_latent')(dist_tensor)
    tf.print("tensor.shape",dist_tensor.shape)
    return dist_input_tensor_list, dist_tensor

def build_efm_ts_encoder(args):
    tf.print('building a new efm encoder without attention')
    ts_input_shape=args.efm_ts_input_shape
    ts_FM_keys = args.efm_ts_keys
    L2_reg = tf.keras.regularizers.l2(args.L2_reg)
    conv_activation=args.activation_conv
    ts_input_lstm_list=[]
    input_h_list=[]
    input_c_list=[]
    ts_input_tensor_list=[]
    shared_LSTM = LSTM(units=16,
                        return_state=True,
                        return_sequences=False,
                        kernel_regularizer=L2_reg)

    for ts_key in ts_FM_keys:
        tf.print(ts_key,'building the layers')
        ts_input_tensor=tf.keras.Input(shape=ts_input_shape,
                                    dtype=tf.dtypes.float32,
                                    name=ts_key)
        ts_input_tensor_list.append(ts_input_tensor)
        input_lstm, input_h, input_c = shared_LSTM(ts_input_tensor)
        input_c_list.append(input_c)

    tf.print('stacking time series inputs')
    ts_stacked_tensor = Stack(axis=1,name='EFM_TS_inputs')(input_c_list)
    ts_tensor = ts_stacked_tensor
    tf.print('time series stacked successfully')
    filters = args.efm_ts_encoder_filters
    for d in range(args.efm_ts_encoder_deep):
        ts_tensor = Conv1D(filters=filters[d],
                        kernel_size=args.kernel_size,
                        strides=args.strides,
                        padding=args.padding,
                        data_format='channels_last',
                        activation=None,
                        use_bias=True,
                        kernel_regularizer=L2_reg,
                        name='efm_ts_encoder_Conv1D_%s'%d)(ts_tensor)
        ts_tensor = act_function(tensor=ts_tensor,conv_activation=conv_activation,name='efm_ts_encoder_activation_%s'%d)
    ts_tensor = Flatten(name='Latent_EFM_TS')(ts_tensor)
    return ts_input_tensor_list,ts_tensor
    
def build_the_decoder(args,tensor,skip_list,channels_list):
    conv_activation = args.activation_conv
    last_activation = args.activation_last
    drop_rate = args.spatial_dropout
    L2_reg = tf.keras.regularizers.l2(args.L2_reg)
    batch_norm = args.batch_normalization
    padding = args.padding
    kernel_2d = args.kernel_size
    pool_type = args.pool_type
    pool_size = args.pool_size
    encoder_deep = args.encoder_deep
    num_per_block = args.num_layers_per_block
    skip = args.skip
    rotate = args.rotate
    noise = args.noise
    noise_std = args.noise_std
    rotate = args.rotate
    input_shape = tensor.shape
    output_shape = args.output_image_size
    kernel_time = tensor.shape[1]

    up_samp_size = (2,pool_size,pool_size)
    tensor = UpSampling3D(size=up_samp_size,name='UpSample3D_bottom_size_%s_%s_%s'%(up_samp_size[0],up_samp_size[1],up_samp_size[2]))(tensor)
    decoder_deep=encoder_deep

    decoder_channels = np.flip(channels_list)[1:]
    print(decoder_channels)
    skip_idx = len(skip_list)-2
    for d in range(decoder_deep):
        num_channels = decoder_channels[d]            
        tf.print('decoder:',d,'kernel_time',kernel_time,'kernel_2d',kernel_2d)

        for n in range(num_per_block):
            kernel_time = tensor.shape[1]
            tensor = Conv3D(input_shape=tensor.shape,
                    filters=num_channels,
                    kernel_size=(int(kernel_time),int(kernel_2d),int(kernel_2d)),
                    padding=padding,
                    kernel_regularizer=L2_reg,
                    data_format='channels_last',
                    dtype=tf.dtypes.float32,
                    use_bias=True,
                    name='decoder_conv3d_%s_%s_2d_kernel_%s_time_kernel_%s'%(d,n,kernel_2d,kernel_time))(tensor)
            if batch_norm==True:
                tensor = tf.keras.layers.BatchNormalization(axis=-1,name='decoder_BN_%s_%s'%(d,n))(tensor)
            tensor = act_function(tensor=tensor,conv_activation=conv_activation,name='decoder_activation_%s_%s_%s'%(conv_activation,d,n))
        tensor = SpatialDropout3D(rate=drop_rate,data_format='channels_last',name='decoder_SD_%s_%s_drop_rate_%s'%(d,n,drop_rate))(tensor)
        if tensor.shape[2]<8:
            up_samp_size = (2,pool_size,pool_size)
        else:
            up_samp_size = (1,pool_size,pool_size)
        tensor = UpSampling3D(size=up_samp_size,name='decoder_UpSample3D_%s_size_%s_%s_%s'%(d,up_samp_size[0],up_samp_size[1],up_samp_size[2]))(tensor)

    tensor = Conv3D(input_shape=tensor.shape,
                    filters=64,
                    kernel_size=(int(kernel_time),int(kernel_2d),int(kernel_2d)),
                    padding=padding,
                    kernel_regularizer=L2_reg,
                    data_format='channels_last',
                    dtype=tf.dtypes.float32,
                    use_bias=True,
                    name='output_block_64')(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation,name='output_block_activation_%s_%s'%(conv_activation,64))

    tensor = Conv3D(input_shape=tensor.shape,
                    filters=32,
                    kernel_size=(int(kernel_time),int(kernel_2d),int(kernel_2d)),
                    padding=padding,
                    kernel_regularizer=L2_reg,
                    data_format='channels_last',
                    dtype=tf.dtypes.float32,
                    use_bias=True,
                    name='output_block_32')(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation,name='output_block_activation_%s_%s'%(conv_activation,32))
    
    tensor = Conv3D(input_shape=tensor.shape,
                    filters=16,
                    kernel_size=(int(kernel_time),int(kernel_2d),int(kernel_2d)),
                    padding=padding,
                    kernel_regularizer=L2_reg,
                    data_format='channels_last',
                    dtype=tf.dtypes.float32,
                    use_bias=True,
                    name='output_block_16')(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation,name='output_block_activation_%s_%s'%(conv_activation,16))

    tensor = Conv3D(input_shape=tensor.shape,
                    filters=8,
                    kernel_size=(int(kernel_time),int(kernel_2d),int(kernel_2d)),
                    padding=padding,
                    kernel_regularizer=L2_reg,
                    data_format='channels_last',
                    dtype=tf.dtypes.float32,
                    use_bias=True,
                    name='output_block_8')(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation,name='output_block_activation_%s_%s'%(conv_activation,8))

    kernel_time = tensor.shape[1]
    tensor = Conv3D(input_shape=output_shape,
                    filters=output_shape[-1],
                    kernel_size=(int(kernel_time),int(kernel_2d),int(kernel_2d)),
                    padding=padding,
                    kernel_regularizer=L2_reg,
                    data_format='channels_last',
                    dtype=tf.dtypes.float32,
                    use_bias=True,
                    activation=last_activation,
                    name='output_tensor')(tensor)
    return tensor



def build_gridded_inputs(args):
    glm_inputs=[]
    glm_features=[]
    if args.use_GLM==True:
        keys = args.GLM_keys
        input_shape = args.GLM_input_shape
        for key in keys:
            glm_input_all_times = []
            for t in ['t0','t1','t2','t3']:
                glm_input=tf.keras.Input(shape=input_shape,
                                        dtype=tf.dtypes.float32,
                                        name='GLM_%s_%s'%(key,t))
                glm_inputs.append(glm_input)
                if args.rotate==True:
                    glm_input=tf.keras.layers.RandomRotation(factor=(-args.rotate_dec,args.rotate_dec),fill_mode='reflect')(glm_input)
                #insert other augmentation techniques
                glm_input = tf.expand_dims(glm_input,axis=1)
                glm_input_all_times.append(glm_input)
            one_glm_feature=Concatenate(axis=1,name='Concatenate_GLM_%s_times'%key)(glm_input_all_times)
            glm_features.append(one_glm_feature)
        glm_tensor = Concatenate(axis=-1,name='Concatenate_all_GLM_features_times')(glm_features)
        print(glm_tensor)
    print(glm_inputs)

    mrms_inputs = []
    input_shape = args.MRMS_input_shape
    mrms_features=[]
    if args.use_MRMS:
        if args.use_VI==True:
            keys=args.VI_keys
            for key in keys:
                mrms_input_all_times=[]
                for t in ['t0','t1','t2','t3']:
                    mrms_input = tf.keras.Input(shape=input_shape,
                                        dtype=tf.dtypes.float32,
                                        name='MRMS_%s_%s'%(key,t))
                    mrms_inputs.append(mrms_input)
                    if args.rotate==True:
                        mrms_input=tf.keras.layers.RandomRotation(factor=(-args.rotate_dec,args.rotate_dec),fill_mode='reflect')(mrms_input)
                    mrms_input=tf.expand_dims(mrms_input,axis=1)
                    mrms_input_all_times.append(mrms_input)
                mrms_feature=Concatenate(axis=1,name='MRMS_%s_all_times'%key)(mrms_input_all_times)
                mrms_features.append(mrms_feature)
        
        if args.use_Z==True:
            keys=args.Z_keys
            for key in keys:
                mrms_input_all_times=[]
                for t in ['t0','t1','t2','t3']:
                    mrms_input = tf.keras.Input(shape=input_shape,
                                        dtype=tf.dtypes.float32,
                                        name='MRMS_%s_%s'%(key,t))
                    mrms_inputs.append(mrms_input)
                    if args.rotate==True:
                        mrms_input=tf.keras.layers.RandomRotation(factor=(-args.rotate_dec,args.rotate_dec),fill_mode='reflect')(mrms_input)
                    mrms_input=tf.expand_dims(mrms_input,axis=1)
                    mrms_input_all_times.append(mrms_input)
                mrms_feature=Concatenate(axis=1,name='MRMS_%s_all_times'%key)(mrms_input_all_times)
                mrms_features.append(mrms_feature)

        if args.use_Zdr==True:
            keys=args.Zdr_keys
            for key in keys:
                mrms_input_all_times=[]
                for t in ['t0','t1','t2','t3']:
                    mrms_input = tf.keras.Input(shape=input_shape,
                                        dtype=tf.dtypes.float32,
                                        name='MRMS_%s_%s'%(key,t))
                    mrms_inputs.append(mrms_input)
                    if args.rotate==True:
                        mrms_input = tf.keras.layers.RandomRotation(factor=(-args.rotate_dec,args.rotate_dec),fill_mode='reflect')(mrms_input)
                    mrms_input=tf.expand_dims(mrms_input,axis=1)
                    mrms_input_all_times.append(mrms_input)
                mrms_feature = Concatenate(axis=1,name='MRMS_%s_all_times'%key)(mrms_input_all_times)
                mrms_features.append(mrms_feature)
    mrms_tensor = Concatenate(axis=-1,name='All_MRMS_features_times')(mrms_features)
    print(mrms_inputs)
    print(mrms_tensor)
    all_gridded_inputs=glm_inputs+mrms_inputs
    gridded_tensor=Concatenate(axis=-1,name='All_Gridded_Data')([glm_tensor,mrms_tensor])
    return all_gridded_inputs, gridded_tensor

def build_gridded_encoder_lstm(args,tensor):
    num_filters=32
    conv_size=args.kernel_size
    kernel_size=(args.kernel_size,args.kernel_size)
    padding=args.padding
    L2=args.L2_reg
    conv_activation=args.activation_conv

    tensor, h, c = ConvLSTM2D(filters=num_filters, 
                                kernel_size=kernel_size, 
                                padding=padding, 
                                return_sequences=False,
                                return_state=True,
                                kernel_regularizer=tf.keras.regularizers.l2(L2),
                                input_shape = tensor.shape,
                                name='convlstm2d_kernel_2d_%s'%(conv_size))(tensor)
    print('c state shape',c.shape)

    pool_size=(4,4)
    for d in range(4):
        num_filters=num_filters*2
        tensor=Conv2D(filters=num_filters,
                kernel_size=(args.kernel_size,args.kernel_size),
                strides=args.strides,
                padding=args.padding,
                data_format='channels_last',
                activation=None,
                use_bias=True,
                kernel_regularizer=tf.keras.regularizers.l2(L2),
                input_shape=tensor.shape,
                name='gridded_c_state_encoder_%s'%d)(tensor)
        tensor = act_function(tensor=tensor,conv_activation=conv_activation,name='c_state_encoder_%s'%d)
        print(d,tensor.shape)
        if d<=2:
            tensor = MaxPooling2D(pool_size=pool_size,name='MaxPooling2D_c_state_encoder_%s'%d)(tensor)
    tensor=Flatten()(tensor)
    return tensor

def build_model(args):

    if args.use_MRMS==True or args.use_GLM==True:
        all_gridded_inputs,gridded_tensor = build_gridded_inputs(args=args)
        gridded_latent_tensor = build_gridded_encoder_lstm(args=args,tensor=gridded_tensor)
        print('gridded_latent_tensor',gridded_latent_tensor.shape)

    if args.use_EFM==True:
        efm_ts_inputs, efm_ts_latent_tensor = build_efm_ts_encoder(args=args)#flattened
        efm_dist_inputs, efm_dist_latent_tensor = build_efm_dist_encoder(args=args)#flattened

    #build the dense layers with all of the EFM data
    if args.use_EFM==True:
        all_flat_tensor = Concatenate(axis=-1,name='Latent_EFM_Tensor')([gridded_latent_tensor,efm_ts_latent_tensor,efm_dist_latent_tensor])
        tf.print('all_flat_tensor.shape:', all_flat_tensor.shape)
        
        dense_units = int(next_power_of_two(n=int(all_flat_tensor.shape[1])))
        print(dense_units)
        num_dense_layers=0
        while dense_units>=2048:
            all_flat_tensor=Dense(units=dense_units,activation=None,name='Dense_Layer_%s'%(num_dense_layers))(all_flat_tensor)
            all_flat_tensor=act_function(tensor=all_flat_tensor,conv_activation=args.activation_conv,name='Dense_activation_%s_Layer_%s'%(args.activation_conv,num_dense_layers))
            dense_units=int(dense_units/2)
            print(num_dense_layers,dense_units)
            num_dense_layers+=1
        into_decoder_tensor=Reshape((1,2,2,512))(all_flat_tensor)

    channels_list = args.efm_dist_encoder_filters
    output_tensor=build_the_decoder(args=args, tensor=into_decoder_tensor,skip_list=[],channels_list=channels_list)
    model = Model(inputs=efm_ts_inputs+efm_dist_inputs+all_gridded_inputs,outputs=output_tensor)
    return model

if __name__=='__main__':

    parser = create_parser()
    args = parser.parse_args()
    model=build_model(args)

    # model = build_efm_only_model(args)
    print(model.summary())
    # render_fname='LC_model_arch_efm_dist_only.png'
    # plot_model(model,to_file=render_fname, show_shapes=True, show_layer_names=True)
    
    
