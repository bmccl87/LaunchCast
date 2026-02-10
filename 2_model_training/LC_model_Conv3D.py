import tensorflow as tf
import keras
from keras.layers import SpatialDropout2D, SpatialDropout3D, AveragePooling3D 
from keras.layers import Dense, Concatenate, Masking, Conv2D, Conv3D, UpSampling2D
from keras.layers import ConvLSTM2D, MaxPooling2D, MaxPooling3D, UpSampling3D
from keras.layers import ConvLSTM1D, LSTM, Reshape, Conv1D, Lambda
from keras.layers import GlobalMaxPool3D, Dropout, Flatten
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
    pool_size=4

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
        print('encoding the distances,',d,dist_tensor.shape)
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

def build_conv3d_rotate(args):
    gridded_inputs, gridded_obs_tensor, nwp_tensor = build_gridded_inputs(args=args)
    nwp_latent, nwp_skip_list = build_encoder_v2(args=args,tensor=nwp_tensor,data_type='NWP')
    nwp_latent = Reshape((nwp_latent.shape[-1],))(nwp_latent)

    obs_latent, obs_skip_list = build_encoder_v2(args=args,tensor=gridded_obs_tensor,data_type='Obs')
    obs_latent = Reshape((obs_latent.shape[-1],))(obs_latent)

    efm_dist_inputs, efm_dist_latent = build_efm_dist_encoder(args=args)
    ts_input_tensor_list,ts_tensor = build_efm_ts_encoder(args=args)

    tensor = Concatenate(axis=-1)([nwp_latent,obs_latent,efm_dist_latent,ts_tensor])

    tensor = build_latent_v3(args=args,tensor=tensor)
    
    tensor = build_decoder_v2(args=args,
                                nwp_skip_list=nwp_skip_list,
                                obs_skip_list=obs_skip_list,
                                tensor=tensor)
    output_tensor = tensor
    model = Model(inputs=gridded_inputs+efm_dist_inputs+ts_input_tensor_list,outputs=output_tensor)
    return model

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

    hrrr_inputs = []
    input_shape = args.HRRR_input_shape
    hrrr_feature_tensors=[]
    if args.use_HRRR==True:
        keys=args.hrrr_features
        for key in keys:
            hrrr_input_all_times=[]
            for t in ['t0','t1','t2','t3']:
                hrrr_input = tf.keras.Input(shape=input_shape,
                                    dtype=tf.dtypes.float32,
                                    name='HRRR_%s_%s'%(key,t))
                hrrr_inputs.append(hrrr_input)
                if args.rotate==True:
                    hrrr_input=tf.keras.layers.RandomRotation(factor=(-args.rotate_dec,args.rotate_dec),fill_mode='reflect')(hrrr_input)
                hrrr_input=tf.expand_dims(hrrr_input,axis=1)
                hrrr_input_all_times.append(hrrr_input)
            hrrr_feature=Concatenate(axis=1,name='HRRR_%s_all_times'%key)(hrrr_input_all_times)
            hrrr_feature_tensors.append(hrrr_feature)
    nwp_tensor = Concatenate(axis=-1,name='All_HRRR_features_times')(hrrr_feature_tensors)
    print(nwp_tensor)
    
    all_gridded_inputs=glm_inputs + mrms_inputs + hrrr_inputs
    gridded_obs_tensor=Concatenate(axis=-1,name='All_Gridded_Obs_Data')([glm_tensor,mrms_tensor])
    
    return all_gridded_inputs, gridded_obs_tensor, nwp_tensor

def build_encoder_v2(args,tensor,data_type):
    
    conv_activation = args.activation_conv
    last_activation = args.activation_last
    drop_rate = args.spatial_dropout
    L2_reg = tf.keras.regularizers.l2(args.L2_reg)
    batch_norm = args.batch_normalization
    padding = args.padding
    
    kernel_time = 4
    pool_type = args.pool_type
    pool_size = args.pool_size
    encoder_deep = args.encoder_deep+2
    num_per_block = args.num_layers_per_block
    skip = args.skip
    rotate = args.rotate
    noise = args.noise
    noise_std = args.noise_std
    rotate = args.rotate
    output_shape = args.output_image_size

    if noise==True:
        tensor = tf.keras.layers.GaussianNoise(stddev=args.noise_std, seed=None)(tensor)

    skip_list = []
    channels_list = []
    tkl = []

    num_channels = next_power_of_two(n=tensor.shape[-1])
    for d in range(encoder_deep):
        print(d,num_channels)
        print(tensor.shape)
        channels_list.append(num_channels)
        #adjust the spatial kernel sizes based on the lat/lon shape
        if tensor.shape[2]==64:
            kernel_2d = 8
        elif tensor.shape[2]==32:
            kernel_2d = 8
        elif tensor.shape[2]==16:
            kernel_2d = 8
        elif tensor.shape[2]==8:
            kernel_2d = 4
        else:
            kernel_2d = 2

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
                            name='%s_encoder_conv3d_%s_%s_2d_kernel_%s_time_kernel_%s'%(data_type,d,n,kernel_2d,kernel_time))(tensor)
            if batch_norm==True:
                tensor = tf.keras.layers.BatchNormalization(axis=-1,name='%s_encoder_BN_%s_%s'%(data_type,d,n))(tensor)
            tensor = act_function(tensor=tensor,conv_activation=conv_activation,name='%s_encoder_activation_%s_%s_%s'%(data_type,conv_activation,d,n))
        
        #add the tensor to skip list after the convolutional block
        skip_list.append(tensor)

        #drop out after the convolutional block
        tensor = SpatialDropout3D(rate=drop_rate,data_format='channels_last',name='%s_encoder_SD_%s_drop_rate_%s'%(data_type,d,drop_rate))(tensor)
        if tensor.shape[2]<=16:
            pool_size_deep = (1,pool_size,pool_size)
        else:
            pool_size_deep = (2,pool_size,pool_size)
        pool_str = '%s_%s_%s'%(pool_size_deep[0],pool_size_deep[1],pool_size_deep[2])

        #pool max or average based on the flags
        if pool_type=='max':
            tensor = MaxPooling3D(pool_size=pool_size_deep,name='%s_encoder_MaxPool3D_%s_pool_size_%s'%(data_type,d,pool_str))(tensor)
        if pool_type=='avg':
            tensor = AvgPooling3D(pool_size=pool_size_deep,name='%s_encoder_AvgPool3D_%s_pool_size_%s'%(data_type,d,pool_str))(tensor)
        
        #adjust the channel number 
        num_channels=num_channels*2
        if tensor.shape[-1]==1:
            return tensor, skip_list
    return tensor, skip_list

def build_latent_v2(args,tensor):
    num_per_block = args.num_layers_per_block
    kernel_2d = 2
    padding=args.padding
    conv_activation = args.activation_conv
    last_activation = args.activation_last
    drop_rate = args.spatial_dropout
    L2_reg = tf.keras.regularizers.l2(args.L2_reg)
    batch_norm = args.batch_normalization
    padding = args.padding

    #build the bottom
    num_channels = tensor.shape[-1]*2
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
                            name='bottom_conv3d_%s_2d_kernel_%s_time_kernel_%s'%(n,kernel_2d,kernel_time))(tensor)
        if batch_norm==True:
                tensor = tf.keras.layers.BatchNormalization(axis=-1,name='bottom_BN_%s'%(n))(tensor)
        tensor = act_function(tensor=tensor,conv_activation=conv_activation,name='bottom_activation_%s_%s'%(conv_activation,n))
    tensor = SpatialDropout3D(rate=drop_rate,data_format='channels_last',name='bottom_SD_%s_drop_rate_%s'%(0,drop_rate))(tensor)
    return tensor

def build_latent_v3(args,tensor):
    """
    This latent vector uses dense layers
    """
    num_per_block = args.num_layers_per_block
    kernel_2d = 2
    padding=args.padding
    conv_activation = args.activation_conv
    last_activation = args.activation_last
    drop_rate = args.spatial_dropout
    L2_reg = tf.keras.regularizers.l2(args.L2_reg)
    batch_norm = args.batch_normalization
    pool_size = args.pool_size

    tensor = Reshape((tensor.shape[-1],))(tensor)
    tensor = Dropout(rate=drop_rate)(tensor)
    
    tensor = Dense(tensor.shape[-1],
                    activation=None,
                    use_bias=True,
                    kernel_regularizer=L2_reg)(tensor)
    tensor = Dropout(rate=drop_rate)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation,name='bottom_activation_%s_2'%(conv_activation))
    
    tensor = Dense(2048,
                    activation=None,
                    use_bias=True,
                    kernel_regularizer=L2_reg)(tensor)
    tensor = Dropout(rate=drop_rate)(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation,name='bottom_activation_%s_2'%(conv_activation))
    return tensor

def build_decoder_v2(args,obs_skip_list,nwp_skip_list,tensor):

    conv_activation = args.activation_conv
    last_activation = args.activation_last
    drop_rate = args.spatial_dropout
    L2_reg = tf.keras.regularizers.l2(args.L2_reg)
    batch_norm = args.batch_normalization
    padding = args.padding
    
    pool_type = args.pool_type
    pool_size = args.pool_size
    encoder_deep = args.encoder_deep+2
    num_per_block = args.num_layers_per_block
    skip = args.skip
    rotate = args.rotate
    noise = args.noise
    noise_std = args.noise_std
    rotate = args.rotate
    output_shape = args.output_image_size

    tensor = Reshape((1,2,2,512))(tensor)
    if skip==True:
        tensor = Concatenate(axis=-1)([obs_skip_list[-1],nwp_skip_list[-1],tensor])
    if tensor.shape[2]<=16:
        up_samp_size = (1,pool_size,pool_size)
    else:
        up_samp_size = (2,pool_size,pool_size)
    # pool_str = '%s_%s_%s'%(up_samp_size[0],up_samp_size[1],up_samp_size[2])
    # tensor = UpSampling3D(size=up_samp_size,name='UpSample3D_bottom_size_%s_%s_%s'%(up_samp_size[0],up_samp_size[1],up_samp_size[2]))(tensor)
    
    decoder_deep = encoder_deep

    #build the decoder
    unet_3plus = False
    # decoder_channels = np.flip(channels_list)
    skip_idx = len(nwp_skip_list)-2
    num_channels=512
    for d in range(decoder_deep):
        #adjust the spatial kernel sizes based on the lat/lon shape
        if tensor.shape[2]==64:
            kernel_2d = 8
        elif tensor.shape[2]==32:
            kernel_2d = 8
        elif tensor.shape[2]==16:
            kernel_2d = 8
        elif tensor.shape[2]==8:
            kernel_2d = 4
        else:
            kernel_2d = 2
        kernel_time = tensor.shape[1]

        print('decoder:',d,'kernel_time',kernel_time,'kernel_2d',kernel_2d)
        for n in range(num_per_block):
            if tensor.shape[2]==output_shape[2]:
                if unet_3plus==True:
                    unet_3plus=False
                    if skip==True:
                        skip_64 = skip_list[0]
                        pool_size_2 = (1,2,2)
                        skip_64 = MaxPooling3D(pool_size=pool_size_2,name='UNet3_plus_64_skip_1')(skip_64)
                        skip_64 = MaxPooling3D(pool_size=pool_size_2,name='UNet3_plus_64_skip_2')(skip_64)
                        print('skip_64',skip_64)
                        skip_32 = skip_list[1]
                        skip_32 = MaxPooling3D(pool_size=pool_size_2,name='UNet3_plus_32_skip_1')(skip_32)
                        print('skip_32',skip_32)
                    tensor = Concatenate(axis=-1)([tensor,skip_64,skip_32])

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
        
        if d<(decoder_deep-1):
            if tensor.shape[2]<16:
                up_samp_size = (1,pool_size,pool_size)
            else:
                up_samp_size = (2,pool_size,pool_size)
            tensor = UpSampling3D(size=up_samp_size,name='decoder_UpSample3D_%s_size_%s_%s_%s'%(d,up_samp_size[0],up_samp_size[1],up_samp_size[2]))(tensor)
        num_channels=num_channels/2

        if skip==True:
            if skip_idx>=0:
                print('skipping')
                tensor = Concatenate(axis=-1)([obs_skip_list[skip_idx],nwp_skip_list[skip_idx],tensor])
                skip_idx=skip_idx-1
        

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
    tensor = Conv3D(input_shape=tensor.shape,
                    filters=1,
                    kernel_size=(int(kernel_time),int(kernel_2d),int(kernel_2d)),
                    padding=padding,
                    kernel_regularizer=L2_reg,
                    data_format='channels_last',
                    dtype=tf.dtypes.float32,
                    use_bias=True,
                    bias_initializer=tf.keras.initializers.Constant([args.cc_bias]),
                    name='output_tensor')(tensor)
    output_tensor=tensor
    
    return output_tensor

if __name__=='__main__':
    parser = create_parser()
    args = parser.parse_args()
    model = build_conv3d_rotate(args=args)
    
    if model != None:
        print(model.summary())
        # plot_model(model,to_file='NWP_Obs_Separate_Encoder.png', show_shapes=True, show_layer_names=True)
        # print(model.summary())
