import tensorflow as tf
import keras
from keras.layers import SpatialDropout2D, SpatialDropout3D, AveragePooling3D 
from keras.layers import Dense, Concatenate, Masking, Conv2D, Conv3D, UpSampling2D
from keras.layers import ConvLSTM2D, MaxPooling2D, MaxPooling3D, UpSampling3D
from keras.layers import ConvLSTM1D, LSTM, Reshape, Conv1D, Lambda
from keras.layers import GlobalMaxPool3D, Dropout, Flatten, Masking
from keras.models import Model
import numpy as np
from tensorflow.keras.utils import plot_model
from LC_parser import *
import math

EFM_MASK_VALUE=-999999.0
inputs_dict = {}
class Stack(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.stack(inputs, axis=self.axis)

class FlattenStations(tf.keras.layers.Layer):
    def call(self, x):
        b = tf.shape(x)[0]
        s = tf.shape(x)[1]
        t = tf.shape(x)[2]
        f = tf.shape(x)[3]
        return tf.reshape(x, (b * s, t, f))

class UnflattenStations(tf.keras.layers.Layer):
    def call(self, x):
        # we need original batch size and station count
        b = tf.shape(x)[0] // self.num_stations
        s = self.num_stations
        f = tf.shape(x)[-1]
        return tf.reshape(x, (b, s, f))

    def __init__(self, num_stations, **kwargs):
        super().__init__(**kwargs)
        self.num_stations = num_stations

    def get_config(self):
        config = super().get_config()
        config.update({"num_stations": self.num_stations})
        return config

def next_power_of_two(n: int) -> int:
    if n < 1:
        raise ValueError("Input must be a positive integer.")
    return 1 if n == 1 else 2 ** math.ceil(math.log2(n))

def act_function(tensor,conv_activation,name):
    if conv_activation=='relu':
        tensor = tf.keras.activations.relu(tensor)
    if conv_activation=='leaky_relu':
        tensor = tf.keras.layers.LeakyReLU(alpha=0.01)(tensor)
    if conv_activation=='gelu':
        tensor = tf.keras.activations.gelu(tensor)
    return tensor

def build_efm_ts_encoder(args):
    tf.print('building a new efm encoder without attention')
    ts_input_shape=args.efm_ts_input_shape
    ts_FM_keys = args.efm_keys
    L2_reg = tf.keras.regularizers.l2(args.L2_reg)
    conv_activation=args.activation_conv

    efm_input_list=[]
    shared_LSTM = LSTM(units=16,
                        return_state=False,
                        return_sequences=False,
                        kernel_regularizer=L2_reg)

    for ts_key in ts_FM_keys:
        ts_input_tensor=tf.keras.Input(shape=ts_input_shape,
                                    dtype=tf.dtypes.float32,
                                    name=ts_key)
        inputs_dict[ts_key]=ts_input_tensor
        efm_input_list.append(ts_input_tensor)
    
    tf.print('stacking time series w/mask, and shared lstm inputs')
    stacked_input_tensor = Stack(axis=1,name='stacked_efm_inputs')(efm_input_list)
    bs_stacked_input_tensor = FlattenStations(name="efm_flatten")(stacked_input_tensor)
    mask_input = Masking(mask_value=EFM_MASK_VALUE,name='stacked_efm_input_mask',dtype=tf.dtypes.float32)(bs_stacked_input_tensor)
    stacked_input_lstm = shared_LSTM(mask_input)
    ts_stacked_tensor = UnflattenStations(num_stations=26, name="efm_unflatten")(stacked_input_lstm)
    tf.print('time series stacked successfully')
    
    filters = args.efm_ts_encoder_filters
    for d in range(args.efm_ts_encoder_deep):
        ts_stacked_tensor = Conv1D(filters=filters[d],
                        kernel_size=args.kernel_size,
                        strides=args.strides,
                        padding=args.padding,
                        data_format='channels_last',
                        activation=None,
                        use_bias=True,
                        kernel_regularizer=L2_reg,
                        name='efm_ts_stacked_c_encoder_Conv1D_%s'%d)(ts_stacked_tensor)
        ts_stacked_tensor = act_function(tensor=ts_stacked_tensor,conv_activation=conv_activation,name='efm_ts_activation_%s'%d)
    print(ts_stacked_tensor.shape)
    ts_stacked_tensor = tf.expand_dims(tf.expand_dims(ts_stacked_tensor, -1), -1)
    print(ts_stacked_tensor.shape)

    exp_weighted_distance_input = tf.keras.Input(shape=(26,16,16),dtype=tf.dtypes.float32,name='exp_weighted_distance_input')
    inputs_dict['exp_weighted_distance_input']=exp_weighted_distance_input
    distance_expand = tf.expand_dims(exp_weighted_distance_input, axis=2)
    distance_weighted = ts_stacked_tensor*distance_expand
    tensor = tf.reduce_sum(distance_weighted, axis=1)
    tensor = tf.transpose(tensor, [0, 2, 3, 1])
    print(tensor.shape)

    tensor = Conv2D(filters=tensor.shape[-1]*2,
                        kernel_size=(args.kernel_size,args.kernel_size),
                        strides=args.strides,
                        padding=args.padding,
                        data_format='channels_last',
                        activation=None,
                        use_bias=True,
                        kernel_regularizer=L2_reg,
                        name='efm_proximity_weighted_encoder_Conv2D')(tensor)
    tensor = tf.expand_dims(tensor,axis=1)
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
                inputs_dict['GLM_%s_%s'%(key,t)]=glm_input
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
                    inputs_dict['MRMS_%s_%s'%(key,t)]=mrms_input
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
                    inputs_dict['MRMS_%s_%s'%(key,t)]=mrms_input
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
                    inputs_dict['MRMS_%s_%s'%(key,t)]=mrms_input
                    mask_input = tf.keras.Input(shape=input_shape,
                                                dtype=tf.dtypes.float32,
                                                name='MRMS_%s_mask_%s'%(key,t))
                    inputs_dict['MRMS_%s_mask_%s'%(key,t)]=mask_input
                    if args.rotate==True:
                        mrms_input = tf.keras.layers.RandomRotation(factor=(-args.rotate_dec,args.rotate_dec),fill_mode='reflect')(mrms_input)
                        mask_input = tf.keras.layers.RandomRotation(factor=(-args.rotate_dec,args.rotate_dec),fill_mode='reflect')(mask_input)
                    mrms_input=tf.expand_dims(mrms_input,axis=1)
                    mask_input=tf.expand_dims(mask_input,axis=1)
                    mrms_input = Concatenate(axis=-1)([mrms_input,mask_input])
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
                inputs_dict['HRRR_%s_%s'%(key,t)]=hrrr_input
                if args.rotate==True:
                    hrrr_input=tf.keras.layers.RandomRotation(factor=(-args.rotate_dec,args.rotate_dec),fill_mode='reflect')(hrrr_input)
                hrrr_input=tf.expand_dims(hrrr_input,axis=1)
                hrrr_input_all_times.append(hrrr_input)
            hrrr_feature=Concatenate(axis=1,name='HRRR_%s_all_times'%key)(hrrr_input_all_times)
            hrrr_feature_tensors.append(hrrr_feature)
    nwp_tensor = Concatenate(axis=-1,name='All_HRRR_features_times')(hrrr_feature_tensors)
    print(nwp_tensor)

    gridded_obs_tensor=Concatenate(axis=-1,name='All_Gridded_Obs_Data')([glm_tensor,mrms_tensor])
    
    return gridded_obs_tensor, nwp_tensor

def build_encoder_v3(args,tensor,data_type):
    
    conv_activation = args.activation_conv
    last_activation = args.activation_last
    drop_rate = args.spatial_dropout
    L2_reg = tf.keras.regularizers.l2(args.L2_reg)
    batch_norm = args.batch_normalization
    padding = args.padding
    
    kernel_time = 4
    pool_type = args.pool_type
    pool_size = args.pool_size
    encoder_deep = args.encoder_deep
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
            kernel_2d = 9
        elif tensor.shape[2]==32:
            kernel_2d = 9
        else:
            kernel_2d = 5

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
    
    tensor = Conv3D(input_shape=tensor.shape,
                            filters=num_channels,
                            kernel_size=(int(kernel_time),int(kernel_2d),int(kernel_2d)),
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            use_bias=True,
                            name='%s_encoder_conv3d_final_2d_kernel_%s_time_kernel_%s'%(data_type,kernel_2d,kernel_time))(tensor)
    if batch_norm==True:
        tensor = tf.keras.layers.BatchNormalization(axis=-1,name='%s_encoder_BN_%s_%s'%(data_type,d,n))(tensor)
    tensor = act_function(tensor=tensor,conv_activation=conv_activation,name='%s_encoder_activation_%s_%s_%s'%(data_type,conv_activation,d,n))
    return tensor, skip_list

def build_decoder_v3(args,tensor):
    decoder_filters = [256,128,64,32,16,8,4,2,1]
    kernel_2d=5

    conv_activation = args.activation_conv
    last_activation = args.activation_last
    drop_rate = args.spatial_dropout
    L2_reg = tf.keras.regularizers.l2(args.L2_reg)
    batch_norm = args.batch_normalization
    padding = args.padding
    
    for d,num_channels in enumerate(decoder_filters):
        for n in range(args.num_layers_per_block):
            kernel_time = tensor.shape[1]
            tensor = Conv3D(input_shape=tensor.shape,
                                filters=num_channels,
                                kernel_size=(int(kernel_time),int(kernel_2d),int(kernel_2d)),
                                padding=padding,
                                kernel_regularizer=L2_reg,
                                data_format='channels_last',
                                dtype=tf.dtypes.float32,
                                use_bias=True,
                                name='decoder_conv3d_2d_kernel_%s_time_kernel_%s_channels_%s_%s'%(kernel_2d,kernel_time,num_channels,n))(tensor)
            if batch_norm==True:
                tensor = tf.keras.layers.BatchNormalization(axis=-1,name='decoder_BN_%s_%s'%(d,n))(tensor)
            tensor = act_function(tensor=tensor,conv_activation=conv_activation,name='decoder_activation_%s_%s_%s'%(conv_activation,d,n))
        
        #drop out after the convolutional block
        tensor = SpatialDropout3D(rate=drop_rate,data_format='channels_last',name='decoder_SD_%s_drop_rate_%s'%(d,drop_rate))(tensor)
        if d==1 or d==2:
            tensor=UpSampling3D(size=(2,1,1),name='UpSampleTime_%s'%d)(tensor)
    
    #output layer
    tensor = Conv3D(input_shape=tensor.shape,
                            filters=num_channels,
                            kernel_size=(1,1,1),
                            padding=padding,
                            kernel_regularizer=L2_reg,
                            data_format='channels_last',
                            dtype=tf.dtypes.float32,
                            use_bias=True,
                            name='output_layer_1_1_1_conv')(tensor)
    return tensor

def build_model(args):

    gridded_obs_tensor, nwp_tensor = build_gridded_inputs(args=args)
    nwp_latent, nwp_skip_list = build_encoder_v3(args=args,tensor=nwp_tensor,data_type='NWP')
    obs_latent, obs_skip_list = build_encoder_v3(args=args,tensor=gridded_obs_tensor,data_type='Obs')
    efm_latent = build_efm_ts_encoder(args=args)
    tensor = Concatenate(axis=-1)([nwp_latent,obs_latent,efm_latent])
    tensor = build_decoder_v3(args,tensor)
    output_tensor = tensor
    model=Model(inputs=inputs_dict,outputs=output_tensor)
    print(model.summary())
    return model
    
    
if __name__=='__main__':
    print('building the 64216 model yo')
    parser = create_parser()
    args = parser.parse_args()
    build_model(args=args)
    print('END OF MAIN - LC_model64216.py')