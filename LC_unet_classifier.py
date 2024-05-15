import tensorflow as tf
from keras.layers import InputLayer, Dense, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, AveragePooling2D
from tensorflow.keras.models import Sequential, Model




def create_unet(image_size=64,
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

    for i in range(n_hrrr_params):

        temp_tensor=tf.keras.Input(shape=(image_size,image_size,1),dtype=tf.dtypes.float64,name='hrrr_input_'+str(i))
        
        if i==0:
            input_tensor=temp_tensor
        else:
            input_tensor=tf.concat([input_tensor,temp_tensor])
        
        tensor = input_tensor

        output_tensor = Conv2D(filters=n_types,padding=padding,strides=1,kernel_size=(conv_size,conv_size),use_bias=True,name='MERLIN_Output',activation='linear')(tensor)

         #compile the model 
        model = Model(inputs=input_tensor,outputs=output_tensor)
        opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
        model.compile(optimizer=opt,loss=loss)

        return model

    


