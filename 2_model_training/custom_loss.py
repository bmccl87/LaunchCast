import tensorflow as tf
import sys

class combined_bin_reg_loss(tf.keras.losses.Loss):
    def __init__(self, 
                alpha=0.5, 
                bravo=0.5, 
                charlie=0.5, 
                delta=0.5,
                name='combined_bin_reg_loss',
                reduction='mean'):
        super().__init__(name=name)
        self.alpha = alpha
        self.bravo = bravo
        self.charlie = charlie
        self.delta = delta

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Binary classification loss
        binary_loss_ic = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=y_true[:,:,:,0], y_pred=y_pred[:,:,:,0],from_logits=False,axis=0))
        binary_loss_cg = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=y_true[:,:,:,1], y_pred=y_pred[:,:,:,1],from_logits=False,axis=0))

        # Regression loss 
        reg_loss_ic = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true=y_true[:,:,:,2], y_pred=y_pred[:,:,:,2]))
        reg_loss_cg = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true=y_true[:,:,:,3], y_pred=y_pred[:,:,:,3]))

        # Combine with weights
        total_loss = self.alpha*binary_loss_ic + \
                        self.bravo*binary_loss_cg + \
                        self.charlie*reg_loss_ic + \
                        self.delta*reg_loss_cg
        
        return total_loss