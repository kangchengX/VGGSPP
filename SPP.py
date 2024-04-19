import tensorflow as tf
import numpy as np

# SPP pooling level structure
SPP_LEVEL = [1,2,4]

class SPP_layer(tf.keras.layers.Layer):
    def __init__(self, pool_type='max_pool'):
        super(SPP_layer, self).__init__()
        self.pool_type = pool_type
        self.num_levels = len(SPP_LEVEL)
 
    def call(self, inputs):
 
        shape = inputs.shape
        _, h, w, _ = shape[0], shape[1], shape[2], shape[3]

        # SPP pooling process
        for level in SPP_LEVEL:

            h_pad = np.ceil(h/level).astype(np.int32)*level - h
            w_pad = np.ceil(w/level).astype(np.int32)*level - w
            paddings = tf.constant([[0,0],[0,h_pad],[0,w_pad],[0,0]])
            inputs_pad = tf.pad(inputs,paddings,"CONSTANT")

            kernel_size = [1, np.ceil(h/level).astype(np.int32), np.ceil(w/level).astype(np.int32), 1]
            stride_size = kernel_size

            if self.pool_type == 'max_pool':
                pool = tf.nn.max_pool(inputs_pad, ksize=kernel_size, strides=stride_size, padding='SAME')
                pool = tf.compat.v1.layers.flatten(pool)           
 
            else:
                pool = tf.nn.avg_pool(inputs_pad, ksize=kernel_size, strides=stride_size, padding='SAME')
                pool = tf.compat.v1.layers.flatten(pool)

            # concatenate levels together
            if level == 1:
                x_flatten = pool
            else:
                x_flatten = tf.concat((x_flatten, pool), axis=1)

        return x_flatten