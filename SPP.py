import tensorflow as tf
import numpy as np

#金字塔池化的 level 结构
SPP_LEVEL = [1,2,4]

class SPP_layer(tf.keras.layers.Layer):
    def __init__(self, pool_type='max_pool'):
        super(SPP_layer, self).__init__()
        self.pool_type = pool_type


    def build(self, input_shape):
        self.num_levels = len(SPP_LEVEL)
        self.pool_type = 'max_pool'
 
        # 为该层创建可训练权重变量
        # self.kernel = self.add_weight(name='kernel',
        #                               shape=input_shape,
        #                               initializer='uniform',
        #                               trainable=True)
        # Be sure to call this at the end

 
    def call(self, inputs):
 
        shape = inputs.shape
        batch_size, h, w, channel = shape[0], shape[1], shape[2], shape[3]

        #金字塔池化过程
        for level in SPP_LEVEL:

            h_pad = np.ceil(h/level).astype(np.int32)*level - h
            w_pad = np.ceil(w/level).astype(np.int32)*level - w
            paddings = tf.constant([[0,0],[0,h_pad],[0,w_pad],[0,0]])
            inputs_pad = tf.pad(inputs,paddings,"CONSTANT")

            kernel_size = [1, np.ceil(h/level).astype(np.int32), np.ceil(w/level).astype(np.int32), 1]
            stride_size = kernel_size

            if self.pool_type == 'max_pool':
                pool = tf.nn.max_pool(inputs_pad, ksize=kernel_size, strides=stride_size, padding='SAME')
                #将特征图展为一维
                pool = tf.compat.v1.layers.flatten(pool)           
                ##不能用于 batch_size 为 None
                #pool = tf.reshape(pool, (batch_size, -1))
 
            else:
                pool = tf.nn.avg_pool(inputs_pad, ksize=kernel_size, strides=stride_size, padding='SAME')
                pool = tf.compat.v1.layers.flatten(pool)

            #将各个level拼接
            if level == 1:
                x_flatten = pool
            else:
                x_flatten = tf.concat((x_flatten, pool), axis=1)

        return x_flatten