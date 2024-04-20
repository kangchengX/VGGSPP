import tensorflow as tf
import numpy as np

class SPP(tf.keras.layers.Layer):
    def __init__(self, pool_type='max_pool', levels = [1,2,4]):
        super().__init__()
        self.pool_type = pool_type
        self.levels = levels
 
    def call(self, inputs):
 
        shape = inputs.shape
        _, h, w, _ = shape[0], shape[1], shape[2], shape[3]

        # SPP pooling process
        for level in self.levels:

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


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels, max_pool=True):
        super().__init__()
        self.num_convs = num_convs
        self.num_channels = num_channels
        
        # Building the internal Sequential model
        self.conv_layers = tf.keras.Sequential()
        for _ in range(num_convs):
            self.conv_layers.add(tf.keras.layers.Conv2D(num_channels, 
                                                        kernel_size=3, 
                                                        padding='same', 
                                                        activation='relu'))   
        if max_pool:
            self.conv_layers.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    def call(self, inputs):
        outputs = self.conv_layers(inputs)
        return outputs


class VGGSPP(tf.keras.Model):
    def __init__(self, conv_arch, levels, num_classes):
        super().__init__()
        if type(conv_arch) is str:
            conv_arch = CONV_CONFIG[conv_arch]
        self.conv_blocks = tf.keras.Sequential(
            [ConvBlock(num_convs, num_channels, max_pool=max_pool) 
             for num_convs, num_channels, max_pool in conv_arch]
        )
        self.spp_layer = SPP(levels) 
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1000, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs):
        outputs = self.conv_blocks(inputs)
        outputs = self.spp_layer(outputs)
        outputs = self.classifier(outputs)
        return outputs
    
CONV_CONFIG = {
    'vgg11': ((1, 64, True), (1, 128, True), (2, 256, True), (2, 512, True), (2, 512, False)),
    'vgg16': ((2, 64, True), (2, 128, True), (3, 256, True), (3, 512, True), (3, 512, False)),
    'vgg19': ((2, 64, True), (2, 128, True), (3, 256, True), (3, 512, True), (3, 512, False))
}