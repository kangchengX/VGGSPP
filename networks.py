import tensorflow as tf
import numpy as np
from typing import Literal
from constants import CONV_CONFIG

class SPP(tf.keras.layers.Layer):
    '''The SPP layer. See https://arxiv.org/abs/1406.4729'''
    def __init__(self, 
                 pool_type : Literal['max_pool', 'avg_pool'] | None = 'max_pool', 
                 levels : list | None = [1,2,4]):
        '''Initialize the model
        
        Args:
            pool_type : the pooling type, 'max_pool' or avg_pool'
            levels : the SPP levels
        '''
        super().__init__()

        if pool_type not in ('max_pool', 'avg_pool'):
            raise ValueError(f'unknown pool_type {pool_type}')
        
        self.pool_type = pool_type
        self.levels = levels
 
    def call(self, inputs):
 
        shape = inputs.shape
        _, h, w, _ = shape[0], shape[1], shape[2], shape[3]

        # SPP pooling
        for level in self.levels:
            # calculate the size of pooling kernel dynamicly to ensure same output shape
            h_pad = np.ceil(h/level).astype(np.int32)*level - h
            w_pad = np.ceil(w/level).astype(np.int32)*level - w
            paddings = tf.constant([[0,0],[0,h_pad],[0,w_pad],[0,0]])
            inputs_pad = tf.pad(inputs,paddings,"CONSTANT")

            kernel_size = [1, np.ceil(h/level).astype(np.int32), np.ceil(w/level).astype(np.int32), 1]
            stride_size = kernel_size

            if self.pool_type == 'max_pool':
                pool = tf.nn.max_pool(inputs_pad, ksize=kernel_size, strides=stride_size, padding='SAME')        
            else:
                pool = tf.nn.avg_pool(inputs_pad, ksize=kernel_size, strides=stride_size, padding='SAME')
            
            # flatten the features
            pool = tf.compat.v1.layers.flatten(pool)

            # concatenate levels together
            if level == 1:
                x_flatten = pool
            else:
                x_flatten = tf.concat((x_flatten, pool), axis=1)

        return x_flatten


class ConvBlock(tf.keras.layers.Layer):
    '''The convolutional block, containing all the convolutional layers and pooling layers'''
    def __init__(self, 
                 num_convs : int, 
                 num_channels : int, 
                 max_pool : bool | None = True):
        '''Initialize the model
        
        Args:
            num_convs : number of convolutional layers with kernel_size = 3, padding = 'same' and actication = 'relu'
            num_channel : number of output channels
            max_pool : True indicated adding a max pooling layer in the end with pooling_size = 2 and strides = 2
        '''
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
    '''The implementation of VGGSPP model'''
    def __init__(self, 
                 conv_arch : Literal['vgg11','vgg16','vgg19'], 
                 levels : list, 
                 num_classes : int):
        '''Initialize the model
        
        Args:
            conv_arch : the configuration for the VGG network, ['vgg11','vgg16','vgg19'].
            levels : the SPP levels
            num_classes : number of the classes
        '''
        super().__init__()
        # convolutional layers plus traditional pooling
        if type(conv_arch) is str:
            conv_arch = CONV_CONFIG[conv_arch]
        self.conv_blocks = tf.keras.Sequential(
            [ConvBlock(num_convs, num_channels, max_pool=max_pool) 
             for num_convs, num_channels, max_pool in conv_arch]
        )
        # SPP layer
        self.spp_layer = SPP(levels)
        # MLP block
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