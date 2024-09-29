import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
import numpy as np
from typing import Literal
from config import get_config

class SPP(Layer):
    """The SPP layer. See https://arxiv.org/abs/1406.4729"""
    def __init__(
        self, 
        pool_type: Literal['max_pool', 'avg_pool'] | None = 'max_pool', 
        levels: tuple | None = (1,2,4)
    ):
        """
        Initialize the model.
        
        Args:
            pool_type (str): the pooling type, `'max_pool'` or `'avg_pool'`.
            levels (tuple): the SPP levels.
        """
        super().__init__()

        if pool_type not in ('max_pool', 'avg_pool'):
            raise ValueError(f'unknown pool_type {pool_type}')
        
        self.pool_type = pool_type
        self.levels = levels
        self.flatten = layers.Flatten()
 
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

            # pooling
            if self.pool_type == 'max_pool':
                pool = tf.nn.max_pool(inputs_pad, ksize=kernel_size, strides=stride_size, padding='SAME')        
            else:
                pool = tf.nn.avg_pool(inputs_pad, ksize=kernel_size, strides=stride_size, padding='SAME')
            
            # flatten the features
            pool_flatten = self.flatten(pool)

            # concatenate levels together
            if level == self.levels[0]:
                x_flatten = pool_flatten
            else:
                x_flatten = tf.concat((x_flatten, pool_flatten), axis=1)

        return x_flatten


class ConvBlock(Layer):
    """The convolutional block, containing all the convolutional layers and pooling layers"""
    def __init__(
        self, 
        num_convs : int, 
        num_channels : int, 
        max_pool : bool | None = True
    ):
        """
        Initialize the model.
        
        Args:
            num_convs (int): number of convolutional layers with kernel_size = 3, padding = 'same' and actication = 'relu'.
            num_channel (int): number of output channels.
            max_pool (bool): True indicated adding a max pooling layer in the end with pooling_size = 2 and strides = 2.
        """

        super().__init__()
        self.num_convs = num_convs
        self.num_channels = num_channels
        
        # Building the internal Sequential model
        self.conv_layers = models.Sequential()
        for _ in range(num_convs):
            self.conv_layers.add(
                layers.Conv2D(
                    num_channels, 
                    kernel_size=3, 
                    padding='same', 
                    activation='relu'
                )
            )
        if max_pool:
            self.conv_layers.add(layers.MaxPool2D(pool_size=2, strides=2))

    def call(self, inputs):
        outputs = self.conv_layers(inputs)
        return outputs


class VGGSPP(Model):
    """The implementation of VGGSPP model."""
    def __init__(
        self, 
        conv_arch: Literal['vgg11','vgg16','vgg19'], 
        levels: list, 
        num_classes: int
    ):
        """
        Initialize the model.
        
        Args:
            conv_arch (str): the configuration for the VGG network, can be `'vgg11'`, `'vgg16'`, or `'vgg19'`.
            levels (list): the SPP levels.
            num_classes (int): number of the classes.
        """

        super().__init__()

        # convolutional layers plus traditional pooling
        conv_arch = get_config(conv_arch)
        self.conv_blocks = models.Sequential(
            [ConvBlock(num_convs, num_channels, max_pool=max_pool) 
             for num_convs, num_channels, max_pool in conv_arch]
        )

        # SPP layer
        self.spp_layer = SPP(levels=levels)

        # MLP block
        self.classifier = models.Sequential([
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1000, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs):
        outputs = self.conv_blocks(inputs)
        outputs = self.spp_layer(outputs)
        outputs = self.classifier(outputs)
        return outputs