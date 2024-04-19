import tensorflow as tf
import numpy as np
VGG_DENSE1 = 4096 #全连接层第一层单元数
VGG_DENSE2 = 4096 #全连接层第二层单元数
VGG_DENSE3 = 1000 #全连接层第三层单元数
CLASS_NUM = 4 #类别数

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
    

def vgg_block(num_convs, num_channels,max_pool):
    """建立VGG中的卷积层和池化层"""
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size=3,
                                    padding='same',activation='relu'))
    if max_pool:
        blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk


def build_vggsppnet(conv_arch):
    """建立VGGSPP"""
    net = tf.keras.models.Sequential()
    #添加VGG中的卷积层和池化层
    for (num_convs, num_channels,max_pool) in conv_arch:
        net.add(vgg_block(num_convs,num_channels,max_pool))
    #添加SPP和VGG的全连接层,利用 softmax 分为 4 类
    net.add(tf.keras.models.Sequential([SPP.SPP_layer(),#将 SPP.SPP_layer()换成tf.keras.layers.Flatten()是VGG模型
             tf.keras.layers.Dense(VGG_DENSE1,activation='relu'),
             tf.keras.layers.Dropout(0.5),
             tf.keras.layers.Dense(VGG_DENSE2,activation='relu'),
             tf.keras.layers.Dropout(0.5),
             tf.keras.layers.Dense(VGG_DENSE3,activation='relu'),
             tf.keras.layers.Dense(CLASS_NUM,activation='softmax')]))
    return net


def build_vggspp(keyword = 'vgg16'):
    if keyword == 'vgg11':
        conv_arch = ((1, 64, True), (1, 128, True), (2, 256, True), (2, 512, True), (2, 512, False))  #VGG11SPP
    if keyword == 'vgg16':
        conv_arch = ((2, 64, True), (2, 128, True), (3, 256, True), (3, 512, True), (3, 512, False))  #VGG16SPP
    if keyword == 'vgg19':
        conv_arch = ((2, 64, True), (2, 128, True), (4, 256, True), (4, 512, True), (4, 512, False))  #VGG19SPP
    net = build_vggsppnet(conv_arch)
    return net