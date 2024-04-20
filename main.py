# 设置 tensorflow 禁止使用 GPU
# 在import tensorflow之前
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import networks
import data

BATCH_SIZE = 16 #batch大小128
EPOCH_SIZE = 20 #epoch大小20
MOM_BETA = 0.9 #momentum算法的摩擦系数
LEARN_RATE = 0.01 #梯度下降的学学习率


##tensorflow gpu 版本时 设置显存申请方式
# for gpu in tf.config.experimental.list_physical_devices('GPU'):
#     print(gpu)
#     tf.config.experimental.set_memory_growth(gpu, True)


def show_loss_plot(loss_results, accuracy_results, images_type):
    """visulize loss and accuracy
    
    Args:
        loss_results (list[float]): list of loss values
        accuracy_results (list[float]): list of accuracy values
        images_type: type of the input images
    """
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(loss_results)
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(accuracy_results)
    plt.savefig(str(images_type) + '-' + 'show_loss.jpg')
    plt.show()


def model_pred(model, images_type, images):
    """compute model's preds
    
    Args:
        model: the model
        images_type: type of the input images
        images: images to feed to the model at this batch
    """

    # for input images with the same shape
    if  images_type == data.CHANGE_GRAY_SIN or images_type == data.CHANGE_BGR_SIN:
        preds = model(images, training=True)  
    # for input images with different shapes
    else:
        preds = []
        for image in images:
            image = np.expand_dims(image,0)
            preds.append(model(image, training=True))
        preds = tf.convert_to_tensor(preds)
        preds = tf.squeeze(preds,1)

    return preds


def train(batch_size:int, epoch:int, images_type, 
          data_loader:data.DataLoader, learning_rate:float, momentum:float, 
          model_config:dict, save_model_epoch=False, save_model=True):
    """train the model
    
    Args:
        batch_size (int): size of the batch
        epoch (int): number of epochs
        images_type: type of the input images
        data_loader: DataLoader
        learning_rate: learning rate
        momentum: momentum of SGD
        model_config (dict): configurations of the model
        save_model_epoch (bool): true indicates saving the model for each epoch
        save_model (bool): true indicates saving the final trained model

    Returns:
        model: the trained model
    """

    train_loss_results = []
    train_accuracy_results = []

    model = networks.VGGSPP(**model_config)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    train_loss = tf.keras.metrics.Mean(name='train_loss') # mean of losses within one epoch
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # start training
    for e in range(epoch):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for images, labels in data_loader.get_batch_train(batch_size):
            with tf.GradientTape() as tape:
                preds = model_pred(model,images_type,images)
                loss = loss_object(labels, preds) 

            # updata parameters
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # updata metrics
            train_loss(loss)
            train_accuracy(labels, preds)

        train_loss_results.append(train_loss.result())
        train_accuracy_results.append(train_accuracy.result())

        if save_model_epoch:
            model.save_weights(str(images_type) + "-" + str(e + 1) + "_epoch_weight.h5")

        print('Epoch {}, Loss: {}, Accuracy: {}%'.format(e + 1,
                                                         train_loss.result(),
                                                         train_accuracy.result()*100))
        
    if save_model: 
        model.save_weights(str(images_type) + ".h5")

    show_loss_plot(train_loss_results, train_accuracy_results, images_type)

    return model


def test(model, data_loader:data.DataLoader, images_type):
    """test the model
    
    Args:
        model: the model to train, can be:
            Model class
            (model_path, model_cofig): the model_path and model_configuration
            model_path: the whole model
        model_config: configuration of the model
        data_loader: data_loader
        images_type: type of the input images

    Returns:
        accuracy: the accuracy
    """

    if type(model) is tuple:
        model_path, model_config = model
        model = networks.VGGSPP(**model_config)
        model.load_weights(model_path)
    elif type(model) is str:
        model = tf.keras.models.load_model(model)

    preds = model_pred(model,images_type, data_loader.images_test)
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()(data_loader.labels_test, preds).result()

    print(f'Accuracy on test set is {accuracy}')

    return accuracy

    
def train_test(batch_size:int, epoch:int, images_type, 
               data_loader:data.DataLoader, learning_rate:float, momentum:float, 
               model_config:dict, save_model_epoch=False, save_model=True):
    """train and test the model
    
    Args:
        batch_size (int): size of the batch
        epoch (int): number of epochs
        images_type: type of the input images
        data_loader: DataLoader
        learning_rate: learning rate
        momentum: momentum of SGD
        model_config (dict): configurations of the model
        save_model_epoch (bool): true indicates saving the model for each epoch
        save_model (bool): true indicates saving the final trained model
    """

    model = train(batch_size,epoch,images_type,data_loader,learning_rate,momentum,
                  model_config,save_model_epoch,save_model)
    test(model,data_loader,images_type)


if __name__ == '__main__':
    img_type_list = ['gray_sin','gray_mul','bgr_sin','bgr_mul']
    image_size = (224,224)
    model_config = {
        'conv_arch': 'vgg16',
        'levels': [1,2,4], 
        'num_classes': 4
    }
    data_loader = data.DataLoader('gray_sin', image_size)
    data_loader.load_folder('data',0.75,True)

    for images_type in img_type_list:
        print(images_type + ':')
        data_loader.images_type = images_type
        data_loader.load_filenames()
        train_test(data_loader=data_loader, 
                   batch_size=16,
                   epoch=20, 
                   images_type=images_type,
                   learning_rate=0.01,
                   momentum=0.8,
                   model_config=model_config,
                   save_model_epoch=False,
                   save_model=False)