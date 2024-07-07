import numpy as np
import tensorflow as tf
import networks
from typing import Literal, Union, List
from data import DataLoader
from display_tools import show_loss_plot

def model_pred(model : tf.keras.Model, 
               images_type : Literal['gray_sin', 'gray_mul','bgr_sin','bgr_mul'], 
               images : Union[np.ndarray, List[np.ndarray]],
               training : bool):
    """compute model's preds
    
    Args:
        model : the model
        images_type : the type of the shape of input images, can be:
            'gray_sin' : gray channel and images have the same (height, width)
            'gray_mul' : gray channel and images have different (height, width)
            'bgr_sin' : BGR channels and images have the same (height, width)
            'bgr_mul' : BGR channels and images have different (height, width)
        images : the images for this batch. array with shape (batch, height, width, channels) for images_type is 'gray_sin' or 'bgr_sin', 
            list of arrays with shape (height,width, channels) for images_type is 'gray_mul' for 'bgr_mul'
        training : True indicates training, False indicates inference
    """

    # for input images with the same shape
    if  images_type == 'gray_sin' or images_type == 'bgr_sin':
        preds = model(images, training=training)  
    # for input images with different shapes
    else:
        preds = []
        for image in images:
            image = np.expand_dims(image,0)
            preds.append(model(image, training=training))
        preds = tf.convert_to_tensor(preds)
        preds = tf.squeeze(preds,1)

    return preds


def train(batch_size : int, 
          epoch : int, 
          data_loader : DataLoader, 
          learning_rate : float, 
          momentum : float, 
          model_config : dict, 
          save_model_epoch = False, 
          save_model = True):
    """train the model
    
    Args:
        batch_size : size of the batch
        epoch : number of epochs
        data_loader : DataLoader
        learning_rate : learning rate
        momentum : momentum of SGD
        model_config : configurations of the model
        save_model_epoch : true indicates saving the model for each epoch
        save_model : true indicates saving the final trained model

    Returns:
        model: the trained model
    """
    # record training loss and accuracy for each epoch
    train_loss_results = []
    train_accuracy_results = []

    model = networks.VGGSPP(**model_config)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # metric objects
    train_loss = tf.keras.metrics.Mean(name='train_loss') # mean of losses within one epoch
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # start training
    for e in range(epoch):
        train_loss.reset_state()
        train_accuracy.reset_state()

        for images, labels in data_loader.get_batch_train(batch_size):
            with tf.GradientTape() as tape:
                preds = model_pred(model=model, 
                                   images_type=data_loader.images_type, 
                                   images=images, 
                                   training=True)
                loss = loss_object(labels=labels, preds=preds) 

            # updata parameters
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # updata metrics
            train_loss(loss)
            train_accuracy(labels, preds)

        train_loss_results.append(train_loss.result())
        train_accuracy_results.append(train_accuracy.result())

        if save_model_epoch:
            model.save_weights(data_loader.images_type + "-" + str(e + 1) + "_epoch_weight.h5")

        print('Epoch {}, Loss: {}, Accuracy: {}%'.format(e + 1,
                                                         train_loss.result(),
                                                         train_accuracy.result()*100))
        
    if save_model: 
        model.save_weights(data_loader.images_type+ ".h5")

    show_loss_plot(loss_results=train_loss_results, 
                   accuracy_results=train_accuracy_results, 
                   images_type=data_loader.images_type,
                   show_plot=False)

    return model


def test(model : tf.keras.Model, 
         data_loader : DataLoader):
    """test the model
    
    Args:
        model: the model to test, can be:
            Model class,
            (model_path, model_cofig): the model_path and model_configuration,
            model_path: the whole model
        data_loader: data_loader

    Returns:
        accuracy: the accuracy on the test set
    """

    if type(model) is tuple:
        model_path, model_config = model
        model = networks.VGGSPP(**model_config)
        model.load_weights(model_path)
    elif type(model) is str:
        model = tf.keras.models.load_model(model)
    preds = model_pred(model=model, 
                       images_type=data_loader.images_type, 
                       images=data_loader.images_test)
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()(data_loader.labels_test, preds).result()

    print(f'Accuracy on test set is {accuracy}')

    return accuracy

    
def train_test(batch_size : int, 
               epoch : int , 
               data_loader : DataLoader, 
               learning_rate : float, 
               momentum : float, 
               model_config : dict, 
               save_model_epoch = False, 
               save_model = True):
    """train and test the model
    
    Args:
        batch_size : size of the batch
        epoch : number of epochs
        data_loader : DataLoader containing the data set
        learning_rate : learning rate
        momentum : momentum of SGD
        model_config : configuration of the model, keys are
            conv_arch : the configuration for the VGG network, ['vgg11','vgg16','vgg19'].
            levels : list of the SPP levels
            num_classes : number of the classes
        save_model_epoch : true indicates saving the model for each epoch
        save_model : true indicates saving the final trained model
    """

    model = train(batch_size=batch_size,
                  epoch=epoch,
                  data_loader=data_loader,
                  learning_rate=learning_rate,
                  momentum=momentum,
                  model_config=model_config,
                  save_model_epoch=save_model_epoch,
                  save_model=save_model)
    test(model, data_loader)