import numpy as np
import tensorflow as tf
import os, json
from models import VGGSPP
from data import DataLoader
from typing import Union, List
from display_tools import show_loss_plot


class Processor:
    """The class to train and test the model."""
    def __init__(
            self,
            model: VGGSPP,
            data_loader: DataLoader,
            learning_rate: float | None,
            batch_size: float | None,
            num_epochs: float | None,
            folder: str | None = None
    ):
        """
        Initialize the model.

        Args:
            model (VGGSPP): the VGGSPP model.
            data_loader (DataLoader): the DataLoader containing the training and test data. Default to `None`.
            learning_rate (float): learning rate. Default to `None`.
            batch_size (int): size of the batch. Default to `None`.
            num_epochs (int): number of epochs. Default to `None`.
            folder (str): folder to save the model and training log. If 'None', the folder will be the `images_shapes_type`. Default to `None`.
        """

        self.model = model
        self.data_loader = data_loader
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.train_losses = []
        self.train_accuracies = []

        self.test_accuracy = None

        if folder is None:
            self.folder = self.data_loader.images_shapes_type
        else:
            self.folder = folder

    def _pred(self, images: Union[np.ndarray, List[np.ndarray]], training: bool):
        """
        Compute model's preds.

        Args:
            images (ndarray | list): images to feed to the model.
            training (bool): `True` indicates training, `False` indicates inference.
        """

        # for input images with different shapes
        if isinstance(images, list):
            preds = []
            for image in images:
                image = np.expand_dims(image,0)
                preds.append(self.model(image, training=training))
            preds = tf.convert_to_tensor(preds)
            preds = tf.squeeze(preds,1)
        
        # for input images with the same shape
        else:
            preds = self.model(images, training=training)  

        return preds
    
    def _training(
            self,
            save_model_epochs: int | None = None, 
            save_model: bool | None = True,
            show_plot: bool | None = False,
            save_filename: str | None=None
    ):
        """
        Model training.
        
        Args:
            save_model_epochs (int | None): If not `None`, the model will be saved every `save_model_epochs` of epochs \
                in the `self.folder` folder with name "model - {epoch}.h5". Default to `None`.
            save_model (bool): If `True`, the model after all the epochs will be saved \
                in the `self.folder` folder with name "model - {epoch}.h5". Default to `True`.
            show_plot (bool): If `True`, the losses plot will be shown.
            save_filename (str): Path to save the plot if not `None`. Default to `None`. 
        """
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        # metric objects
        train_loss = tf.keras.metrics.Mean(name='train_loss') # mean of losses within one epoch
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        # start training
        for e in range(self.num_epochs):
            train_loss.reset_state()
            train_accuracy.reset_state()

            for images, labels in self.data_loader.get_batch_train(self.batch_size):
                with tf.GradientTape() as tape:
                    preds = self._pred(images, training=True)
                    loss = loss_object(labels, preds) 

                # update parameters
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                # update metrics
                train_loss(loss)
                train_accuracy(labels, preds)

            self.train_losses.append(float(train_loss.result()))
            self.train_accuracies.append(float(train_accuracy.result()))

            if save_model_epochs is not None and (e+1) % save_model_epochs == 0:
                self.model.save_weights(
                    os.path.join(self.folder, "model - " + str(e + 1) + ".h5")
                )

            print('Epoch {}, Loss: {}, Accuracy: {}%'.format(e + 1, train_loss.result(), train_accuracy.result()*100))
            
        if save_model: 
            self.model.save_weights(
                os.path.join(self.folder, "model - " + str(self.num_epochs) + ".h5")
            )

        show_loss_plot(
            loss_results=self.train_losses, 
            accuracy_results=self.train_accuracies, 
            images_shapes_type=self.data_loader.images_shapes_type,
            show_plot=show_plot,
            save_filename=save_filename
        )

    def _inference(self):
        """Model inference."""

        preds = self._pred(self.data_loader.images_test, training=False)
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()(self.data_loader.labels_test, preds).numpy()

        self.test_accuracy = float(accuracy)

        print(f'Accuracy on test set is {accuracy}')

    def _save_log(self):
        """Save the training log"""
        log = {
            'batch_size' : self.batch_size,
            'num_epochs' : self.num_epochs,
            'learning_rate' : self.learning_rate,
            'train_losses' : self.train_losses,
            'train_accuracies' : self.train_accuracies,
            'test_accuracy' : self.test_accuracy
        }

        with open(os.path.join(self.folder, 'training_log.json'), 'w') as f:
            json.dump(log, f, indent=4)
    
    def train(
            self,
            save_model_epochs: int | None = None, 
            save_model: bool | None = True,
            show_plot: bool | None = False,
            save_filename: str | None=None
    ):
        """
        Train the model and save the training log.

        Args:
            save_model_epochs (int | None): If not `None`, the model will be saved every `save_model_epochs` of epochs \
                in the `self.folder` folder with name "model - {epoch}.h5". Default to `None`.
            save_model (bool): If `True`, the model after all the epochs will be saved \
                in the `self.folder` folder with name "model - {epoch}.h5". Default to `True`.
            show_plot (bool): If `True`, the losses plot will be shown.
            save_filename (str): Path to save the plot if not `None`. Default to `None`. 
        """
        self._training(
            save_model_epochs=save_model_epochs, 
            save_model=save_model, 
            show_plot=show_plot,
            save_filename=save_filename
        )
        self._save_log()

    def test(self):
        """Test the model and save the log."""
        self._inference()
        self._save_log()

    def train_and_test(
            self,
            save_model_epochs: int | None = None, 
            save_model: bool | None = True,
            show_plot: bool | None = False,
            save_filename: str | None=None
    ):
        """
        Train the model and evaluate the model after all the epochs on the test set
        
        Args:
            save_model_epochs (int | None): If not `None`, the model will be saved every `save_model_epochs` of epochs \
                in the `self.folder` folder with name "model - {epoch}.h5". Default to `None`.
            save_model (bool): If `True`, the model after all the epochs will be saved \
                in the `self.folder` folder with name "model - {epoch}.h5". Default to `True`.
            show_plot (bool): If `True`, the losses plot will be shown.
            save_filename (str): Path to save the plot if not `None`. Default to `None`. 
        """
        self._training(
            save_model_epochs=save_model_epochs, 
            save_model=save_model, 
            show_plot=show_plot,
            save_filename=save_filename
        )
        self._inference()
        self._save_log()
        

# def model_pred(
#         model: Model, 
#         images_shapes_type: Literal['gray_sin', 'gray_mul','bgr_sin','bgr_mul'], 
#         images: Union[np.ndarray, List[np.ndarray]],
#         training: bool
# ):
#     """
#     Compute model's preds.
    
#     Args:
#         model : the model
#         images_shapes_type : the type of the shape of input images, can be:
#             'gray_sin' : gray channel and images have the same (height, width)
#             'gray_mul' : gray channel and images have different (height, width)
#             'bgr_sin' : BGR channels and images have the same (height, width)
#             'bgr_mul' : BGR channels and images have different (height, width)
#         images : the images for this batch. array with shape (batch, height, width, channels) for images_shapes_type is 'gray_sin' or 'bgr_sin', 
#             list of arrays with shape (height,width, channels) for images_shapes_type is 'gray_mul' for 'bgr_mul'.
#         training : True indicates training, False indicates inference.
#     """

#     # for input images with the same shape
#     if  images_shapes_type == 'gray_sin' or images_shapes_type == 'bgr_sin':
#         preds = model(images, training=training)  
#     # for input images with different shapes
#     else:
#         preds = []
#         for image in images:
#             image = np.expand_dims(image,0)
#             preds.append(model(image, training=training))
#         preds = tf.convert_to_tensor(preds)
#         preds = tf.squeeze(preds,1)

#     return preds


# def train(
#         batch_size: int, 
#         epoch: int, 
#         data_loader : DataLoader, 
#         learning_rate : float, 
#         momentum : float, 
#         model_config : dict, 
#         save_model_epochs = False, 
#         save_model = True
# ):
#     """
#     Train the model
    
#     Args:
#         batch_size : size of the batch
#         epoch : number of epochs
#         data_loader : DataLoader
#         learning_rate : learning rate
#         momentum : momentum of SGD
#         model_config : configurations of the model
#         save_model_epochs : true indicates saving the model for each epoch
#         save_model : true indicates saving the final trained model

#     Returns:
#         model: the trained model
#     """
#     # record training loss and accuracy for each epoch
#     train_loss_results = []
#     train_accuracy_results = []

#     model = models.VGGSPP(**model_config)
#     optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

#     # metric objects
#     train_loss = tf.keras.metrics.Mean(name='train_loss') # mean of losses within one epoch
#     train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

#     # start training
#     for e in range(epoch):
#         train_loss.reset_state()
#         train_accuracy.reset_state()

#         for images, labels in data_loader.get_batch_train(batch_size):
#             with tf.GradientTape() as tape:
#                 preds = model_pred(model=model, 
#                                    images_shapes_type=data_loader.images_shapes_type, 
#                                    images=images, 
#                                    training=True)
#                 loss = loss_object(labels, preds) 

#             # updata parameters
#             gradients = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#             # updata metrics
#             train_loss(loss)
#             train_accuracy(labels, preds)

#         train_loss_results.append(train_loss.result())
#         train_accuracy_results.append(train_accuracy.result())

#         if save_model_epochs:
#             model.save_weights(data_loader.images_shapes_type + "-" + str(e + 1) + "_epoch_weight.h5")

#         print('Epoch {}, Loss: {}, Accuracy: {}%'.format(e + 1,
#                                                          train_loss.result(),
#                                                          train_accuracy.result()*100))
        
#     if save_model: 
#         model.save_weights(data_loader.images_shapes_type+ ".h5")

#     show_loss_plot(loss_results=train_loss_results, 
#                    accuracy_results=train_accuracy_results, 
#                    images_shapes_type=data_loader.images_shapes_type,
#                    show_plot=False)

#     return model


# def test(
#         model: Model, 
#         data_loader: DataLoader
# ):
#     """test the model
    
#     Args:
#         model: the model to test, can be:
#             Model class,
#             (model_path, model_cofig): the model_path and model_configuration,
#             model_path: the whole model
#         data_loader: data_loader

#     Returns:
#         accuracy: the accuracy on the test set
#     """

#     if type(model) is tuple:
#         model_path, model_config = model
#         model = models.VGGSPP(**model_config)
#         model.load_weights(model_path)
#     elif type(model) is str:
#         model = tf.keras.models.load_model(model)
#     preds = model_pred(model=model, 
#                        images_shapes_type=data_loader.images_shapes_type, 
#                        images=data_loader.images_test)
#     accuracy = tf.keras.metrics.SparseCategoricalAccuracy()(data_loader.labels_test, preds).result()

#     print(f'Accuracy on test set is {accuracy}')

#     return accuracy

    
# def train_test(batch_size : int, 
#                epoch : int , 
#                data_loader : DataLoader, 
#                learning_rate : float, 
#                momentum : float, 
#                model_config : dict, 
#                save_model_epochs = False, 
#                save_model = True):
#     """train and test the model
    
#     Args:
#         batch_size : size of the batch
#         epoch : number of epochs
#         data_loader : DataLoader containing the data set
#         learning_rate : learning rate
#         momentum : momentum of SGD
#         model_config : configuration of the model, keys are
#             conv_arch : the configuration for the VGG network, ['vgg11','vgg16','vgg19'].
#             levels : list of the SPP levels
#             num_classes : number of the classes
#         save_model_epochs : true indicates saving the model for each epoch
#         save_model : true indicates saving the final trained model
#     """

#     model = train(batch_size=batch_size,
#                   epoch=epoch,
#                   data_loader=data_loader,
#                   learning_rate=learning_rate,
#                   momentum=momentum,
#                   model_config=model_config,
#                   save_model_epochs=save_model_epochs,
#                   save_model=save_model)
#     test(model, data_loader)