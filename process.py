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
            save_model_epochs (int | None): If not `None`, the model will be saved every `save_model_epochs` of epochs
                in the `self.folder` folder with name "model - {epoch}.h5". Default to `None`.
            save_model (bool): If `True`, the model after all the epochs will be saved
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
            save_model_epochs (int | None): If not `None`, the model will be saved every `save_model_epochs` of epochs
                in the `self.folder` folder with name "model - {epoch}.h5". Default to `None`.
            save_model (bool): If `True`, the model after all the epochs will be saved
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
        save_filename: str | None = None
    ):
        """
        Train the model and evaluate the model after all the epochs on the test set
        
        Args:
            save_model_epochs (int | None): If not `None`, the model will be saved every `save_model_epochs` of epochs
                in the `self.folder` folder with name "model - {epoch}.h5". Default to `None`.
            save_model (bool): If `True`, the model after all the epochs will be saved
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
