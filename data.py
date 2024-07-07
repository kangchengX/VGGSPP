import cv2
import os
import numpy as np
import warnings
from typing import Literal


class DataLoader():
    '''The DataLoader of the images
    
    Attributes:
        imges_type : type of the imges
        images_size : (height, width) for input images with same shape
        ratio : the ratio for training set in the whole data set
        classes : keys are the class names, values are the coresponding numerical label values
        size_train : number of the images in the loaded training set
        size_test : number of the images in the loaded test set
        images_train : the loaded images in training set. Initial is []. After loaded, list for images with different shapes,
            ndarray for images with the same shape
        images_test : the loaded images in test set. Initial is []. After loaded, list for images with different shapes,
            ndarray for images with the same shape
        labels_train : the loaded labels in training set. Initial is []. After loaded, list for images with different shapes,
            ndarray for images with the same shape
        labels_test : the loaded labels in test set. Initial is []. After loaded, list for images with different shapes,
            ndarray for images with the same shape 
    '''
    def __init__(self, 
                 images_type : Literal['gray_sin', 'gray_mul','bgr_sin','bgr_mul'], 
                 image_size : tuple | None = None, 
                 ratio : float | None = 0.75):
        '''Initialize the model
        
        Args:
            images_type : the type of the shape of input images, can be:
                'gray_sin' : gray channel and images have the same (height, width)
                'gray_mul' : gray channel and images have different (height, width)
                'bgr_sin' : BGR channels and images have the same (height, width)
                'bgr_mul' : BGR channels and images have different (height, width)
            image_size : tuple (height, width) for input images with the same shape, None for
                input images with different shapes
            ratio : the ratio for training set in the whole data set
        '''
        self.images_type = images_type
        self.image_size = image_size
        self.ratio = ratio
        self.classes = {}

        # record the filenames of images that openCV failed to read
        self.filenames_fail = []

        # size of the training and test sets
        self.size_train = 0
        self.size_test = 0

        # data for training and test sets
        self.images_train = []
        self.images_test = []
        self.labels_train = []
        self.labels_test = []

        
    def _process_image(self, img):
        '''convert the image to a (height, width, channel) array with dtype float32 and devided by 255
        
        Args:
            img : the image to process
        
        Returns:
            img : the normalized image. None if failed to read the image
        '''
        width, height = self.image_size
        
        # convert images to the same size
        if self.images_type == 'gray_sin' or self.images_type == 'bgr_sin':
            img = cv2.resize(img,(width,height))

        img = img.astype(np.float32)/255.0

        # expand dim for gray images
        if self.images_type == 'gray_sin' or self.images_type == 'gray_mul':
            img = np.expand_dims(img,axis=-1)

        return img
    
    def _load_image_label(self, label : int, filename : str, train : bool):
        '''Load image and the label to the training or test set
        
        Args:
            label : label of the image
            filename : path of the image
            train : True indicates loading the image and label to the training set.
                False indicates loading the image and the label to the test set
        '''
        # read image from the file to numpy
        if self.images_type == 'gray_sin' or self.images_type == 'gray_mul':
            img = cv2.imread(filename,0)
        else:
            img = cv2.imread(filename)
        
        if img is None:
            self.filenames_fail.append(filename)
            return

        img = self._process_image(img)

        # load image to the training or test set
        if train:
            self.images_train.append(img)
            self.labels_train.append(label)

        else:
            self.images_test.append(img)
            self.labels_test.append(label)

    def _shuffle(self):
        '''Shuffle the training and test set sparately'''
        # get the shuffled indices
        indices_train = np.random.choice(self.size_train, self.size_train, replace=False)
        indices_test = np.random.choice(self.size_test, self.size_test, replace=False)

        # shuffle the training set
        # note, here images and labels are still lists whatever the images_type is
        self.images_train = [self.images_train[i] for i in indices_train]
        self.labels_train = [self.labels_train[i] for i in indices_train]

        # shuffle the test set
        self.images_test = [self.images_test[i] for i in indices_test]
        self.labels_test = [self.labels_test[i] for i in indices_test]

    def load(self, folder : str, shuffle : bool | None = True):
        '''Load dataset from the folder and devide the data set to training set and test set
        
        Args:
            folder : the folder containing the data set. The folder only has subfolders. 
                Each subfolder has the folder name as the class name and containd the images of this class.
            shuffle : True indicates shuffle the training and test sets separately
        '''

        if len(self.images_train) != 0:
            self.images_train = []
            self.images_test = []
            self.labels_train = []
            self.labels_test = []
            self.filenames_fail = []
            self.size_train = 0
            self.size_test = 0
            self.classes = {}

            warnings.warn('the DataLoader has loaded data before', RuntimeWarning)

        # for each label, generate train set and test set
        for label, dir in enumerate(os.listdir(folder)):
            self.classes[dir] = label
            filenames = os.listdir(os.path.join(folder, dir))

            size = len(filenames) # number of images for this class
            size_train = int(size*self.ratio) # number of images in the training set for this class

            # devide filenames to train and test
            for filename in filenames:
                filenames_train= filenames[:size_train]
                filenames_test = filenames[size_train:]

            # load training set
            for filename in filenames_train:
                filename_full = os.path.join(folder, dir, filename)
                self._load_image_label(label=label ,filename=filename_full, train=True)

            # load test set
            for filename in filenames_test:
                filename_full = os.path.join(folder, dir, filename)
                self._load_image_label(label=label ,filename=filename_full, train=False)

        self.size_train = len(self.images_train)
        self.size_test = len(self.images_test)

        # shuffle images
        if shuffle:
            self._shuffle()

        # convert the data set to ndarray if the images have same shape
        if self.images_type == 'gray_sin' or self.images_type == 'bgr_sin':
            self.images_train = np.array(self.images_train,dtype=np.float32)
            self.labels_train = np.array(self.labels_train)
            self.images_test = np.array(self.images_test,dtype=np.float32)
            self.labels_test = np.array(self.labels_test)

        if len(self.filenames_fail) !=0 :
            raise warnings.warn(f'{len(self.filenames_fail)} files were not successfully loaded', RuntimeWarning)
        
    def get_batch_train(self, batch_size : int):
        """get images for each batch for training
        
        Returns:
            zip(images_train_all, labels_train_all) : list of tuples (images_train, labels_train).
                If images_type is 'gray_sin' or 'bgr_sin', images_train are an array with shape (batch, height, width, channels),
                    labels_train are an array with shape (batch). 
                If images_type is 'gray_mul' or 'bgr_nul', images_train are a list of length of batch containing elements with shape (height, width, channels),
                    lables_train are a list of length of batch
        """
        num_iter = self.size_train//batch_size
        images_train_all = []
        labels_train_all = []
        # add the data for each batch
        for iter in range(0,num_iter):
            images_train_all.append(self.images_train[iter*batch_size:(iter+1)*batch_size])
            labels_train_all.append(self.labels_train[iter*batch_size:(iter+1)*batch_size])
        # add the remaining data to the training set
        if self.size_train % batch_size != 0:
            images_train_all.append(self.images_train[(iter+1)*batch_size:])
            labels_train_all.append(self.labels_train[(iter+1)*batch_size:])
            
        return zip(images_train_all, labels_train_all)
    

if __name__ == '__main__':
    print('run')
    data_loader = DataLoader('gray_sin', (244,244))
    data_loader.load('data')