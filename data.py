import cv2
import os
import numpy as np

## ways to convert the images
CHANGE_GRAY_SIN = 0 # convert to gray images with the same size
CHANGE_BGR_SIN = 1 # convert to BGR images with the same size
CHANGE_GRAY_MUL = 2 # convert to gray images with different sizes
CHANGE_BGR_MUL = 3 # convert to BGR images with different sizes

## image size for the same size convertion
IMG_PRE_OUT_WIDTH = 244
IMG_PRE_OUT_HEIGHT = 244


class DataLoader():
    def __init__(self, change_type, image_size, filenames_train=[],filenames_test=[]):
        self.change_type = change_type
        self.image_size = image_size
        self.classes = [] # record names of classes
        self.num_false = 0 # record the number of images that openCV failed to read
        self.num_train = 0
        self.num_test = 0
        self.filenames_train = filenames_train
        self.filenames_test = filenames_test
        self.images_train = []
        self.images_test = []
        self.labels_train = []
        self.labels_test = []
        
    def _process_image(self, filename):
        '''convert images to (width, height, channel) arrays with dtype float32 and devided by 255
        
        Args:
            filename: path of the image
            change_type: the way to convert the image
            image_size: size of the image (for same size convertion)
        '''
        width, height = self.image_size

        if self.change_type == CHANGE_GRAY_SIN or self.change_type == CHANGE_GRAY_MUL:
            img = cv2.imread(filename,0)
        else:
            img = cv2.imread(filename)
        
        if img is None:
            return None
        
        # convert images to the same size
        if self.change_type == CHANGE_GRAY_MUL or self.change_type == CHANGE_BGR_MUL:
            img = cv2.resize(img,(width,height))

        img = img.astype(np.float32)/255.0

        # expand dim for gray images
        if self.change_type == CHANGE_GRAY_SIN or self.change_type == CHANGE_GRAY_MUL:
            img = np.expand_dims(img,axis=-1)

        return img
    
    def _load_image(self,label, filename, train):
        img = self._process_image(filename)
        if img is None:
            self.num_false += 0
            return

        # add the data to train set
        if train:
            self.images_train.append(img)
            self.labels_train.append(label)
            self.filenames_train.append(filename)

        else:
            self.images_test.append(img)
            self.labels_test.append(label)
            self.filenames_test.append(filename)


    def load_folder(self,folder=None, ratio=0.75,shuffle=False):
        '''load dataset and devide the data set to training set and testing set'''

        self.images_train = []
        self.images_test = []
        self.labels_train = []
        self.labels_test = []

        # for each label, generate train set and test set
        for label, dir in enumerate(os.listdir(folder)):
            self.classes.append(dir)
            filenames = os.listdir(os.path.join(folder, dir))

            size = len(filenames)
            size_train = int(size*ratio)

            for filename in filenames:
                # devide filenames to train and test
                size = len(filenames)
                size_train = int(size*ratio)

                filenames_train= filenames[:size_train]
                filenames_test = filenames[size_train:]

            for filename in filenames_train:
                # skip files that openCV fails to read
                filename = os.path.join(folder,dir,filename)
                self._load_image(label,filename,True)

            for filename in filenames_test:
                filename = os.path.join(folder,dir,filename)
                self._load_image(label,filename,False)

        if shuffle:
            size_train_entire = len(self.images_train)
            size_test_entire = len(self.images_test)

            indices_train = np.random.choice(size_train_entire,size_train_entire,False)
            indices_test = np.random.choice(size_test_entire,size_test_entire,False)

            self.images_train = self.images_train[indices_train]
            self.labels_train = self.labels_train[indices_train]

            self.images_test = self.images_test[indices_test]
            self.labels_test = self.labels_test[indices_test]

        self.num_train = len(self.filenames_train)
        self.num_test = len(self.filenames_test)

        if self.change_type == CHANGE_BGR_SIN or self.change_type == CHANGE_GRAY_SIN:
            self.images_train = np.array(self.images_train,dtype=np.float32)
            self.labels_train = np.array(self.labels_train)
            self.images_test = np.array(self.images_test,dtype=np.float32)
            self.labels_test = np.array(self.labels_test)


    def load_filenames(self):
        self.images_train = []
        self.images_test = []
        self.labels_train = []
        self.labels_test = []
        for filename, label in zip(self.filenames_train,self.labels_train):
            self._load_image(label,filename,True)
        for filename, label in zip(self.filenames_test,self.labels_test):
            self._load_image(label, filename,False)

        self.num_train = len(self.filenames_train)
        self.num_test = len(self.filenames_test)

        if self.change_type == CHANGE_BGR_MUL or self.change_type == CHANGE_GRAY_SIN:
            self.images_train = np.array(self.images_train,dtype=np.float32)
            self.labels_train = np.array(self.labels_train)
            self.images_test = np.array(self.images_test,dtype=np.float32)
            self.labels_test = np.array(self.labels_test)


    def get_batch_train(self, batch_size):
        """get images for each batch for training"""
        index_all = np.random.choice(self.num_train,self.num_train,replace= False)
        num_iter = self.num_train//batch_size
        train_images_all = []
        train_labels_all = []
        for iter in range(0,num_iter):
            index = index_all[iter*batch_size:(iter+1)*batch_size]
            if self.change_type == CHANGE_GRAY_SIN or self.change_type == CHANGE_BGR_SIN:
                train_images_all.append(self.train_images[index])
                train_labels_all.append(self.train_labels[index])
            else:
                train_images_list = []
                for ind in index:
                    train_images_list.append(self.train_images[ind])
                train_images_all.append(train_images_list)
                train_labels_all.append(self.train_labels[index])
        return train_images_all,train_labels_all,num_iter