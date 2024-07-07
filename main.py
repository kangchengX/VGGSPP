import tensorflow as tf
import process
from data import DataLoader


if __name__ == '__main__':
    # check GPU avaliablity
    if tf.config.list_physical_devices('GPU'):
        print('Using GPU')
    else:
        print('Using CPU')
    
    # configurations
    train_test_config = {
        'batch_size' : 16,
        'epoch' : 20,
        'momentum' : 0.8,
        'learning_rate' : 0.01
    }

    model_config = {
        'conv_arch': 'vgg16',
        'levels': [1,2,4], 
        'num_classes': 4
    }

    data_loader = DataLoader('bgr_mul')
    data_loader.load('data')

    process.train_test(data_loader=data_loader, model_config=model_config, **train_test_config)