import tensorflow as tf
import process
from data import DataLoader
from models import VGGSPP
import argparse, os, warnings, json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model config related
    parser.add_argument('--conv_arch', type=str, default='vgg16',
                        help='Configuration for VGG part.')
    parser.add_argument('--levels', nargs='+', type=int, default=[1, 2, 3],
                        help='A list of integers for levels of the SPP layer')
    parser.add_argument('--num_classes', type=int,
                        help='number of classes. If not set a value, the number of classes will be inferred from  the data folder.')
    
    # data related
    parser.add_argument('--images_shapes_type', type=str,
                        help='shape of the images, can be bgr_sin, bgr_mul, gray_sin, gray_mul')
    parser.add_argument('--image_size', type=int, default=224,
                        help='the size of the images if the images have the same shape')
    parser.add_argument('--data_foler', type=str, default='data',
                        help='folder of the data, which contains child folders as classes, each of which contains images of the class')
    
    # training related
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size. Default to 32.')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='number of epochs. Default to 40.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate. Default to 0.01')

    
    # files saving related
    parser.add_argument('--save_folder', type=str,
                        help='folder to save the results. if not given, the folder would be images_shape_type')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='whether to save the model')
    parser.add_argument('--save_model_epochs', type=int,
                        help='If not None, the model will be saved every save_model_epochs of epochs. Default to None')

    args = parser.parse_args()

    # check GPU avaliablity
    if tf.config.list_physical_devices('GPU'):
        print('Using GPU')
    else:
        print('Using CPU')

    data_loader = DataLoader(args.images_shapes_type, args.image_size)
    data_loader.load('data')

    if args.num_classes is None:
        args.num_classes = len(data_loader.classes)

    if args.save_folder is None:
        args.save_folder = args.images_shapes_type

    if os.path.exists(args.save_folder):
        warnings.warn(f"folder {args.save_folder} already exists")
    else:
        os.makedirs(args.save_folder)
    
    model = VGGSPP(conv_arch=args.conv_arch, levels=args.levels, num_classes=args.num_classes)

    with open(os.path.join(args.save_folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    processor = process.Processor(
        model=model,
        data_loader=data_loader,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        folder=args.save_folder
    )

    processor.train_and_test(
        save_model_epochs=args.save_model_epochs,
        save_model=args.save_model,
        show_plot=False,
        save_filename=os.path.join(args.save_folder, 'loss and accuracy.png')
    )