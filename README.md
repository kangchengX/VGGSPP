# VGGSPP
address: https://github.com/kangchengX/VGGSPP.git .

## Introduction
This repo implements a VGG network with SPP layer to enable the model to receive images with different shapes.

The model is then used to classify images of culture relics and compare different classification results under 4 differnt types of input image shapes - gray images with the same (height, width), gray images with different (height, width), BGR images with the same (height, width), BGR images with different (height, width).

## Use
### data
Images are stored in a folder named 'data'. The images for each class are store in a child folder seperately with the class name as the folder name.

wavelet.py could be used to reduce noise of the images. 

### train and test the model
Run the main.py

## Other
This project is my first project related to computer vision and deep learning. Although some code or ideas might look naive now, this repo is meaningful and invaluable for the experience in deep learning from scratch.