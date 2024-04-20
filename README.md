# VGGSPP
address: https://github.com/kangchengX/VGGSPP.git .

## Introduction
This repo implements a VGG network with SPP layer to make the model receive images with different shapes.

The model is then used to classify images of culture relics and compare different classification results under differnt images types - gray images with the same shape, gray images with different shapes, bgr images with the same shape, bgr images with different shapes.

## Use
### data
Images are store in a folder called 'data'. The images for each class are store in a child folder seperately.

wavelet.py could be used to reduce noise of the images. 

### train
Run the main.py

## Other
This project is my first project related computer vision and deep learning. Although some code or ideas might naive now, this repo is meaningful and invaluable for this experience in deep learning from scratch.