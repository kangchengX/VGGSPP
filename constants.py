# configuration for the VGG network
CONV_CONFIG = {
    'vgg11': ((1, 64, True), (1, 128, True), (2, 256, True), (2, 512, True), (2, 512, False)),
    'vgg16': ((2, 64, True), (2, 128, True), (3, 256, True), (3, 512, True), (3, 512, False)),
    'vgg19': ((2, 64, True), (2, 128, True), (3, 256, True), (3, 512, True), (3, 512, False))
}