from typing import Literal


def get_config(config_name: Literal['vgg11', 'vgg16', 'vgg19']):
    """
    Get (VGG) config.
    
    Args:
        config_name (str): the configuration for the VGG network, can be `'vgg11'`, `'vgg16'`, or `'vgg19'`.

    Returns:
        the configuration.
    """
    if config_name == 'vgg11':
        return ((1, 64, True), (1, 128, True), (2, 256, True), (2, 512, True), (2, 512, False))
    elif config_name == 'vgg16':
        return ((2, 64, True), (2, 128, True), (3, 256, True), (3, 512, True), (3, 512, False))
    elif config_name == 'vgg19':
        return ((2, 64, True), (2, 128, True), (3, 256, True), (3, 512, True), (3, 512, False))
    else:
        raise ValueError(f'Unsupported config_name : {config_name}')