import matplotlib.pyplot as plt
import os
from typing import List

def show_loss_plot(
        loss_results: List[float], 
        accuracy_results: List[float], 
        images_shapes_type: str,
        save_filename: str | None = None,
        show_plot: bool | None = True
):
    """
    Visulize loss and accuracy.
    
    Args:
        loss_results (list): list of loss values.
        accuracy_results (list): list of accuracy values.
        images_shapes_type (str): the type of the shape of input images, can be: \
            `'gray_sin'`: gray channel and images have the same (height, width); \
            `'gray_mul'`: gray channel and images have different (height, width); \
            `'bgr_sin'`: BGR channels and images have the same (height, width); \
            `'bgr_mul'`: BGR channels and images have different (height, width).
        save_plot (bool): True indicates saving the polt with filename `images_shapes_type + '-' + 'show_loss.jpg'`.
        show_plot (bool): True indicates showing the figure.

    Return:
        fig (Figure): Figure of this figure.
    """
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(loss_results)
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(accuracy_results)
    
    if save_filename is not None:
        plt.savefig(os.path.splitext(save_filename)[0] + '.png')

    if show_plot:
        plt.show()

    return fig