<p align="center">
    <h1 align="center">VGGSPP</h1>
</p>
<p align="center">
    <em>Image classification for images with flexible (different) shapes</em>
</p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=default&logo=TensorFlow&logoColor=white" alt="TensorFlow">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
</p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Modules](#modules)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Data](#data)
  - [Usage](#usage)
</details>
<hr>

##  Overview

The VGGSPP project implements [SPP](https://arxiv.org/abs/1406.4729) layer with VGG to enable the model receive **images with different shapes** for classification tasks. A training class is also implemented to train the model with inputs with different shapes. Other features include support for noise reduction using wavelet expecially for self-built dataset. 

---

##  Repository Structure

```sh
└── VGGSPP/
    ├── config.py
    ├── data.py
    ├── display_tools.py
    ├── main.py
    ├── models.py
    ├── process.py
    ├── README.md
    ├── requirements.txt
    └── wavelet.py
```

---

##  Modules

<details open><summary>.</summary>

| File                                 | Summary |
| ---                                  | --- |
| [config.py](config.py)               | Defines configurations for different variations of the VGG neural network architectures (VGG11, VGG16, and VGG19). |
| [data.py](data.py)                   | Manages the loading, processing, and partitioning of image datasets into training and test sets, supporting different image formats and sizes, with functionalities for normalization, resizing, and shuffling to optimize model training and evaluation within the VGGSPP project architecture.                                              |
| [display_tools.py](display_tools.py) | Provides visualization functions for training metrics, specifically plotting loss and accuracy.                            |
| [main.py](main.py)                   | Orchestrates training and testing of the VGGSPP model, handling configurations through command-line arguments and GPU checks. It initializes data loading, model setup, and execution flow.             |
| [models.py](models.py)               | Integrates advanced neural network structures, including custom layers and models, for image processing. Features include Spatial Pyramid Pooling and multiple convolution blocks, configurable for varied architectures like VGG, enhancing adaptability in handling different image resolutions and classifications within the project.     |
| [process.py](process.py)             | Facilitates training, testing, and evaluation of the VGGSPP model, managing model operations like parameter optimization, metric tracking, and data handling through the DataLoader. It supports customizable training cycles, model saving, and performance logging, integrating visual loss and accuracy assessment tools.                  |
| [requirements.txt](requirements.txt) | Specifies the necessary libraries for the VGGSPP project.               |
| [wavelet.py](wavelet.py)             | Wavelet.py provides image denoising functionality through wavelet transform techniques, crucial for enhancing image quality by reducing noise while maintaining key features, which is expecially useful for self-built dataset.         |

</details>

---

##  Getting Started

**System Requirements:**

* **Python**: `version 3.12.2`

###  Installation

<h4>From <code>source</code></h4>

> 1. Clone the repository:
>
> ```console
> $ git clone https://github.com/kangchengX/VGGSPP.git
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd VGGSPP
> ```
>
> 3. Install the dependencies:
> ```console
> $ pip install -r requirements.txt
> ```

### Data
The Data folder should have the following structure:

```sh
└── data/
    ├── classname1
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── classname2
    └── ...
```
, i.e., the `data` folder has child folders with class names as names, each of which contains the images of the correponding class.

The width and height of the images don't need to be the same.

[wavelet.py](wavelet.py) could be used to denoise the images (this is not necessary).

###  Usage

<h4>From <code>source</code></h4>

> Use the command below:
> ```console
> $ python main.py [OPTIONS]
> ```

| Option | Type | Description | Default Value |
|--------|------|-------------|---------------|
| --conv_arch | String | Configuration for VGG part. | `vgg16` |
| --levels | List of Integers | A list of integers for levels of the SPP layer. | `[1, 2, 3]` |
| --num_classes | Integer | Number of classes. If not set, the number of classes will be inferred from the data folder. | `None` |
| --images_shapes_type | String | Shape of the images, can be bgr_sin, bgr_mul, gray_sin, gray_mul. | `None` |
| --image_size | Integer | The size of the images if the images have the same shape. | `224` |
| --data_folder | String | Folder of the data, which contains child folders as classes, each of which contains images of the class. | `data` |
| --batch_size | Integer | Batch size. | `32` |
| --num_epochs | Integer | Number of epochs. | `40`|
| --learning_rate | Float | Learning rate. | `0.01` |
| --save_folder | String | Folder to save the results. If not given, the folder would be images_shape_type. | `None` |
| --save_model | Boolean | Whether to save the model after all epochs. | `False` |
| --save_model_epochs | Integer | If not None, the model will be saved every save_model_epochs of epochs. | `None` |

<h4>Example:</h4>

>```console
> $ python main.py --levels 1 2 4 --images_shapes_type gray_mul
> ```
