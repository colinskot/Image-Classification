# Image-Classification

A Convolutional Neural Network (CNN) is trained to identify 10 different classes of images on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.

## Getting Started

Simply run the Jupyter Notebook dlnd_image_classification.ipynb or you can run the script image_classification.py

```
python image_classification.py
```

### Prerequisites

You can install the required packages through Anaconda's environment manager using the machine-learning.yml file

```
conda env create -f machine-learning.yml
```

Then, activate the environment and run image_classification.py

```
activate machine-learning
```

Otherwise, check out the machine-learning.yml file for dependencies and their versions

## Running the tests

Simply add test cases to problem_unittests.py or run it

```
python problem_unittests.py
```

## Built With

* [TensorFlow](https://www.tensorflow.org/install/install_windows) - The machine learning framework
* [Anaconda](https://repo.continuum.io/archive/Anaconda3-5.1.0-Windows-x86_64.exe) - The environment manager
* [Jupyter Notebook](http://jupyter.org/install) - The code documentation
