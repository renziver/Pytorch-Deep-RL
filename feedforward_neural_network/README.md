# Feedforward Neural Network

### Handwritten Digit Recognition

![Architecture of the feedforward neural network implemented](assets/images/ffnn.png)

#### Overview
The Feedforward Neural Network is composed of the following specifications:

Parameters | Value |
--- | --- |
Input size | 784 `(MNIST image size is 28x28, flattened to 784)` |
Number of Hidden layer| 1|
Hidden layer 1 size | 500 |
Output layer size | 10 `(MNIST has 10 types of digits)`|
Number of epochs | 10 |
Batch size | 100|
Learning rate | 0.001|
Activation Function | ReLU
#### Dataset
The Dataset used is the [MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist/) that is composed of 60,000 examples on the training set and 10,000 examples on the test set. The implementation directly used the [Pytorch Torchvision MNIST dataset](https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html).

#### Results
#### Usage