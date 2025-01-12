# **README for Veggie CNN**
## **VeggieCNN**
## Introduction
This repository contains a simple Convolutional Neural Network (CNN) built using PyTorch to classify six different types of vegetables from a curated image dataset. The model architecture includes:
+ **2 Convolutional Layers:** For feature extraction
+ **2 Pooling Layers:** For down-sampling.
+ **2 Fully Connected Linear Layers:** For final classification.
  
The activation function used is ReLU throughout the network.

I created this after watching a 5 hour PyTorch Youtube course and wanted to find a basic way of applying those skills I gained to something possibly useful. 

Feel free to check out the code and dataset **[(link)](https://www.kaggle.com/datasets/jocelyndumlao/a-dataset-for-vegetable-identification/data)**, and let me know your thoughts or suggestions for improvement!
### **Experimented with different**
+ Learning rates
+ Optimizers
+ Epoch Sizes
+ Batch Sizes
## Results
### With 2 Epochs, batch size of 128, and a learning rate of 0.001 the model achieved...
+ Accuracy of the network: 98.04 %
+ Accuracy of Augmented Beans: 96.88 %
+ Accuracy of Augmented Ladies finger: 98.56 %
+ Accuracy of Augmented Onion: 95.44 %
+ Accuracy of Augmented Pointed gourd: 98.80 %
+ Accuracy of Augmented Potato: 98.80 %
+ Accuracy of Augmented eggplant: 99.76 %
### With 3 Epochs, batch size of 128, and a learning rate of 0.001 the model achieved...
+ Accuracy of the network: 99.0 %
+ Accuracy of Augmented Beans: 99.28 %
+ Accuracy of Augmented Ladies finger: 99.76 %
+ Accuracy of Augmented Onion: 98.80 %
+ Accuracy of Augmented Pointed gourd: 100.00 %
+ Accuracy of Augmented Potato: 98.32 %
+ Accuracy of Augmented eggplant: 97.84 %

## Added weird functionality for extremely basic HTML
+ Can use flask and the command flask run to run on your local system through the main.py script but you would have to redownload the neural network model from the veggienet.py commented code at the end of main().
