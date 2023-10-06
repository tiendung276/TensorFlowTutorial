# Fashion MNIST Classification

This Python script is a simple implementation of a neural network for image classification on the Fashion MNIST dataset. The script uses TensorFlow and Keras for building and training the model.

## Overview

The script performs the following steps:

1. **Imports necessary libraries**: The required Python libraries such as TensorFlow, NumPy, and Matplotlib are imported.

2. **Loads the Fashion MNIST dataset**: The dataset is loaded directly from Keras datasets. The dataset is split into training and testing sets.

3. **Preprocesses the data**: The image data is normalized by dividing by 255.0 to convert pixel values to be between 0 and 1.

4. **Defines the model**: A sequential model is defined with a flatten layer to convert each image from a 2D array to 1D array, a dense layer with 128 nodes, and an output layer with 10 nodes (for the 10 classes of clothing).

5. **Compiles the model**: The model is compiled with the Adam optimizer, sparse categorical crossentropy as the loss function, and accuracy as the metric.

6. **Trains the model**: The model is trained on the training data for 6 epochs.

7. **Evaluates the model**: The model's performance is evaluated on the test data.

8. **Makes predictions**: The model makes predictions on the test data.

9. **Visualizes predictions**: The script uses Matplotlib to visualize the first 5 test images, their predicted labels, and the true labels.

