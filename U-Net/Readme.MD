U-Net Feature Extraction using PyTorch on Cat and Dog Dataset

This repository contains code for performing U-Net feature extraction using PyTorch on a cat and dog dataset. The code implements the U-Net architecture with an encoder and a classifier.

UNetEncoder

The UNetEncoder class represents the encoder part of the U-Net model. It consists of contracting layers that extract features from the input images. The encoder is initialized with a pre-trained U-Net model, and the forward method performs the forward pass through the encoder layers, returning the extracted features.

CatDogClassifier

The CatDogClassifier class represents the classifier part of the U-Net model. It takes an instance of the UNetEncoder as input and builds a classifier on top of it. The classifier consists of fully connected layers that map the extracted features to the number of classes. The forward method performs the forward pass through the encoder, flattens the features, and passes them through the classifier layers to obtain class predictions.

