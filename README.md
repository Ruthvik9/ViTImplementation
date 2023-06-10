# ViTImplementation
Implementation of the Vision Transformer from scratch based on the "An image is worth 16X16 words" paper

This repository contains a PyTorch implementation of the Vision Transformer (ViT), a transformer model designed for image classification tasks. This model was proposed in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al.

## Overview
The Vision Transformer (ViT) model is a transformer-based model that views an image as a sequence of flat patches and processes these patches with a transformer encoder. Unlike traditional Convolutional Neural Networks (CNNs) that use convolutions to process image data, ViT relies entirely on self-attention mechanisms.

The key features of the Vision Transformer include:

Patch-based tokenization: The image is divided into small patches (e.g., 16x16 pixels), which are linearly transformed (flattened) to create a sequence of image tokens.

Class token: An additional token is added at the beginning of the sequence, which collects global information about the image through multiple layers of transformer encodings. This class token is used in the final classification layer to predict the image class.

Positional embeddings: Since the transformer model doesn't natively handle the order of the input sequence, positional embeddings are added to the image tokens to provide information about the relative or absolute position of the patches in the image.

Transformer encoder: The core of the ViT model is a sequence of transformer encoders. Each transformer encoder consists of a multi-head self-attention mechanism and a multi-layer perceptron (MLP), with residual connections and layer normalization applied at each stage.

The implementation in this repository includes a custom PyTorch module for the Vision Transformer, with specific classes for the transformer encoder and the overall model architecture.

## Code Structure
The code in this repository is structured as follows:

ViTEncoder: This is a PyTorch module for a single transformer encoder in the Vision Transformer model. It includes a multi-head self-attention mechanism and a two-layer MLP with GELU activation.
ViTClassifier: This is the main Vision Transformer model. It takes in a batch of images, processes them into patches, and passes them through a series of transformer encoders. The output of the final encoder is passed through a classification head to predict the image classes.
Training Loop: The training loop function trains the model on the CIFAR-100 dataset using the specified loss function and optimizer. 

## References
Original ViT paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
PyTorch Documentation
[CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar
