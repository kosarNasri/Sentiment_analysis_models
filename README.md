# Sentiment Analysis on Digikala Comments

This project focuses on sentiment analysis of comments from Digikala, a popular e-commerce platform. We implemented two separate models: Convolutional Neural Networks (CNN) and a pre-trained transformer model from Huggingface. Each model was evaluated for its performance, and their respective accuracies were recorded.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models Used](#models-used)
  - [CNN Models](#cnn-models)
  - [Huggingface Model](#huggingface-model)                              
- [Reasons for Model Choices](#reasons-for-model-choices)
- [Advantages of Each Model](#advantages-of-each-model)
- [References](#references)

## Introduction

Sentiment analysis is a crucial task in natural language processing (NLP) with applications in customer feedback, brand monitoring, and more. This project aims to analyze sentiments expressed in comments on Digikala, leveraging the power of both deep learning (CNN) and transformer models (Huggingface).

## Dataset

The dataset consists of comments from Digikala, which have been labeled with sentiment scores. The data was preprocessed to remove noise and prepare it for input into our models.

## Models Used

### CNN Models

We implemented two separate Convolutional Neural Network (CNN) models for sentiment analysis. CNNs are effective for text classification tasks due to their ability to capture local features through convolutional filters.

1. **CNN Model 1**: A simple CNN architecture with an embedding layer, followed by convolutional and pooling layers, and a fully connected output layer.
2. **CNN Model 2**: An enhanced CNN with additional convolutional layers and dropout for regularization.

### Huggingface Model

I used a pre-trained transformer model from Huggingface, specifically designed for NLP tasks. Transformers have revolutionized NLP by enabling models to understand context and semantics more deeply.


## Reasons for Model Choices

1. **CNN Models**:
   - **Local Feature Extraction**: CNNs are known for their ability to capture local patterns in text data.
   - **Efficiency**: CNNs are computationally less intensive compared to transformers, making them faster for training and inference on large datasets.
   - **Previous Success**: CNNs have been successful in various text classification tasks, providing a solid baseline for our experiments.

2. **Huggingface Model**:
   - **Contextual Understanding**: Transformer models like BERT excel at understanding context and the relationship between words in a sentence.
   - **Pre-trained Power**: Using a pre-trained model allows leveraging large-scale training on diverse datasets, improving performance on smaller, domain-specific datasets like ours.
   - **Flexibility**: Transformers can be fine-tuned for specific tasks, making them versatile for various NLP applications.

## Advantages of Each Model

1. **CNN Models**:
   - **Speed**: Faster training and inference times.
   - **Simplicity**: Easier to implement and require fewer resources.
   - **Effective for Smaller Data**: Performs well on smaller datasets where training a transformer might not be feasible.

2. **Huggingface Model**:
   - **Accuracy**: Generally higher accuracy due to better contextual understanding.
   - **Robustness**: More robust to variations in text due to extensive pre-training.
   - **Transfer Learning**: Ability to transfer learning from large datasets to our specific domain.


## References

- [Huggingface Transformers](https://huggingface.co/transformers/)
