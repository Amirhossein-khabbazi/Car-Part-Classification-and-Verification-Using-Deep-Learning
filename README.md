# Car Part Classification and Verification Using Deep Learning

## Overview
This project focuses on car part classification and car verification using deep learning techniques. Leveraging the CompCars dataset, which consists of over 11,000 images of various car models and parts, we explore the performance of convolutional neural networks (CNNs) and Siamese neural networks (SNNs) on two key tasks:
- **Car Part Classification**: Identifying specific car components from images using ResNet50 and InceptionV3 CNN architectures.
- **Car Verification**: Determining if two images represent the same car using a Siamese neural network (SNN) with ResNet50 and MobileNetV2 as base models.

This project demonstrates the effectiveness of different models and loss functions, with a particular focus on focal loss to handle class imbalances.

## Key Features
- **Car Part Classification**:
  - Utilizes ResNet50 and InceptionV3 CNN architectures.
  - Compares sparse categorical crossentropy and focal loss functions.
  - ResNet50 with focal loss achieved a validation accuracy of 97.81%, while InceptionV3 with focal loss reached 98.82%.
- **Car Verification**:
  - Employs a Siamese Neural Network with ResNet50 and MobileNetV2 as base models.
  - ResNet50 with focal loss achieved the highest verification accuracy at 95.64%.
  - MobileNetV2, while faster, provided 86% accuracy, making it suitable for mobile or resource-limited applications.

## Dataset
The project uses the **CompCars dataset**, which contains 11,059 images across various car models and parts. The dataset is split into training, validation, and test sets for rigorous performance evaluation.

## Model Architectures
1. **ResNet50 and InceptionV3 for Classification**:
   - These CNNs are trained from scratch to classify specific car parts. Both architectures showed strong classification capabilities, with InceptionV3 providing marginally higher accuracy.
   - **Focal Loss** improved accuracy by handling class imbalances, especially in underrepresented categories.

2. **Siamese Neural Network for Verification**:
   - This network comprises two identical subnetworks that generate feature embeddings for each input image, which are compared using L1 distance to determine similarity.
   - **ResNet50** with focal loss outperformed MobileNetV2 in this task, making it ideal for applications where accuracy is prioritized.

## Results
- **Car Part Classification**:
  - ResNet50 with focal loss achieved 100% accuracy on the test set.
  - InceptionV3 with focal loss achieved 98.11% accuracy, misclassifying only one image.

- **Car Verification**:
  - ResNet50 with focal loss achieved 95.64% accuracy on the test set, outperforming other configurations.
  - MobileNetV2 provided a viable alternative for efficiency-focused applications.


