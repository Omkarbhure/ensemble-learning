# Handwritten Digit Recognition Using Ensemble Learning

## Overview

This project recognizes handwritten digits using an ensemble of multiple machine learning classifiers.

The system combines:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

using a Soft Voting Ensemble.

## Objective

The objective is to improve the robustness of handwritten digit classification by combining predictions from multiple machine learning models.

## Dataset

The project uses the MNIST handwritten digit dataset.

- 70,000 images
- 10 classes
- Digits 0-9
- Image size: 28 × 28 pixels
- 784 pixel features

## Architecture

```text
User Image
    ↓
Image Preprocessing
    ↓
28 × 28 Grayscale Image
    ↓
784 Features
    ↓
┌───────────────┬──────────────┬─────────────┐
│   Logistic    │    Random    │     SVM     │
│  Regression   │    Forest    │             │
└───────────────┴──────────────┴─────────────┘
                    ↓
             Voting Ensemble
                    ↓
             Final Prediction