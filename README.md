# Handwritten Digit Recognition using Decision Trees and Random Forests

## Overview
This project compares Decision Trees (DT) and Random Forests (RF) for handwritten digit recognition using the MNIST dataset (https://www.kaggle.com/datasets/hojjatk/mnist-dataset).  
The aim is to evaluate both models based on:

- Classification accuracy  
- Confusion matrices  
- Training & testing time  
- Effect of hyperparameters  

The project is implemented in MATLAB for a machine learning coursework.

---

## Objectives
- Train and evaluate a Decision Tree classifier  
- Train and evaluate a Random Forest classifier  
- Compare model performance using standard metrics  
- Analyse why Random Forests tend to outperform single trees  
- Visualize and interpret results  

---

## Hypothesis

**H₁**: A Random Forest classifier will achieve significantly higher classification accuracy on handwritten digit recognition than a single Decision Tree classifier.

**H₀**: There is no significant difference in classification accuracy between Random Forest and Decision Tree models.

---

## How to Run
1. Open MATLAB and set the project folder as the working directory.
2. Run `main.m` to reproduce all experiments and figures.

---

## Results Summary

| Model                     | CV Accuracy | Train Accuracy | Test Accuracy | Train Time | CV Time    |
|---------------------------|-------------|----------------|---------------|------------|------------|
| Decision Tree             | 86.1%       | 90.7%          | 87.2%         | 3.0 s      | 53.7 s     |
| Random Forest (400 trees) | 96.9%       | 100.0%         | 97.1%         | 138.1 s    | 1114.8 s   |

---

## Why Random Forest Performs Better
- Reduces variance through averaging  
- Handles noisy features more effectively  
- Builds stronger decision boundaries  
