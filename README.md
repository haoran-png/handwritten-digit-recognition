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

## Methods

### Decision Tree
- **Split criterion:** Gini / Entropy  
- **Key parameters:**  
  - Max depth  
  - Min leaf size  
- **Pros:** interpretable, fast  
- **Cons:** high variance, prone to overfitting  

### Random Forest
- **Approach:** ensemble of multiple decision trees  
- **Features:** random feature subsets, bootstrap sampling  
- **Key parameters:**  
  - Number of trees  
  - Max depth  
- **Pros:** robust, higher accuracy, reduces overfitting  
- **Cons:** slower, less interpretable  

---

## Hypothesis

**H₁**: A Random Forest classifier will achieve significantly higher classification accuracy on handwritten digit recognition than a single Decision Tree classifier.

**H₀**: There is no significant difference in classification accuracy between Random Forest and Decision Tree models.

---

## Results Summary
*(Replace with your actual numbers)*

| Model                     | CV Accuracy | Train Accuracy | Test Accuracy | Train Time | CV Time     |
|---------------------------|-------------|----------------|---------------|------------|-------------|
| Decision Tree             | 86.1%       | 90.7%          | 87.2%         | 2.96 s     | 53.72 s     |
| Random Forest (400 trees) | 96.9%       | 100.0%         | 97.1%         | 138.15 s   | 1114.76 s   |

---

## Why Random Forest Performs Better
- Reduces variance through averaging  
- Handles noisy features more effectively  
- Builds stronger decision boundaries  
