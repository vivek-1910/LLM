# Machine Learning Assignment - ANN and Logistic Regression

## Overview
This assignment demonstrates three machine learning classification techniques:
1. **Binary Classification using Artificial Neural Network (ANN)** - Heart Disease Prediction
2. **Multi-class Classification using ANN with Softmax** - Iris Flower Classification  
3. **Logistic Regression** - Student Pass/Fail Prediction

## Course Information
- **Course**: UE23AM342BB2 - Large Language Models and Their Applications
- **Due Date**: 20 Feb 2026
- **Institution**: [Your Institution Name]

---

## Part 1: Binary Classification ANN - Heart Disease Prediction

### File: `1_binary_classification_ann.py`

### Objective
Build an Artificial Neural Network to predict whether a patient has heart disease (binary classification).

### Dataset
- **Source**: Heart Disease (Cleveland Heart Disease Dataset)
- **Size**: 303 samples
- **Target**: Binary (0 = No disease, 1 = Disease present)
- **Features**: 13 medical features including age, sex, chest pain type, etc.

### Key Concepts Explained

#### 1. **Data Preprocessing**
```python
# Standardization: Scale features to have mean=0, std=1
# Why? Neural networks converge faster with scaled data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```
- **Importance**: Features with different scales can dominate learning
- **Formula**: x_scaled = (x - mean) / std_dev

#### 2. **Neural Network Architecture**
```
Input Layer (13 features)
    ↓
Hidden Layer 1: 64 neurons + ReLU + Dropout(0.3)
    ↓
Hidden Layer 2: 32 neurons + ReLU + Dropout(0.3)
    ↓
Hidden Layer 3: 16 neurons + ReLU
    ↓
Output Layer: 1 neuron + Sigmoid
```

**Activation Functions Explained:**
- **ReLU (Rectified Linear Unit)**: f(x) = max(0, x)
  - Advantages: Non-linear, computationally efficient, prevents vanishing gradient
  
- **Sigmoid**: f(x) = 1 / (1 + e^(-x))
  - Outputs probability between 0 and 1 for binary classification

- **Dropout**: Randomly deactivates neurons during training
  - Prevents overfitting by reducing co-adaptation

#### 3. **Loss Function: Binary Cross-Entropy**
```
Loss = -1/m * Σ(y * log(ŷ) + (1-y) * log(1-ŷ))
```
- Measures difference between true and predicted probabilities
- Suitable for binary classification with probabilistic outputs

#### 4. **Performance Metrics**
1. **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
   - Overall correctness of predictions
   
2. **Precision**: TP / (TP + FP)
   - Of predicted positive cases, how many were actually positive
   - Important when false positives are costly
   
3. **Recall**: TP / (TP + FN)
   - Of actual positive cases, how many we detected
   - Important when false negatives are costly (medical diagnosis)
   
4. **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
   - Harmonic mean of precision and recall
   
5. **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve
   - Measures discrimination ability across all classification thresholds

### Training Parameters
- **Optimizer**: Adam (Adaptive Moment Estimation)
  - Adapts learning rate for each parameter
  - Formula: m_t = β₁*m_(t-1) + (1-β₁)*g_t, v_t = β₂*v_(t-1) + (1-β₂)*g_t²
  
- **Batch Size**: 16 (number of samples before updating weights)
- **Epochs**: 100 (complete passes through entire dataset)
- **Validation Split**: 20% (reserve for validation during training)

---

## Part 2: Multi-class Classification ANN with Softmax

### File: `2_multiclass_classification_ann.py`

### Objective
Build an ANN to classify iris flowers into 3 species: Setosa, Versicolor, or Virginica.

### Dataset
- **Source**: Iris Dataset (built-in with scikit-learn)
- **Size**: 150 samples
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Features**: 4 (sepal length, sepal width, petal length, petal width)

### Key Differences from Binary Classification

#### 1. **One-Hot Encoding**
Instead of single output (0 or 1), we use 3 outputs representing each class:
```
Setosa:     [1, 0, 0]
Versicolor: [0, 1, 0]
Virginica:  [0, 0, 1]
```

#### 2. **Softmax Activation Function**
```
For output vector z:
softmax(z_i) = e^(z_i) / Σ(e^(z_j)) for all j

Properties:
- Outputs probability distribution (all sum to 1)
- Each output represents probability of belonging to that class
- Differentiable for backpropagation
```

**Why Softmax?**
- Converts raw scores to probabilities
- Ensures sum of probabilities = 1
- Allows proper probabilistic interpretation

#### 3. **Categorical Cross-Entropy Loss**
```
Loss = -Σ(y_true * log(y_pred))
```
- For multi-class problems with one-hot encoded targets
- Similar to binary cross-entropy but generalized to multiple classes

#### 4. **Network Architecture**
```
Input Layer (4 features)
    ↓
Hidden Layer 1: 64 neurons + ReLU + Dropout(0.25)
    ↓
Hidden Layer 2: 32 neurons + ReLU + Dropout(0.25)
    ↓
Hidden Layer 3: 16 neurons + ReLU
    ↓
Output Layer: 3 neurons + Softmax (probability distribution)
```

#### 5. **Multi-class Metrics**
- **Macro-averaging**: Average metrics across all classes equally
- **Micro-averaging**: Calculate metrics on total samples (same as accuracy)
- **Weighted-averaging**: Average weighted by class frequency

---

## Part 3: Logistic Regression

### File: `3_logistic_regression.py`

### Objective
Use logistic regression to predict whether a student passes or fails based on study habits.

### Dataset
- **Size**: 200 samples
- **Target**: Binary (0 = Fail, 1 = Pass)
- **Features**: 
  - Study hours (0-10)
  - Previous GPA (1.5-4.0)
  - Attendance percentage (50-100%)
  - Assignments completed (0-10)

### Key Concepts

#### 1. **Logistic Regression Model**
```
z = w₁*x₁ + w₂*x₂ + ... + wₙ*xₙ + b

P(y=1|x) = 1 / (1 + e^(-z)) = sigmoid(z)
```
- Linear combination of features transformed through sigmoid
- Outputs probability between 0 and 1

#### 2. **Logistic Loss Function**
```
Loss = -1/m * Σ(y*log(ŷ) + (1-y)*log(1-ŷ))
```
- Found using Maximum Likelihood Estimation
- Convex function (single global minimum)

#### 3. **Decision Boundary**
- Threshold at 0.5 by default (customizable)
- If P(y=1) > 0.5 → predict class 1 (Pass)
- If P(y=1) ≤ 0.5 → predict class 0 (Fail)

#### 4. **Feature Coefficients Interpretation**
- **Positive coefficient**: Feature increases probability of positive class
- **Negative coefficient**: Feature decreases probability
- **Magnitude**: Larger absolute value = stronger effect

#### 5. **Advantages of Logistic Regression**
- Simple and interpretable
- Fast to train
- Works well with linear relationships
- Provides probability estimates
- No hyperparameters to tune (much simpler than neural networks)

#### 6. **Disadvantages of Logistic Regression**
- Assumes linear relationship between features and log-odds
- May underperform on complex non-linear patterns
- Less flexible than neural networks

---

## Installation and Requirements

### Required Libraries
```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn
```

### Versions Used
- numpy >= 1.20
- pandas >= 1.3
- scikit-learn >= 0.24
- tensorflow >= 2.6
- keras (included with tensorflow)
- matplotlib >= 3.4
- seaborn >= 0.11

---

## How to Run

### Option 1: Run Individual Scripts
```bash
# Binary Classification
python 1_binary_classification_ann.py

# Multi-class Classification
python 2_multiclass_classification_ann.py

# Logistic Regression
python 3_logistic_regression.py
```

### Option 2: Run All at Once
```bash
python run_all_assignments.py
```

---

## Expected Output

### Console Output
Each script will print:
1. Dataset loading and exploration
2. Data preprocessing steps
3. Model architecture
4. Training progress
5. Test metrics (accuracy, precision, recall, F1, ROC-AUC)
6. Confusion matrix
7. Classification report

### Generated Visualizations
1. **1_binary_classification_results.png**
   - Training history (accuracy & loss)
   - Confusion matrix
   - ROC curve

2. **2_multiclass_classification_results.png**
   - Training history (accuracy & loss)
   - Confusion matrix
   - Softmax probability distributions

3. **3_logistic_regression_results.png**
   - Feature importance (coefficients)
   - Confusion matrix
   - ROC curve
   - Decision boundary

---

## Code Explanation

### Example 1: Standardization
```python
# Why standardize neural networks?
scaler = StandardScaler()  # Create scaler object
X_train_scaled = scaler.fit_transform(X_train)  # Fit to training data
X_test_scaled = scaler.transform(X_test)  # Apply same transformation

# Formula: x_scaled = (x - mean) / std_dev
# Example: If study_hours has mean=5, std=2
# Original value: 7
# Scaled value: (7 - 5) / 2 = 1.0
```

### Example 2: Dropout Regularization
```python
# Randomly deactivates neurons during training
layers.Dropout(0.3)  # Deactivate 30% of neurons

# During training: Only 70% of neurons are active (randomly selected)
# During inference: All neurons are active (scaled by 1/0.7)
# Effect: Reduces overfitting, forces networks to be redundant
```

### Example 3: Softmax Activation
```python
# For output vector z = [2.0, 1.0, 0.1]
import numpy as np

def softmax(z):
    e_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
    return e_z / e_z.sum()

z = np.array([2.0, 1.0, 0.1])
probs = softmax(z)  # [0.659, 0.242, 0.099]
# Sum = 1.0 (valid probability distribution)
# Class 0 has highest probability → predict class 0
```

### Example 4: Logistic Regression Coefficients
```python
# After training:
# study_hours coef: 0.85      → Positive: more study → higher pass probability
# prev_gpa coef: 0.60         → Positive: higher GPA → higher pass probability
# attendance coef: 0.45       → Positive: higher attendance → higher pass probability
# assignments_completed: 0.52 → Positive: more assignments → higher pass probability

# Interpretation: All features positively contribute to passing
# study_hours has strongest effect (0.85 > others)
```

---

## Performance Comparison

### Binary Classification (ANN)
- Typical Accuracy: 80-85%
- ROC-AUC: 0.85-0.90
- Best for: Complex non-linear patterns

### Multi-class Classification (ANN)
- Typical Accuracy: 95-98%
- Softmax → proper probabilistic outputs
- Best for: Multiple class problems with non-linear relationships

### Logistic Regression
- Typical Accuracy: 85-90%
- Pros: Simple, fast, interpretable
- Cons: Assumes linear relationship

---

## Important Notes for Viva

### Questions You Should Be Able to Answer

1. **What is the difference between ReLU and Sigmoid?**
   - ReLU: Non-linear, efficient, used in hidden layers
   - Sigmoid: Outputs probability [0,1], used for binary classification output

2. **Why use Dropout?**
   - Prevents overfitting by randomly deactivating neurons
   - Makes network more robust

3. **Explain Softmax activation?**
   - Converts outputs to probability distribution (sum = 1)
   - Essential for multi-class classification

4. **What does standardization do?**
   - Scales features to mean=0, std=1
   - Helps neural networks converge faster
   - Prevents features with large ranges from dominating

5. **Difference between ANN and Logistic Regression?**
   - ANN: Non-linear, multiple layers, complex patterns
   - Logistic Regression: Linear model, simple, interpretable

6. **What is one-hot encoding?**
   - Converts categorical labels to binary vectors
   - Required for softmax activation and categorical cross-entropy

7. **How to calculate accuracy from confusion matrix?**
   - Accuracy = (TP + TN) / (TP + TN + FP + FN)

---

## References

### Recommended Reading
- Chapter 4 & 6 from course textbook on Neural Networks
- https://www.geeksforgeeks.org/deep-learning/what-is-keras/
- https://keras.io/api/models/sequential/
- https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

### Datasets
- Heart Disease: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- Iris: Built-in with scikit-learn
- Student Performance: Synthetic (created for learning)

---

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'tensorflow'
**Solution**: 
```bash
pip install tensorflow
```

### Issue: CUDA/GPU errors with TensorFlow
**Solution**: Use CPU version (automatic if GPU CUDA not installed)

### Issue: RuntimeWarning about weights
**Solution**: Normal in neural network training, can be ignored

### Issue: Plots not showing
**Solution**: They are automatically saved as PNG files in the assignment directory

---

## Summary

✅ **Binary Classification ANN**: Predicts heart disease with multiple layers and ReLU/Sigmoid
✅ **Multi-class Classification ANN**: Classifies iris species with Softmax output layer  
✅ **Logistic Regression**: Predicts student pass/fail with interpretable coefficients

All three parts include:
- Complete data preprocessing
- Model architecture explanation
- Performance evaluation with multiple metrics
- Visualizations of results
- Detailed code comments

**Total Training Time**: ~2-3 minutes for all three problems
**Output**: 3 visualization PNG files + Console metrics

---

## Author Notes

This assignment demonstrates:
1. How to build and train neural networks with Keras
2. The importance of data preprocessing and standardization
3. How activation functions (ReLU, Sigmoid, Softmax) work
4. Multi-class vs binary classification differences
5. Classical ML approach (Logistic Regression) vs Deep Learning

Every line of code is commented to support understanding and explanation during evaluation.

---

**Last Updated**: February 2026
**Due Date**: February 20, 2026
