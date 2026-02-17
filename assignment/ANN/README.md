# ANN - Artificial Neural Networks

This folder contains two ANN classification models using Keras:

## 📁 Structure

```
ANN/
├── Binary/
│   ├── binary_classification.py    # Binary classification model
│   └── results.png                 # Visualization of results
│
└── Multiclass/
    ├── multiclass_classification.py # Multiclass classification model
    └── results.png                  # Visualization of results
```

## 📊 Models

### Binary Classification
- **File**: `Binary/binary_classification.py`
- **Task**: Predict heart disease (Yes/No)
- **Dataset**: Heart Disease (297 real patient records)
- **Output**: 1 neuron with Sigmoid activation
- **Accuracy**: ~57%

**Run**: `python Binary/binary_classification.py`

### Multiclass Classification (with Softmax)
- **File**: `Multiclass/multiclass_classification.py`
- **Task**: Classify iris flowers (3 species)
- **Dataset**: Iris Flowers (150 real flower samples)
- **Output**: 3 neurons with Softmax activation
- **Accuracy**: ~93%

**Run**: `python Multiclass/multiclass_classification.py`

## 🔧 Requirements

```bash
pip install -r ../requirements.txt
```

## 🚀 Run from Assignment Root

```bash
# Run binary classification
python ANN/Binary/binary_classification.py

# Run multiclass classification
python ANN/Multiclass/multiclass_classification.py

# Run all models
python run_all.py
```

## 📚 Key Concepts

### Sigmoid Activation (Binary)
- Used in output layer for binary classification
- Output: probability between 0 and 1
- Formula: f(x) = 1 / (1 + e^(-x))

### Softmax Activation (Multiclass)
- Used in output layer for multiclass classification
- Outputs: probability distribution (sum = 1)
- Formula: softmax(z_i) = e^(z_i) / Σ(e^(z_j))

### ReLU Activation (Hidden Layers)
- Non-linear activation function
- Formula: f(x) = max(0, x)
- Prevents vanishing gradient problem

### Dropout Regularization
- Randomly deactivates neurons during training
- Prevents overfitting
- Improves model generalization
