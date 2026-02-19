# Logistic Regression

This folder contains the logistic regression classification model using scikit-learn.

## 📁 Structure

```
LR/
├── logistic_regression.py        # Logistic regression model
├── mushroom_classification.csv   # Binary classification dataset
├── download_dataset.py           # Generate/download dataset
└── results.png                   # Visualization of results
```

## 📊 Model

### Logistic Regression
- **File**: `logistic_regression.py`
- **Task**: Binary classification on synthetic dataset
- **Dataset**: Synthetic Binary Classification (1500 samples, 12 features)
- **Algorithm**: Linear classifier with logistic function
- **K-Fold Accuracy**: 95.5% ± 0.81%
- **Interpretable**: Shows feature importance through coefficients

## 🔧 Requirements

```bash
pip install -r ../requirements.txt
```

## 🚀 Run from Assignment Root

```bash
# Generate dataset
python LR/download_dataset.py

# Run logistic regression
python LR/logistic_regression.py

# Run all models
python run_all.py
```

## 📚 Key Concepts

### Logistic Regression Model
- **Equation**: P(y=1|x) = 1 / (1 + e^(-(w·x + b)))
- **Order**: 1st order polynomial (linear)
- **Optimization**: Maximum Likelihood Estimation

### Advantages of Logistic Regression
✅ Simple and interpretable  
✅ Fast to train  
✅ Provides probability estimates  
✅ No hyperparameters to tune  
✅ Good baseline model  

### Disadvantages
❌ Assumes linear relationship  
❌ May underperform on complex non-linear patterns  
❌ Less flexible than neural networks

### Feature Coefficients
- **Positive coefficient**: Feature increases probability of positive class
- **Negative coefficient**: Feature decreases probability
- **Magnitude**: Larger absolute value = stronger effect

### Performance Metrics
- **Accuracy**: Percentage of correct predictions
- **Precision**: Of predicted positive, how many are correct
- **Recall**: Of actual positive, how many we predicted
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (discrimination ability)

## 🔄 Comparison: Logistic Regression vs ANN

| Aspect | Logistic Regression | ANN |
|--------|-------------------|-----|
| Complexity | Simple, linear | Complex, non-linear |
| Training Time | <1 second | 30+ seconds |
| Interpretability | High (coefficients) | Low (black box) |
| Performance | Good on linear data | Excellent on complex data |
| Overfitting | Less likely | Needs regularization |
| Hyperparameters | None | Many |
