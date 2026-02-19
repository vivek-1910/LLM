# Logistic Regression - Results

## Model Performance

### Dataset
- **Source**: Synthetic Binary Classification Dataset
- **Size**: 1500 samples
- **Features**: 12 continuous features
- **Target**: Binary (0=Negative, 1=Positive)
- **Train-Test Split**: 80-20 (1200 train, 300 test)
- **Class Balance**: ~50/50 split

### Model
```
Linear Classifier: P(y=1|x) = 1 / (1 + e^(-(w·x + b)))
Algorithm: Logistic Regression with Maximum Likelihood Estimation
Solver: LBFGS
```

### Performance Metrics - K-Fold Cross-Validation (Reliable)

| Metric | Mean ± Std |
|--------|-----------|
| **Accuracy** | 0.9550 ± 0.0081 |
| **Precision** | 0.9513 ± 0.0135 |
| **Recall** | 0.9604 ± 0.0168 |
| **F1-Score** | 0.9556 ± 0.0080 |

### Performance Metrics - Single Test Set

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.9633 (96.33%) |
| **Precision** | 0.9434 (94.34%) |
| **Recall** | 0.9868 (98.68%) |
| **F1-Score** | 0.9646 |
| **ROC-AUC** | 0.9840 |

### Confusion Matrix (Test Set: 300 samples)
```
True Negatives:  139
False Positives: 9
False Negatives: 2
True Positives:  150
```

### Feature Coefficients (Top Predictors by Magnitude)

| Feature | Coefficient |
|---------|-------------|
| feature_0 | 1.5831 |
| feature_1 | -1.9422 |
| feature_2 | -0.9860 |
| feature_3 | -0.0146 |
| feature_4 | -2.1465 |
| feature_5 | 0.0116 |
| feature_6 | 0.6655 |
| feature_7 | -0.1336 |
| feature_8 | 0.2091 |
| feature_9 | 2.7122 |
| feature_10 | -2.5722 |
| feature_11 | 0.3162 |
| Intercept | -0.3214 |

### Training Details
- **Model Type**: Binary Logistic Regression
- **Features Standardized**: Yes (StandardScaler)
- **Training Time**: <1 second
- **Max Iterations**: 1000
- **Validation Method**: 5-Fold Stratified Cross-Validation
- **Dataset Size**: 1500 samples with 12 features

### Insights
✅ **Realistic Performance**: K-Fold CV mean accuracy of 0.9550 reflects true model performance

✅ **Complex Feature Space**: 12 features with 1500 samples for robust classification

✅ **Well-Separated Classes**: Tight clusters (1 per class) = easier classification

✅ **Model Works Well**: LR handles 12-feature classification effectively

✅ **Low Label Noise**: 2% label noise ensures dataset quality

### Files Generated
- `results.png` - Visualizations (feature importance, confusion matrix, ROC curve, decision boundary)
- `result.md` - This metrics report
- `mushroom_classification.csv` - Dataset (synthetic, 1000 samples)
- `logistic_regression.py` - Script

---
**Generated**: 2026-02-19 09:38:32
