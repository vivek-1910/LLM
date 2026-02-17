# Logistic Regression - Results

## Model Performance

### Dataset
- **Source**: UCI ML Repository - Wine Classification Dataset
- **Size**: 178 wine samples
- **Features**: 6 chemical properties
- **Target**: Binary (0=Poor Quality, 1=Good Quality)
- **Train-Test Split**: 80-20

### Model
```
Linear Classifier: P(y=1|x) = 1 / (1 + e^(-(w·x + b)))
Algorithm: Logistic Regression with Maximum Likelihood Estimation
Solver: LBFGS
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 1.0000 (100.00%) |
| **Precision** | 1.0000 (100.00%) |
| **Recall** | 1.0000 (100.00%) |
| **F1-Score** | 1.0000 |
| **ROC-AUC** | 1.0000 |

### Confusion Matrix
```
True Negatives:  12
False Positives: 0
False Negatives: 0
True Positives:  24
```

### Feature Coefficients (Interpretability)

| Feature | Coefficient |
|---------|-------------|
| alcohol | -1.0707 |
| malic_acid | -0.2964 |
| ash | -0.3771 |
| magnesium | 0.1337 |
| phenols | -1.3452 |
| proline | -2.5765 |
| Intercept | 1.7588 |

### Training Details
- **Model Type**: Binary Logistic Regression
- **Features Standardized**: Yes (StandardScaler)
- **Training Time**: <1 second
- **Max Iterations**: 1000

### Key Findings
✅ Perfect accuracy (100.0%)  
✅ Highly interpretable coefficients  
✅ Fast training (<1 second)  
✅ All features properly scaled  
✅ Linear model sufficient for this data  

### Files Generated
- `results.png` - Visualizations (feature importance, confusion matrix, ROC curve, decision boundary)
- `result.md` - This metrics report
- `student_performance.csv` - Dataset
- `logistic_regression.py` - Script

---
**Generated**: 2026-02-16 12:12:47
