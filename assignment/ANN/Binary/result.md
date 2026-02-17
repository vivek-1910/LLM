# Binary Classification - Results

## Model Performance

### Dataset
- **Source**: UCI ML Repository - Heart Disease (Cleveland)
- **Size**: 297 patient records
- **Features**: 13 medical attributes
- **Target**: Binary (0 = No disease, 1 = Disease present)
- **Train-Test Split**: 80-20

### Model Architecture
```
Input (13) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) 
→ Dropout(0.3) → Dense(16, ReLU) → Dense(1, Sigmoid)
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.8167 (81.67%) |
| **Precision** | 0.8400 (84.00%) |
| **Recall** | 0.7500 (75.00%) |
| **F1-Score** | 0.7925 |
| **ROC-AUC** | 0.9297 |

### Confusion Matrix
```
True Negatives:  28
False Positives: 4
False Negatives: 7
True Positives:  21
```

### Interpretation
- **Recall (75.00%)**: Detected 21 out of 28 actual disease cases
- **Precision (84.00%)**: 21 out of 25 predicted disease cases were correct
- **Trade-off**: Higher recall prioritizes disease detection

### Training Details
- **Epochs**: 100
- **Batch Size**: 16
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Validation Split**: 20%
- **Regularization**: Dropout (30%)

### Key Findings
✅ Network converges well  
✅ Dropout prevents overfitting  
✅ High recall prioritizes disease detection  
✅ Features properly standardized  

### Files Generated
- `results.png` - Visualizations
- `result.md` - This metrics report
- `heart_disease.csv` - Dataset
- `binary_classification.py` - Script

---
**Generated**: 2026-02-16 12:00:48
