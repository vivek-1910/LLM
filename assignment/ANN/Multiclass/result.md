# Multiclass Classification - Results

## Model Performance

### Dataset
- **Source**: UCI ML Repository - Iris Flower Dataset
- **Size**: 150 flower samples
- **Features**: 4 measurements (sepal/petal length and width)
- **Target**: Multi-class (0=Setosa, 1=Versicolor, 2=Virginica)
- **Train-Test Split**: 80-20

### Model Architecture
```
Input (4) → Dense(64, ReLU) → Dropout(0.25) → Dense(32, ReLU)
→ Dropout(0.25) → Dense(16, ReLU) → Dense(3, Softmax)
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 0.9333 (93.33%) |
| **Precision** | 0.9333 (93.33%) |
| **Recall** | 0.9333 (93.33%) |
| **F1-Score** | 0.9333 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Setosa | 100% | 100% | 100% |
| Versicolor | 90% | 90% | 90% |
| Virginica | 90% | 90% | 90% |

### Confusion Matrix
```
        Predicted
        S  V  V_i
Actual S [10  0   0]
       V [ 0  9   1]
       Vi [ 0  1   9]
```

### Key Findings
✅ Perfect Setosa classification (100%)  
✅ Strong performance across all classes  
✅ Softmax produces valid probability distributions  
✅ One-hot encoding works correctly  

### Training Details
- **Epochs**: 100
- **Batch Size**: 8
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Validation Split**: 20%
- **Regularization**: Dropout (25%)
- **Output Activation**: Softmax (3-class probability)

### Files Generated
- `results.png` - Visualizations (training history, confusion matrix, probabilities)
- `result.md` - This metrics report
- `iris_data.csv` - Dataset
- `multiclass_classification.py` - Script

---
**Generated**: 2026-02-16 12:01:27
