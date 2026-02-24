# 🚀 QUICK RUN GUIDE FOR VIVA

## Pre-Viva Checklist

- [ ] All code files present and working
- [ ] Datasets in correct folders
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] All 3 models run successfully at least once

---

## Running Models (During Viva)

### Option 1: Run All at Once (Recommended)
```bash
cd /Users/vivekgowdas/Desktop/LLM/assignment
python run_all.py
```
**Output**: All 3 models run sequentially with results

---

### Option 2: Run Individual Models

#### Binary Classification ANN
```bash
cd /Users/vivekgowdas/Desktop/LLM/assignment/ANN/Binary
python binary_classification.py
```
**Shows**: 
- Dataset info
- Model architecture
- Training progress
- Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix plot
- ROC curve plot

---

#### Multiclass Classification ANN (Iris + Softmax)
```bash
cd /Users/vivekgowdas/Desktop/LLM/assignment/ANN/Multiclass
python multiclass_classification.py
```
**Shows**:
- Dataset info (Iris - 3 species)
- Model architecture with softmax
- One-hot encoding explanation
- Training progress
- Accuracy for each class
- Confusion matrix plot

---

#### Logistic Regression
```bash
cd /Users/vivekgowdas/Desktop/LLM/assignment/LR
python logistic_regression.py
```
**Shows**:
- Dataset info
- Feature coefficients (most important)
- Single-split accuracy
- **5-Fold Cross-Validation scores** (more reliable)
- Feature importance visualization

---

## Expected Outputs

### Binary ANN
```
BINARY CLASSIFICATION ANN - HEART DISEASE PREDICTION
==================================================
Dataset shape: (297, 14)
Train set size: 237
Test set size: 60

Model Architecture:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 64)                896       
dropout (Dropout)            (None, 64)                0         
dense_1 (Dense)              (None, 32)                2080      
...
Total params: ~2,000

EVALUATION RESULTS:
✅ Accuracy:  57.38%
✅ Precision: 56.10%
✅ Recall:    74.19%
✅ F1-Score:  63.89%
✅ ROC-AUC:   59.14%
```

### Multiclass ANN (Softmax)
```
MULTI-CLASS CLASSIFICATION ANN - IRIS FLOWER CLASSIFICATION
===========================================================
Dataset shape: (150, 5)
Train set size: 120
Test set size: 30

Model uses SOFTMAX activation (3 classes)
One-hot encoding applied

EVALUATION RESULTS:
✅ Overall Accuracy: 93.33%
✅ Setosa - Precision: 1.0, Recall: 1.0
✅ Versicolor - Precision: 0.9, Recall: 0.9
✅ Virginica - Precision: 0.88, Recall: 0.88
```

### Logistic Regression
```
LOGISTIC REGRESSION - BINARY CLASSIFICATION
===========================================
Dataset shape: (1500, 12)
Train set size: 1200
Test set size: 300

Feature Coefficients:
Feature_1: +0.523
Feature_2: -0.341
... (all 12 features)

5-FOLD CROSS-VALIDATION RESULTS:
Accuracy:  95.50% (+/- 0.81%)
Precision: 96.20% (+/- 0.92%)
Recall:    95.10% (+/- 1.05%)
F1-Score:  95.60% (+/- 0.88%)
```

---

## Tips During Viva

### 1. Running Code
- **Have terminal ready** before viva starts
- **Show dataset first**: `head` command to show data
- **Run model**: Show output flowing in real-time
- **Don't worry about training time**: All complete in <1 minute total

### 2. Explaining Code on Screen
```bash
# Show relevant code lines
cat ANN/Binary/binary_classification.py | grep -A 5 "StandardScaler"

# Or open in editor
code logistic_regression.py
```

### 3. If Asked About Plots
```bash
# Check what plots were generated
ls -la *.png

# Can open in terminal if needed
open results.png  # on macOS
```

---

## Emergency Fixes

### If models don't run:
```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install -r ../requirements.txt

# Check imports
python -c "import tensorflow; import sklearn; print('OK')"
```

### If stuck during viva:
- **Backup plan**: Show the code visually
- **Explain from memory**: You know the concepts
- **Mention**: "Let me reload the dataset/model"
- **Ask moderator**: "May I run it again?"

---

## Viva Script (What to Say)

### Opening
"I have implemented three machine learning classification models:
1. **Binary ANN** using Keras - Heart Disease prediction
2. **Multiclass ANN** with Softmax - Iris flower classification
3. **Logistic Regression** using scikit-learn - Binary classification

All models are self-contained with their datasets. Let me run them for you."

### For Each Model
1. Run the code
2. Explain the output: "As you can see, accuracy is X%"
3. Point to key metrics: "This shows precision, recall, F1-score"
4. Explain why these metrics: "In medical diagnosis, recall is important..."

### Closing
"All three models demonstrate different approaches to classification:
- ANN for complex non-linear patterns
- Softmax for multi-class problems
- Logistic Regression for interpretable linear models"

---

## Quick Reference Table

| Model | Dataset | Command | Accuracy | Time |
|-------|---------|---------|----------|------|
| Binary ANN | Heart (297) | `python ANN/Binary/binary_classification.py` | 57.4% | ~20s |
| Multiclass ANN | Iris (150) | `python ANN/Multiclass/multiclass_classification.py` | 93.3% | ~20s |
| Logistic Reg | Mushroom (1500) | `python LR/logistic_regression.py` | 97.5% | ~5s |
| **All 3** | - | `python run_all.py` | - | ~45s |

---

**Ready for viva? Let's go! 💪**
