# Quick Start Guide

## Assignment: ANN & Logistic Regression (UE23AM342BB2)

### ⚡ Quick Setup (3 steps)

#### Step 1: Install Dependencies
```bash
cd /Users/vivekgowdas/Desktop/LLM/assignment
pip install -r requirements.txt
```

#### Step 2: Run All Scripts (automatic)
```bash
python run_all_assignments.py
```

**OR run individually:**
```bash
python 1_binary_classification_ann.py
python 2_multiclass_classification_ann.py
python 3_logistic_regression.py
```

#### Step 3: View Results
- Check console output for metrics
- Open PNG files in `assignment/` folder:
  - `1_binary_classification_results.png`
  - `2_multiclass_classification_results.png`
  - `3_logistic_regression_results.png`

---

## 📊 What Each Script Does

### 1️⃣ Binary Classification ANN
- **Problem**: Predict heart disease (Yes/No)
- **Model**: Neural Network with 3 hidden layers
- **Output**: Single neuron + Sigmoid activation
- **Expected Accuracy**: ~57-60%
- **Run Time**: ~30 sec

### 2️⃣ Multi-class Classification ANN
- **Problem**: Classify iris flowers (3 species)
- **Model**: Neural Network with Softmax output
- **Output**: 3 neurons representing each class
- **Expected Accuracy**: ~93-95%
- **Run Time**: ~20 sec

### 3️⃣ Logistic Regression
- **Problem**: Predict student pass/fail
- **Model**: Linear classifier
- **Output**: Probability of passing
- **Expected Accuracy**: ~97-98%
- **Run Time**: <1 sec

---

## 📝 Key Points for Viva

### Binary Classification
- Sigmoid activation: f(x) = 1/(1+e^-x) outputs [0,1]
- Binary cross-entropy loss
- Dropout regularization to prevent overfitting
- Confusion matrix: TP, FP, TN, FN

### Multi-class Classification
- Softmax activation: e^z_i / Σ(e^z_j)
- One-hot encoding for targets
- Categorical cross-entropy loss
- 3 output neurons (one per class)

### Logistic Regression
- Linear model: P(y=1) = sigmoid(w·x + b)
- Interpretable coefficients
- Suitable for linear relationships
- Fast training, no hyperparameters

---

## 🎯 Expected Output

### Console Output
```
================================================================================
BINARY CLASSIFICATION ANN - HEART DISEASE PREDICTION
================================================================================
...
Accuracy:  0.5738
Precision: 0.5610
Recall:    0.7419
F1-Score:  0.6389
ROC-AUC:   0.5914
✅ Visualization saved: 1_binary_classification_results.png
```

### Files Created
- 3 Python scripts (925+ lines of code)
- 3 PNG visualizations
- README.md (comprehensive documentation)
- requirements.txt (dependencies)
- This file

---

## ❓ Common Questions

**Q: Why is accuracy low for binary classification?**
A: Random synthetic dataset doesn't have strong patterns. Real data would perform better.

**Q: Why does Iris have 93% accuracy?**
A: Iris is a classic, well-structured dataset with clear patterns.

**Q: Why does Logistic Regression have 97.5% accuracy?**
A: Student data has linear relationship between features and passing.

**Q: How long does it take?**
A: Total runtime ~50-60 seconds for all three models.

**Q: Can I run just one script?**
A: Yes! Each script is independent:
```bash
python 1_binary_classification_ann.py  # Only this
```

**Q: Are the visualizations good?**
A: Yes! 4 plots per script showing training, confusion matrix, ROC, etc.

---

## 🔧 Troubleshooting

### Issue: "No module named 'tensorflow'"
**Fix**: `pip install tensorflow`

### Issue: "Python not found"
**Fix**: Use: `/Users/vivekgowdas/Desktop/LLM/.venv/bin/python script.py`

### Issue: Plots not showing
**Fix**: Check the PNG files in the folder instead

### Issue: ModuleNotFoundError
**Fix**: Reinstall: `pip install -r requirements.txt`

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| README.md | Full technical documentation |
| EVALUATION_SUMMARY.md | Points for viva questions |
| requirements.txt | Python dependencies |
| run_all_assignments.py | Run all 3 scripts at once |

---

## ✅ Verification Checklist

Before viva, ensure:
- [ ] All 3 scripts run without errors
- [ ] 3 PNG visualizations generated
- [ ] Can explain each line of code
- [ ] Understand all metrics (accuracy, precision, recall, etc.)
- [ ] Know the formulas (softmax, sigmoid, cross-entropy)
- [ ] Can interpret confusion matrix
- [ ] Familiar with activation functions

---

## 🚀 Ready to Present

The assignment includes:
1. ✅ Binary classification ANN
2. ✅ Multi-class classification with Softmax  
3. ✅ Logistic regression
4. ✅ Performance metrics
5. ✅ Visualizations
6. ✅ Documentation
7. ✅ Executable code
8. ✅ Detailed comments

**Status: COMPLETE AND READY FOR EVALUATION**

---

Need help? Check README.md or EVALUATION_SUMMARY.md for detailed explanations.
