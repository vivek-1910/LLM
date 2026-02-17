# Logistic Regression - Code Explanation

## Overview

This script implements **Logistic Regression**, a classical machine learning algorithm for **binary classification**: predicting whether a wine is of good quality or poor quality based on 6 chemical properties.

**Dataset**: UCI ML Repository - Wine Quality Dataset - 178 real wine samples
**Problem Type**: Binary Classification (2 classes: Good Quality, Poor Quality)
**Solution Method**: Logistic Regression with feature standardization

---

## Key Concept: What is Logistic Regression?

Despite the name "regression", Logistic Regression is a **classification** algorithm.

**Core Idea**: Model the probability that a sample belongs to class 1

```
P(y=1|x) = 1 / (1 + e^(-(w·x + b)))
         = sigmoid(w·x + b)
```

Where:
- `w` = learned weights (coefficients)
- `b` = bias term (intercept)
- `x` = input features
- `σ` = sigmoid function

**Decision Rule** (threshold = 0.5):
- If P(y=1|x) ≥ 0.5 → Predict class 1 (Good Quality)
- If P(y=1|x) < 0.5 → Predict class 0 (Poor Quality)

---

## Step-by-Step Code Breakdown

### STEP 1: Load and Explore Dataset

```python
df = pd.read_csv('student_performance.csv')

# Create binary target from quality
df['pass'] = (df['quality'] > 1).astype(int)
```

- **What it does**: Loads wine dataset and binarizes the target
- **Dataset details**:
  - 178 wine samples
  - 13 features originally, using 6 key chemical properties: alcohol, malic_acid, ash, magnesium, phenols, proline
  - 1 quality score (continuous, converted to binary)

- **Binary target creation**:
  - Quality ≤ 1 → Poor Quality (class 0)
  - Quality > 1 → Good Quality (class 1)

**Key outputs**:
- `df.shape`: Shows (178, 14) - 178 wines, 14 columns
- Class distribution: Shows number of good vs poor wines
- Feature statistics: Mean, std, min, max for each chemical property

---

### STEP 2: Data Preprocessing

#### Separate Features and Target
```python
X = df.drop(['pass', 'quality'], axis=1)  # Remove both target columns
y = df['pass']                              # Binary target (0 or 1)
```

- **Why drop 'quality'?** Because 'pass' is directly derived from 'quality'
- **Why keep features?** Chemical properties (alcohol, phenols, etc.)

#### Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% for testing (36 wines)
    random_state=42,      # Reproducibility
    stratify=y            # Maintain class distribution
)
```

- **Train set**: 142 wines (80%)
- **Test set**: 36 wines (20%)

#### Feature Standardization (Critical!)
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Transformation**:
```
x_scaled = (x - mean) / std_dev
```

**Example** (alcohol feature):
```
Original: [12.3, 13.1, 11.9, ...]
Mean: 12.0  Std: 1.5

After scaling: [0.2, 0.73, -0.067, ...]
(each value centered at 0, ranges from -3 to +3)
```

**Why standardize for Logistic Regression?**
- ✅ Prevents coefficients with huge differences in magnitude
- ✅ Improves numerical stability in optimization
- ✅ Makes interpretation easier (unit = 1 std change)
- ✅ Helps solver converge faster

---

### STEP 3: Train Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,      # Max iterations to find optimal weights
    random_state=42,    # Reproducibility
    solver='lbfgs'      # Optimization algorithm
)

model.fit(X_train_scaled, y_train)
```

#### What the model learns

After training, model has:
```python
model.coef_[0]        # Weights for 6 features
model.intercept_[0]   # Bias term (b)
```

**Example coefficients** (from wine dataset):
```
alcohol:    -1.0707   (higher alcohol → slightly more poor quality??)
malic_acid: -0.2964
ash:        -0.3771
magnesium:  0.1337    (higher magnesium → more good quality)
phenols:    -1.3452
proline:    -2.5765   (strongest negative effect)
Intercept:   1.7588
```

#### Interpretation of Coefficients

- **Sign**: Direction of effect
  - Positive (+): Increases probability of class 1 (Good Quality)
  - Negative (-): Decreases probability of class 1

- **Magnitude**: Strength of effect
  - |−2.58| > |−0.30| → proline has stronger effect than malic_acid

- **Unit**: One standard deviation increase in feature

**Example**: When wine data was standardized:
- 1 std increase in alcohol → log-odds decrease by 1.07
- 1 std increase in magnesium → log-odds increase by 0.13

---

### STEP 4: Make Predictions

#### Probability Predictions
```python
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
```

- `predict_proba()` returns shape (36, 2)
  - Column 0: P(y=0|x) for each sample
  - Column 1: P(y=1|x) for each sample (what we want)

**Example**:
```
Sample 1: [0.3, 0.7] → 70% confidence of good quality
Sample 2: [0.9, 0.1] → 10% confidence of good quality (95% poor)
Sample 3: [0.5, 0.5] → 50% confidence (borderline)
```

#### Class Predictions
```python
y_pred = model.predict(X_test_scaled)
```

- Applies threshold at 0.5
- If P(y=1|x) ≥ 0.5 → predict 1
- If P(y=1|x) < 0.5 → predict 0

---

### STEP 5: Evaluate Model

#### Performance Metrics

```python
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)
```

**Detailed Equations**:

| Metric | Formula | Meaning | Example |
|--------|---------|---------|---------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | % correct overall | 33/36 = 91.7% |
| **Precision** | TP/(TP+FP) | Of predicted good, how many actually good | If predicted 10 good, 8 actually good → 80% |
| **Recall** | TP/(TP+FN) | Of actual good, how many predicted | If 12 actually good, found 8 → 66.7% |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | Harmonic mean (balance) | 0.727 |
| **ROC-AUC** | Probability model ranks random + higher than random - | Discrimination ability | 0.95 = excellent |

#### Confusion Matrix

```
                Predicted
              Good  Poor
Actual  Good   18    2    (2 missed good wines)
        Poor   2    14    (2 false alarms)
```

- **TP = 18**: Correctly predicted good
- **TN = 14**: Correctly predicted poor
- **FP = 2**: False alarms (predicted good, actually poor)
- **FN = 2**: Misses (predicted poor, actually good)

---

### STEP 6: Generate Visualizations and Results

The script auto-generates:

1. **results.png**: 4-panel visualization
   - Coefficient plot: Bar chart showing feature importance
   - Confusion matrix heatmap
   - ROC curve (True Positive vs False Positive rate)
   - Probability distribution of predictions

2. **result.md**: Markdown report with
   - Dataset information
   - Model type and parameters
   - Performance metrics table
   - Feature coefficients table (auto-generated from model.coef_)
   - Confusion matrix details
   - Decision boundary explanation

---

## Key Concepts Summary

### Logistic Function (Sigmoid)

**Formula**: `σ(z) = 1 / (1 + e^(-z))`

**Graph**:
```
Probability
1.0  |           ╱╱╱
     |       ╱╱╱
0.5  |------●-------
     |   ╱╱╱
0.0  |╱╱╱
     |-3  -1  0  1  3  (z = w·x + b)
```

- **S-shaped curve**: Smoothly transitions from 0 to 1
- **At z=0**: σ(0) = 0.5 (equal probability)
- **As z → ∞**: σ(z) → 1 (high confidence, class 1)
- **As z → -∞**: σ(z) → 0 (high confidence, class 0)

### Decision Boundary

After training, decision boundary is a **linear hyperplane** in feature space:

```
w·x + b = 0  where w=[w₁, w₂, ..., w₆], x=[x₁, x₂, ..., x₆]
```

- Points on one side: P(y=1|x) > 0.5 → Predict class 1
- Points on other side: P(y=1|x) < 0.5 → Predict class 0
- Points exactly on boundary: P(y=1|x) = 0.5

**Why linear?** Logistic Regression can only learn linear boundaries (limitation for complex patterns)

### Optimization: Finding Optimal Weights

**Loss Function** (Binary Cross-Entropy):
```
Loss = (1/m) × Σ [ -y*log(ŷ) - (1-y)*log(1-ŷ) ]
```

Where:
- `m` = number of samples
- `y` = true label (0 or 1)
- `ŷ` = predicted probability

**Intuition**:
- If true label is 1 but predict 0.1: loss = -log(0.1) = 2.3 (high penalty)
- If true label is 1 and predict 0.9: loss = -log(0.9) = 0.1 (low penalty)

**Solver** (lbfgs):
- Iteratively adjusts weights to minimize loss
- Stops when converged or max_iter reached

---

## Logistic Regression vs Neural Networks

| Aspect | Logistic Regression | ANN |
|--------|-------------------|-----|
| **Decision boundary** | Linear (hyperplane) | Non-linear (curved) |
| **Interpretability** | Highly interpretable (coefficients) | Black box |
| **Feature interactions** | Manual (must engineer) | Automatic (learned) |
| **Training data needed** | Small (works with ~100 samples) | More data (works best with 1000+) |
| **Training speed** | Very fast | Slower |
| **Model complexity** | Simple | Complex |
| **When to use** | Simple, interpretable problems | Complex, non-linear problems |

**For wine dataset**: Logistic regression achieves ~100% accuracy! (Problem is linearly separable)

---

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Convergence warning | max_iter too low | Increase max_iter (we use 1000) |
| Poor performance | Features on different scales | Use StandardScaler ✓ |
| Underfitting | Model too simple | Try polynomial features or ANN |
| Class imbalance | Skewed data | Use class_weight='balanced' |
| Non-linear pattern | Linear model limitation | Use ANN instead |

---

## Running the Script

```bash
cd LR
python logistic_regression.py
```

**Output Files**:
- `results.png` - Visualization dashboard with coefficient plot
- `result.md` - Performance report with feature coefficients table
- Terminal output with model summary and classification report

---

## Expected Performance

Based on wine dataset characteristics:
- **Accuracy**: 95-100% (linearly separable problem! ✓)
- **Precision**: 95-100% (few false alarms)
- **Recall**: 95-100% (catches most good wines)
- **ROC-AUC**: 0.98-1.00 (excellent discrimination)

*This dataset is well-suited for logistic regression because:*
- ✅ Only 6 features (low dimensionality)
- ✅ Clear separation between classes
- ✅ Linear patterns are sufficient
- ✅ High interpretability needed for business decisions

---

## Coefficient Interpretation Example

**From trained model**:
```
proline: -2.5765 (strongest negative effect)
```

**Real-world meaning**:
- Wines with higher proline content are more likely to be classified as poor quality
- This seems contradictory! But remember:
  1. These are standardized coefficients
  2. Proline correlates with other features too
  3. The model learns patterns in the data

**Use case**: Winemakers could use coefficients to understand which chemical properties the model values for quality assessment.

---

## Formula Walkthrough: Prediction Example

**New wine with scaled features**: x = [0.5, -0.2, 0.1, 0.3, 0.8, -0.4]

**Step 1**: Linear combination
```
z = w·x + b
  = (-1.07×0.5) + (-0.30×-0.2) + (-0.38×0.1) + (0.13×0.3) + (-1.35×0.8) + (-2.58×-0.4) + 1.76
  = -0.535 + 0.06 - 0.038 + 0.039 - 1.08 + 1.032 + 1.76
  = 1.238
```

**Step 2**: Apply sigmoid
```
ŷ = 1 / (1 + e^(-1.238))
  = 1 / (1 + 0.290)
  = 1 / 1.290
  = 0.775
```

**Step 3**: Predict class
```
P(Good Quality) = 0.775 = 77.5%
P(Poor Quality) = 0.225 = 22.5%

Threshold = 0.5
0.775 > 0.5 → Predict: GOOD QUALITY ✓
```

**Confidence**: 77.5% - moderately confident (not borderline, not absolute)
