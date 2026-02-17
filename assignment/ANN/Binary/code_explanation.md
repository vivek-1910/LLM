# Binary Classification ANN - Code Explanation

## Overview

This script implements an **Artificial Neural Network (ANN)** to solve a **binary classification problem**: predicting whether a patient has heart disease or not. The model learns patterns from 13 medical features to classify patients into two classes: disease present (1) or not present (0).

**Dataset**: UCI ML Repository - Heart Disease (Cleveland) - 297 real patient records
**Problem Type**: Binary Classification (2 classes)
**Solution Method**: Deep Neural Network with Sigmoid output

---

## Step-by-Step Code Breakdown

### STEP 1: Load and Explore Dataset

```python
df = pd.read_csv('heart_disease.csv')
```

- **What it does**: Loads the heart disease dataset from a CSV file
- **Dataset details**:
  - 297 patient records from Cleveland heart disease database
  - 13 features: age, sex, chest pain, blood pressure, cholesterol, etc.
  - 1 target column: presence/absence of heart disease (0 or 1)

**Key outputs**:
- `df.shape`: Shows (297, 14) - 297 samples, 14 columns
- `df.info()`: Displays data types and missing values
- `df['target'].value_counts()`: Shows class distribution (imbalanced/balanced?)

---

### STEP 2: Data Preprocessing

#### Separate Features and Target
```python
X = df.drop('target', axis=1)  # Features (13 columns)
y = df['target']                # Target (0 or 1)
```

#### Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,        # 20% for testing, 80% for training
    random_state=42,      # Reproducibility
    stratify=y            # Maintains class distribution
)
```

#### Feature Scaling (Standardization)
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Learn mean/std from train data
X_test_scaled = scaler.transform(X_test)        # Apply same transformation to test
```

**Why scale?**
- Neural networks perform better when features have similar ranges
- Standardization formula: `x_scaled = (x - mean) / std_dev`
- Helps model converge faster during training

---

### STEP 3: Build ANN Model

```python
model = keras.Sequential([
    # Layer 1: 64 neurons, ReLU activation
    layers.Dense(64, activation='relu', input_shape=(13,)),
    layers.Dropout(0.3),
    
    # Layer 2: 32 neurons, ReLU activation
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    
    # Layer 3: 16 neurons, ReLU activation
    layers.Dense(16, activation='relu'),
    
    # Output Layer: 1 neuron, Sigmoid activation
    layers.Dense(1, activation='sigmoid')
])
```

#### Architecture Explanation

| Component | Purpose | Formula |
|-----------|---------|---------|
| **Input Layer** | 13 features from preprocessed data | - |
| **Hidden Layer 1** | 64 neurons extract initial patterns | ReLU: max(0, z) |
| **Dropout(0.3)** | Randomly deactivate 30% of neurons to prevent overfitting | - |
| **Hidden Layer 2** | 32 neurons refine patterns | ReLU: max(0, z) |
| **Dropout(0.3)** | Regularization | - |
| **Hidden Layer 3** | 16 neurons further reduce dimensionality | ReLU: max(0, z) |
| **Output Layer** | 1 neuron outputs probability | Sigmoid: 1/(1+e^(-z)) |

#### Key Activation Functions

- **ReLU (Rectified Linear Unit)**: Solves vanishing gradient problem, enables learning non-linear patterns
  - Formula: `f(z) = max(0, z)`
  - Output ranges from 0 to ∞

- **Sigmoid**: Converts output to probability (0 to 1) for binary classification
  - Formula: `σ(z) = 1 / (1 + e^(-z))`
  - Output: 0.7 means 70% probability of disease present

#### Dropout Regularization
- **Purpose**: Prevents overfitting by forcing network to learn robust features
- **Mechanism**: During training, randomly "drops" (deactivates) neurons
- **Effect**: Model becomes more generalized, performs better on unseen data

---

### STEP 4: Compile Model

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| **optimizer** | 'adam' | Adaptive learning rate - adjusts learning rate per feature |
| **loss** | 'binary_crossentropy' | -[y*log(ŷ) + (1-y)*log(1-ŷ)] - measures prediction error |
| **metrics** | 'accuracy' | Tracks percentage of correct predictions |

---

### STEP 5: Train Model

```python
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,           # 100 complete passes through training data
    batch_size=16,        # Update weights after every 16 samples
    validation_split=0.2, # Use 20% of training data for validation
    verbose=0             # Don't print training progress
)
```

**Training Process**:
1. **Epoch 1**: Model sees all 237 training samples, updates weights 15 times
2. **Epoch 2-100**: Repeat, gradually improving predictions
3. **Validation**: After each epoch, evaluate on validation set (20% of training)
4. **Goal**: Minimize loss while maintaining good validation accuracy (watch for overfitting)

---

### STEP 6: Make Predictions and Evaluate

```python
# Get probability predictions
y_pred_prob = model.predict(X_test_scaled, verbose=0)  # Returns [0.1, 0.9, 0.2, ...]

# Convert to binary predictions (threshold = 0.5)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()  # [0, 1, 0, ...]
```

#### Performance Metrics

```python
accuracy  = accuracy_score(y_test, y_pred)      # (TP+TN)/(TP+TN+FP+FN)
precision = precision_score(y_test, y_pred)     # TP/(TP+FP) - how many predicted disease actually have it
recall    = recall_score(y_test, y_pred)        # TP/(TP+FN) - caught how many disease cases
f1        = f1_score(y_test, y_pred)            # 2*(precision*recall)/(precision+recall) - harmonic mean
roc_auc   = roc_auc_score(y_test, y_pred_prob)  # Area under ROC curve
```

**Interpretation**:
- **Accuracy**: Overall correctness (but misleading for imbalanced data)
- **Precision**: Of predicted diseases, how many are correct (minimize false alarms)
- **Recall**: Of actual diseases, how many did we catch (minimize missed cases)
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Probability that model ranks a random positive case higher than negative

#### Confusion Matrix

```
                Predicted
              Disease  No Disease
Actual  Disease     TP      FN
        No Disease  FP      TN
```

- **TP (True Positive)**: Correctly predicted disease
- **TN (True Negative)**: Correctly predicted no disease
- **FP (False Positive)**: Wrong alarm (healthy predicted as sick)
- **FN (False Negative)**: Missed case (sick predicted as healthy)

---

### STEP 7: Generate Visualizations and Results

The script auto-generates:

1. **results.png**: 4-panel visualization
   - Training history (loss and accuracy curves)
   - Confusion matrix heatmap
   - ROC curve (True Positive Rate vs False Positive Rate)
   - Feature importance (if available)

2. **result.md**: Markdown report with
   - Dataset information
   - Model architecture
   - Performance metrics table
   - Confusion matrix details
   - Interpretation and recommendations

---

## Key Concepts Summary

### Binary Classification Problem
- **Input**: 13 medical features (age, cholesterol, blood pressure, etc.)
- **Output**: Single probability value (0-1) indicating disease probability
- **Decision Rule**: threshold at 0.5
  - If probability ≥ 0.5 → Patient has disease (class 1)
  - If probability < 0.5 → Patient does not have disease (class 0)

### Neural Network Advantages
- ✅ Captures non-linear relationships between features
- ✅ Automatically learns important feature interactions
- ✅ Flexible architecture adaptable to problem complexity
- ✅ Can handle many features efficiently

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Low test accuracy | Underfitting | Add more layers/neurons or train longer |
| Overfitting | Model too complex | Add dropout, reduce model size, more data |
| Slow convergence | Poor learning rate | Use Adam optimizer (automatic tuning) |
| Class imbalance issues | Skewed data | Use stratified split, adjust class weights |

---

## Running the Script

```bash
cd ANN/Binary
python binary_classification.py
```

**Output Files**:
- `results.png` - Visualization dashboard
- `result.md` - Performance report
- Terminal output with metrics and model summary

---

## Expected Performance

Based on UCI dataset characteristics:
- **Accuracy**: 75-85% (reasonable for medical data)
- **Precision**: 80-90% (important to minimize false alarms)
- **Recall**: 70-80% (important to catch actual cases)
- **ROC-AUC**: 0.85-0.95 (good discriminative ability)

*Actual results depend on random train-test split and model initialization*
