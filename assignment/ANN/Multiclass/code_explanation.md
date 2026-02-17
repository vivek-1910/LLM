# Multi-Class Classification ANN - Code Explanation

## Overview

This script implements an **Artificial Neural Network (ANN)** with **Softmax activation** to solve a **multi-class classification problem**: classifying iris flowers into one of 3 species based on petal/sepal measurements.

**Dataset**: UCI ML Repository - Iris Flower Dataset - 150 real flower measurements
**Problem Type**: Multi-Class Classification (3 classes: Setosa, Versicolor, Virginica)
**Solution Method**: Deep Neural Network with Softmax output

---

## Step-by-Step Code Breakdown

### STEP 1: Load and Explore Dataset

```python
df = pd.read_csv('iris_data.csv')
```

- **What it does**: Loads the famous Iris dataset containing 150 flower measurements
- **Dataset details**:
  - 150 flower samples
  - 4 features: sepal_length, sepal_width, petal_length, petal_width
  - 1 target: species (0=Setosa, 1=Versicolor, 2=Virginica)

**Key outputs**:
- `df.shape`: Shows (150, 6) - 150 samples, 6 columns (4 features + 1 target + 1 species name)
- Class distribution: Typically balanced (50 samples per species)
- Feature statistics: Mean, std, min, max for each measurement

---

### STEP 2: Data Preprocessing

#### Separate Features and Target
```python
X = df.iloc[:, :-2].values  # All columns except last 2 (target and species_name)
y = df['species'].values     # Numeric target (0, 1, or 2)
```

#### Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% for testing (30 samples)
    random_state=42,      # Reproducibility
    stratify=y            # Maintains 50-50-50 split in train/test
)
```

- **Train set**: 120 samples (40 per class)
- **Test set**: 30 samples (10 per class)

#### Feature Scaling
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Formula**: `x_scaled = (x - mean) / std_dev`

**Why?** Neural networks learn faster with normalized features in similar ranges

#### One-Hot Encoding (Critical for Multi-Class!)
```python
y_train_encoded = keras.utils.to_categorical(y_train, num_classes=3)
y_test_encoded = keras.utils.to_categorical(y_test, num_classes=3)
```

**Converts**:
- Class 0 (Setosa) → [1, 0, 0]
- Class 1 (Versicolor) → [0, 1, 0]
- Class 2 (Virginica) → [0, 0, 1]

**Why?** Categorical cross-entropy loss (which we use) requires this format for multi-class problems

---

### STEP 3: Build ANN Model with Softmax

```python
model = keras.Sequential([
    # Layer 1: 64 neurons
    layers.Dense(64, activation='relu', input_shape=(4,)),
    layers.Dropout(0.25),
    
    # Layer 2: 32 neurons
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.25),
    
    # Layer 3: 16 neurons
    layers.Dense(16, activation='relu'),
    
    # Output Layer: 3 neurons (ONE per class)
    layers.Dense(3, activation='softmax')
])
```

#### Architecture Comparison: Binary vs Multi-Class

| Aspect | Binary Classification | Multi-Class Classification |
|--------|----------------------|---------------------------|
| **Output neurons** | 1 neuron | Number of classes (3) |
| **Output activation** | Sigmoid | **Softmax** |
| **Target encoding** | 0 or 1 | One-hot [1,0,0] format |
| **Loss function** | binary_crossentropy | **categorical_crossentropy** |
| **Output interpretation** | Probability of class 1 | Probability distribution over 3 classes |

#### Softmax Activation (The Key Difference!)

**Formula**: For 3 classes,
```
softmax([z₁, z₂, z₃])ᵢ = e^(zᵢ) / (e^(z₁) + e^(z₂) + e^(z₃))
```

**Example**:
- Raw output: [2.0, 1.0, 0.5]
- Softmax: [0.659, 0.242, 0.099]  (probabilities sum to 1!)
- Interpretation: 65.9% Setosa, 24.2% Versicolor, 9.9% Virginica

**Why Softmax?**
- ✅ Converts outputs to probability distribution (sum = 1)
- ✅ Each output represents confidence in that class
- ✅ Required for categorical cross-entropy loss
- ✅ Natural for picking maximum as predicted class

#### ReLU in Hidden Layers
```
ReLU(z) = max(0, z)
```
- Learns non-linear relationships between features
- Example: The relationship between petal_length → species is non-linear

#### Dropout Regularization
```python
layers.Dropout(0.25)  # Randomly drop 25% of neurons
```
- Prevents overfitting by forcing robust learning
- Smaller than binary (0.25 vs 0.3) because this is simpler dataset

---

### STEP 4: Compile Model

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)
```

#### Binary Crossentropy vs Categorical Crossentropy

**Binary Crossentropy** (for binary):
```
Loss = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
```

**Categorical Crossentropy** (for multi-class):
```
Loss = -Σ(yᵢ * log(ŷᵢ)) for all 3 classes
Example: -[1*log(0.659) + 0*log(0.242) + 0*log(0.099)]
```

- Measures mismatch between true one-hot and predicted softmax

---

### STEP 5: Train Model

```python
history = model.fit(
    X_train_scaled, y_train_encoded,
    epochs=100,
    batch_size=8,             # Smaller batch for 120 samples
    validation_split=0.2,     # 20% validation from training
    verbose=0
)
```

**Training loop** (simplified):
```
For each of 100 epochs:
  For each batch of 8 samples:
    1. Forward pass through network (predict)
    2. Calculate categorical cross-entropy loss
    3. Backward pass (compute gradients)
    4. Update weights using Adam optimizer
  Evaluate on validation set (20% of 120 = 24 samples)
```

**Why smaller batch_size (8 vs 16)?** Smaller dataset (120 samples) → less memory, faster updates

---

### STEP 6: Make Predictions and Evaluate

```python
# Get probability predictions
y_pred_prob = model.predict(X_test_scaled, verbose=0)
# Returns shape (30, 3): 30 samples × 3 class probabilities

# Get class predictions
y_pred = np.argmax(y_pred_prob, axis=1)  # Index of maximum probability
# Example: [0, 1, 2, 0, 1, ...] for 30 test samples
```

#### Performance Metrics for Multi-Class

```python
accuracy = accuracy_score(y_test, y_pred)
# (Number of correct predictions) / (Total predictions)
# Works directly on multi-class!

precision_macro = precision_score(y_test, y_pred, average='macro')
# Average precision across 3 classes: (P₀ + P₁ + P₂) / 3

recall_macro = recall_score(y_test, y_pred, average='macro')
# Average recall across 3 classes: (R₀ + R₁ + R₂) / 3

f1_macro = f1_score(y_test, y_pred, average='macro')
# Harmonic mean of macro precision and macro recall
```

#### Confusion Matrix for Multi-Class

```
                Predicted
              Setosa  Versicolor  Virginica
Actual  Setosa      10      0           0
        Versicolor  0       9           1
        Virginica   0       1           9
```

**Interpretation**:
- Diagonal elements = correct predictions
- Off-diagonal = misclassifications
- Example: 1 Virginica sample misclassified as Versicolor

#### Per-Class Metrics

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, 
                          target_names=['Setosa', 'Versicolor', 'Virginica']))
```

Shows precision, recall, F1 for each class separately:
- **Setosa**: Often 100% (most distinct)
- **Versicolor vs Virginica**: More confusion (more similar)

---

### STEP 7: Generate Visualizations and Results

The script auto-generates:

1. **results.png**: 4-panel visualization
   - Training/validation accuracy curves
   - Confusion matrix heatmap (3×3 grid)
   - Per-class performance bar charts
   - ROC curves (one vs rest for each class)

2. **result.md**: Markdown report with
   - Dataset information (3 classes, 150 samples)
   - Model architecture details
   - Per-class metrics table
   - Confusion matrix
   - Softmax explanation
   - Generated timestamp

---

## Key Concepts Summary

### Multi-Class vs Binary Classification

| Aspect | Binary | Multi-Class |
|--------|--------|-------------|
| **Num classes** | 2 | 3+ |
| **Output layer** | 1 neuron | K neurons |
| **Activation** | Sigmoid | Softmax |
| **Target format** | 0 or 1 | One-hot [0,1,0] |
| **Loss** | binary_crossentropy | categorical_crossentropy |
| **Prediction** | threshold at 0.5 | argmax (highest probability) |
| **Example output** | [0.7] means class 1 | [0.1, 0.8, 0.1] means class 1 |

### Softmax Mathematical Property

**Invariance to adding constant**:
```
softmax([z₁, z₂, z₃]) = softmax([z₁+c, z₂+c, z₃+c])
```
This prevents numerical overflow in computation!

### Feature Interactions Learned

The ANN learns that:
- Setosa: Small petals, distinct from others
- Versicolor: Medium petals, some overlap with Virginica
- Virginica: Large petals, some overlap with Versicolor

Learned through 64 → 32 → 16 neuron pathway

---

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Low accuracy on one class | Class imbalance | This dataset is balanced ✓ |
| Confusion between two classes | Similar features | Acceptable for Versicolor/Virginica |
| Overfitting to training | Model too complex | Dropout helps, sufficient data ✓ |
| Predictions not summing to 1 | Bug in softmax | Use softmax activation ✓ |

---

## Running the Script

```bash
cd ANN/Multiclass
python multiclass_classification.py
```

**Output Files**:
- `results.png` - Visualization dashboard
- `result.md` - Performance report with per-class metrics
- Terminal output with model summary and classification report

---

## Expected Performance

Based on Iris dataset characteristics:
- **Accuracy**: 90-97% (this is an easy dataset for ANNs)
- **Setosa Recall**: 95-100% (very distinct, rarely misclassified)
- **Versicolor/Virginica**: 85-95% (natural confusion due to similarity)
- **Macro F1**: 90%+ (balanced across all classes)

*The Iris dataset is often used for testing because it's well-separated (except Versicolor vs Virginica)*

---

## Softmax Example Walkthrough

**Scenario**: Network outputs raw scores for a mystery flower: [1.5, -0.5, 0.2]

**Step 1**: Transform with softmax
```
e^1.5 ≈ 4.48
e^-0.5 ≈ 0.61
e^0.2 ≈ 1.22
Sum ≈ 6.31

softmax = [4.48/6.31, 0.61/6.31, 1.22/6.31]
        = [0.711, 0.097, 0.193]
```

**Step 2**: Interpret
- 71.1% confidence → Setosa
- 9.7% confidence → Versicolor
- 19.3% confidence → Virginica

**Step 3**: Predict
- Argmax = index 0 → **Predicted class: Setosa**
- True label from test: Setosa
- ✅ **Correct!**
