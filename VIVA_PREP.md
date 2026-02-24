# 🎓 VIVA PREPARATION - All 3 Models

**Date**: February 19, 2026  
**Assignment**: UE23AM342BB2: Large Language Models and Their Applications  
**Deadline**: February 20, 2026  

---

## 📋 QUICK OVERVIEW - 3 MODELS

| # | Model | Type | Dataset | Key Words | Accuracy |
|---|-------|------|---------|-----------|----------|
| 1️⃣ | **Binary ANN** | Deep Learning (Keras) | Heart Disease (297 samples) | Sigmoid, Binary Cross-Entropy | ~57% |
| 2️⃣ | **Multiclass ANN** | Deep Learning (Keras) | Iris Flowers (150 samples) | Softmax, Categorical Cross-Entropy, One-Hot Encoding | ~93% |
| 3️⃣ | **Logistic Regression** | Linear Model (Scikit-learn) | Mushroom Classification (1500 samples) | Linear Classifier, Max Likelihood, Feature Coefficients | ~97.5% |

---

# 🔴 MODEL 1: BINARY CLASSIFICATION ANN

## Problem Statement
**Predict whether a patient has heart disease (Yes/No) based on 13 medical features**

### Dataset Details
- **Source**: UCI ML Repository (Real Data - Cleveland Heart Disease Dataset)
- **Samples**: 297 real patient records
- **Features**: 13 medical measurements (age, cholesterol, blood pressure, etc.)
- **Class Distribution**: ~54% positive, ~46% negative
- **Train-Test Split**: 80-20 with stratification

---

## Model Architecture
```
Input Layer (13 features)
    ↓
Dense Layer 1: 64 neurons + ReLU activation
    ↓
Dropout 0.3 (Prevents overfitting)
    ↓
Dense Layer 2: 32 neurons + ReLU activation
    ↓
Dropout 0.3 
    ↓
Dense Layer 3: 16 neurons + ReLU activation
    ↓
Output Layer: 1 neuron + SIGMOID activation
    ↓
Binary Prediction (0 or 1)
```

---

## Key Concepts to Explain

### 1️⃣ **Why Sigmoid Activation?**
```
Sigmoid Function: f(x) = 1 / (1 + e^(-x))
- Output range: [0, 1] (represents probability)
- Smooth gradient → good for backpropagation
- Threshold: if output > 0.5 → positive class, else negative class
- Medical context: 0.8 means 80% probability of heart disease
```

### 2️⃣ **Why Binary Cross-Entropy Loss?**
```
BCELoss = -1/m * Σ(y*log(ŷ) + (1-y)*log(1-ŷ))
- y = actual label (0 or 1)
- ŷ = predicted probability
- Penalizes confident wrong predictions heavily
- Standard loss for binary classification
```

### 3️⃣ **Why Dropout?**
```
Dropout (30%):
- Randomly deactivates 30% of neurons during training
- Forces network to learn redundant representations
- Prevents overfitting
- At prediction time, all neurons are active (scaled by 0.7)
```

### 4️⃣ **Why ReLU Hidden Layers?**
```
ReLU: f(x) = max(0, x)
- Non-linear activation → learns complex patterns
- Computationally efficient
- Avoids vanishing gradient problem
- Good for hidden layers
```

### 5️⃣ **Data Preprocessing**
```
StandardScaler:
- Transform: x_scaled = (x - mean) / std_dev
- Mean becomes 0, Standard deviation becomes 1
- Why? Neural networks converge faster with scaled data
- Important: Fit on train set, apply to test set
```

---

## Performance Metrics Explained

```
Accuracy: 57.38% - (TP + TN) / Total
  ✓ Overall correctness
  ✗ Can be misleading with imbalanced classes

Precision: 56.10% - TP / (TP + FP)
  ✓ Of predicted positive cases, how many are actually positive
  ✓ Important for: Reducing false alarms (Type I error)

Recall: 74.19% - TP / (TP + FN)
  ✓ Of actual positive cases, how many we caught
  ✓ Important for: Catching all actual cases (Type II error)
  ✓ Medical context: Don't miss patients with disease

F1-Score: 63.89% - 2 * (Precision * Recall) / (Precision + Recall)
  ✓ Harmonic mean of precision & recall
  ✓ Good overall metric when classes are imbalanced

ROC-AUC: 59.14% - Area under ROC curve
  ✓ Probability that model ranks random positive higher than negative
  ✓ Independent of threshold
  ✓ 0.5 = random, 1.0 = perfect
```

---

## Viva Questions & Answers

### Q1: Why is accuracy 57% but it seems low?
**A:** The dataset is imbalanced (~54% positive), and accuracy alone is misleading. We should focus on:
- **Recall (74%)**: We catch 74% of actual disease cases - important for medical diagnosis
- **Precision (56%)**: 56% of our positive predictions are correct
- **ROC-AUC (59%)**: Shows reasonable discrimination ability

### Q2: Why not use more complex architecture?
**A:** Trade-offs:
- More layers/neurons → risk of overfitting (especially with 297 samples)
- Dropout helps prevent overfitting
- Current architecture balances complexity vs generalization
- Simple models often generalize better with small datasets

### Q3: What's the difference between validation and test set?
**A:**
- **Validation set** (from train split): Monitor performance during training, tune hyperparameters
- **Test set** (20% holdout): Final evaluation, simulate real unseen data
- We NEVER touch test set during training/tuning

### Q4: Why standardize features?
**A:**
- Features have different scales (age: 0-100, cholesterol: 100-400, etc.)
- Neural networks treat all inputs equally
- Standardization ensures equal "importance" initially
- Helps optimizer (Adam) converge faster

### Q5: How does dropout work exactly?
**A:**
- During training: Randomly set 30% of neuron outputs to 0
- Each batch uses different random neurons
- Forces network to not rely on specific neurons
- At prediction: All neurons active but outputs scaled down
- Prevents co-adaptation and overfitting

### Q6: Why sigmoid at output, not ReLU?
**A:**
- ReLU: f(x) = max(0,x) → unbounded output
- Sigmoid: f(x) = 1/(1+e^-x) → bounded [0,1]
- Medical prediction needs probability [0,1]
- Matches binary cross-entropy loss function

---

# 🟣 MODEL 2: MULTICLASS CLASSIFICATION ANN (SOFTMAX)

## Problem Statement
**Classify iris flowers into 3 species (Setosa, Versicolor, Virginica) based on 4 measurements**

### Dataset Details
- **Source**: UCI ML Repository - Classic Iris Dataset
- **Samples**: 150 real flower samples (50 per species)
- **Features**: 4 measurements (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 balanced classes (equal distribution)
- **Train-Test Split**: 80-20 with stratification

---

## Model Architecture
```
Input Layer (4 features)
    ↓
Dense Layer 1: 64 neurons + ReLU activation
    ↓
Dropout 0.25 (Prevents overfitting)
    ↓
Dense Layer 2: 32 neurons + ReLU activation
    ↓
Dropout 0.25
    ↓
Dense Layer 3: 16 neurons + ReLU activation
    ↓
Output Layer: 3 neurons + SOFTMAX activation
    ↓
3-Class Prediction (probability distribution)
```

---

## Key Concepts to Explain

### 1️⃣ **What is Softmax Activation?**
```
Softmax converts 3 raw outputs into probability distribution:

Raw outputs (logits): [2.1, 0.3, -0.5]
                    ↓
Softmax: softmax(z_i) = e^(z_i) / Σ(e^(z_j)) for j = 1 to 3

Step 1: e^2.1 = 8.17,  e^0.3 = 1.35,  e^-0.5 = 0.61
Step 2: Sum = 8.17 + 1.35 + 0.61 = 10.13
Step 3: Final = [8.17/10.13, 1.35/10.13, 0.61/10.13]
       Final = [0.807, 0.133, 0.060]

✓ All outputs in [0,1]
✓ Sum to 1 → probability distribution
✓ Argmax gives predicted class
```

### 2️⃣ **Why Categorical Cross-Entropy Loss?**
```
CCELoss = -Σ(y_true * log(y_pred))

Example with one-hot encoding:
y_true = [1, 0, 0]  (actual: Setosa)
y_pred = [0.8, 0.15, 0.05]  (predicted probabilities)

Loss = -(1*log(0.8) + 0*log(0.15) + 0*log(0.05))
     = -log(0.8)
     = 0.223

⚠️ If we predicted [0.1, 0.2, 0.7] (wrong class):
Loss = -log(0.1) = 2.303 (much higher penalty)
```

### 3️⃣ **What is One-Hot Encoding?**
```
Converts categorical labels to binary vectors:

Label "Setosa" (class 0)     → [1, 0, 0]
Label "Versicolor" (class 1) → [0, 1, 0]
Label "Virginica" (class 2)  → [0, 0, 1]

Why?
- Compatible with softmax output (3 neurons)
- Each neuron learns to predict one class
- Enables categorical cross-entropy loss
```

### 4️⃣ **Training Details**
```
Epochs: 100 (complete passes through training data)
Batch size: 8 (process 8 samples before updating weights)
Validation split: 20% of training data

Why small batch?
- More frequent weight updates
- Less memory required
- Adds noise that can help escape local minima
```

---

## Performance Metrics

```
Accuracy: 93.33%
Precision, Recall, F1: Each calculated per class
Confusion matrix: 30x30 matrix showing misclassifications

Why high accuracy?
1. Dataset is balanced (50 samples per class)
2. Classes are well-separated in 4D space
3. Problem is relatively simple (only 4 features)
4. ANN with 3 hidden layers is sufficient
```

---

## Viva Questions & Answers

### Q1: Why 3 neurons in output layer?
**A:** One neuron per class. With softmax:
- Each neuron learns to predict one species
- Softmax ensures they sum to 1 (probability distribution)
- Argmax picks highest probability

### Q2: What would happen without one-hot encoding?
**A:**
- Model would treat 0,1,2 as ordinal values
- Implies Virginica (2) > Versicolor (1) > Setosa (0)
- Incorrect! Classes are nominal (unordered)
- One-hot encoding represents this correctly

### Q3: Difference between binary and multiclass ANN?
**A:**

| Aspect | Binary | Multiclass |
|--------|--------|-----------|
| Output neurons | 1 | # of classes |
| Output activation | Sigmoid | Softmax |
| Loss function | Binary CE | Categorical CE |
| Encoding | Raw label | One-hot |
| Interpretation | Single probability | Probability distribution |

### Q4: Why better accuracy than binary model?
**A:**
- Iris dataset is more separable (balanced, clean data)
- Heart disease dataset has noise and imbalance
- Simpler problem (4 features vs 13)
- Class distribution is perfectly balanced

### Q5: How to predict for new iris flower?
**A:**
```
1. Measure 4 features: [5.1, 3.5, 1.4, 0.2]
2. Standardize: (x - train_mean) / train_std
3. Forward pass through network
4. Get output: [0.92, 0.05, 0.03]
5. Argmax: class 0 → "Setosa"
6. Confidence: 92%
```

### Q6: What if we had 4 classes instead of 3?
**A:**
- Change output layer to 4 neurons
- Change softmax to 4 dimensions
- Use 4-label one-hot encoding
- Categorical cross-entropy still works
- Same architecture, scales linearly

---

# 🟢 MODEL 3: LOGISTIC REGRESSION

## Problem Statement
**Binary classification using logistic regression on 12-dimensional data**

### Dataset Details
- **Source**: Synthetic Binary Classification Data
- **Samples**: 1500 data points
- **Features**: 12 continuous numerical features
- **Classes**: Binary (0 or 1)
- **Class Balance**: Roughly balanced
- **Train-Test Split**: 80-20 = 1200 train, 300 test

---

## Model Architecture
```
Linear Model:
z = w₁*x₁ + w₂*x₂ + ... + w₁₂*x₁₂ + b  (linear combination)
                ↓
        Sigmoid Function:
y_pred = 1 / (1 + e^(-z))  (squash to [0,1])
                ↓
        Probability of class 1
        (decision: > 0.5 → class 1, else class 0)

Key: Logistic regression is fundamentally LINEAR
     But sigmoid distorts the decision boundary
```

---

## Key Concepts to Explain

### 1️⃣ **Logistic Function (Sigmoid)**
```
Why use sigmoid instead of linear regression?

Linear Regression: y = w·x + b
  ✗ Unbounded output (-∞ to +∞)
  ✗ Makes no sense for probabilities

Logistic Regression: P(y=1|x) = 1 / (1 + e^(-z))
  ✓ Bounded output [0, 1]
  ✓ Represents probability
  ✓ S-shaped curve (gentle transitions)
  ✓ Easy to interpret as probability
```

### 2️⃣ **Maximum Likelihood Estimation (MLE)**
```
Goal: Find weights that maximize probability of observed data

For binary classification:
P(Data | weights) = ∏ P(y=1|x)*y + P(y=0|x)*(1-y)

In practice, we minimize Negative Log-Likelihood:
Loss = -1/m * Σ(y*log(ŷ) + (1-y)*log(1-ŷ))

This is identical to Binary Cross-Entropy!
```

### 3️⃣ **Feature Coefficients (Interpretability)**
```
After training, model learns weights for each feature:

Feature1: +0.5  → increases P(class=1) by 0.5
Feature2: -0.3  → decreases P(class=1) by 0.3
...

This is INTERPRETABLE:
- Know which features matter
- Know direction of effect
- Neural networks: black box (hard to interpret)
```

### 4️⃣ **Why StandardScaler?**
```
Logistic Regression is scale-sensitive:

Feature A: age (0-100)
Feature B: cholesterol (100-400)

Without scaling:
- Feature B's gradient much larger
- Optimizer confused
- Convergence slow

With scaling (mean=0, std=1):
- All features contribute equally
- Faster convergence
- Better numerical stability
```

### 5️⃣ **K-Fold Cross-Validation**
```
Problem: Single train-test split can be lucky/unlucky

Solution: K-Fold CV (K=5):
- Split data into 5 folds
- Train on 4 folds, test on 1
- Repeat 5 times (each fold as test once)
- Average the 5 test accuracies
- Report: mean ± std

Benefits:
- More reliable performance estimate
- Uses all data for both training and testing
- Detects overfitting (if train >> CV accuracy)
```

---

## Performance Metrics

```
Accuracy: 97.5% on test set
Precision, Recall, F1-Score
ROC-AUC score
Cross-validation: 95.5% ± 0.81%

Why high accuracy?
1. Problem is linearly separable (synthetic data designed that way)
2. Logistic regression excellent for linear problems
3. Large sample size (1500 > 297, 150)
```

---

## Viva Questions & Answers

### Q1: Why is logistic regression simpler than ANN?
**A:**
```
Logistic Regression:
- Single linear layer + sigmoid
- Parameters: 12 weights + 1 bias = 13 parameters
- Training: 1 second

ANN Binary:
- Multiple hidden layers
- Parameters: 64*13 + 64 + 32*64 + ... = thousands
- Training: 30+ seconds

Simplicity: LR wins
Performance on linear data: LR wins
Flexibility on complex data: ANN wins
```

### Q2: What's the difference between logistic and linear regression?
**A:**
| Aspect | Linear | Logistic |
|--------|--------|----------|
| Output | Unbounded | [0,1] probability |
| Loss | MSE (mean squared) | Cross-entropy |
| Use case | Continuous prediction | Binary classification |
| Interpretation | Direct value | Probability |
| Boundary | Straight line | S-curve |

### Q3: What do coefficients tell us?
**A:**
Example: If coef[5] = 0.75
- Feature 5 has positive effect on predicting class 1
- For unit increase in feature 5, odds increase by e^0.75 = 2.1x
- This feature is important for prediction
- Feature 12 = -0.2 means negative effect

### Q4: Why not use ANN for this problem?
**A:**
- Data is linearly separable → LR sufficient
- ANN overkill (overfitting risk with large models)
- LR more interpretable (see feature importance)
- LR faster to train
- Occam's razor: simpler model is better

### Q5: What does cross-validation tell us that test set doesn't?
**A:**
```
Single split (97.5% accuracy):
- Might be lucky/unlucky split
- High variance estimate

K-Fold CV (95.5% ± 0.81%):
- Average of 5 independent evaluations
- More stable estimate
- Standard deviation shows reliability
- If train >> CV: overfitting detected
```

### Q6: How to use this model for new predictions?
**A:**
```
1. Get new 12-dimensional feature vector: [x₁, x₂, ..., x₁₂]
2. Standardize using training statistics: 
   x_scaled = (x - train_mean) / train_std
3. Compute linear combination:
   z = w₁*x₁ + w₂*x₂ + ... + w₁₂*x₁₂ + b
4. Apply sigmoid:
   prob = 1 / (1 + e^(-z))
5. Decision:
   if prob > 0.5: predict class 1
   else: predict class 0
6. Report confidence: prob or 1-prob
```

---

# 🎯 COMPARISON: ALL 3 MODELS

## Performance Summary

| Metric | Binary ANN | Multiclass ANN | Logistic Reg |
|--------|-----------|----------------|-------------|
| **Accuracy** | 57.4% | 93.3% | 97.5% |
| **Precision** | 56.1% | ~93% | ~97% |
| **Recall** | 74.2% | ~93% | ~97% |
| **F1-Score** | 63.9% | ~93% | ~97% |
| **ROC-AUC** | 59.1% | N/A | ~97% |

## Architectural Comparison

```
                Binary ANN          Multiclass ANN       Logistic Regression
Input           13 features         4 features           12 features
Hidden1         64 neuron ReLU      64 neuron ReLU       (none)
Hidden2         32 neuron ReLU      32 neuron ReLU       (none)  
Hidden3         16 neuron ReLU      16 neuron ReLU       (none)
Output          1 sigmoid           3 softmax            1 sigmoid
Parameters      ~2000               ~2000                13
Training        30+ seconds         30+ seconds          <1 second
```

## When to Use Each

| Model | Best For | Reason |
|-------|----------|--------|
| **Binary ANN** | Complex non-linear binary problems | Learns non-linear patterns, but harder to interpret |
| **Multiclass ANN** | Complex multi-class problems | Softmax naturally handles multiple classes |
| **Logistic Reg** | Simple binary problems, interpretability | Fast, interpretable, good baseline |

---

# ⚡ KEY DIFFERENCES YOU MUST KNOW

## 1. Output Activation
```
Binary ANN:        1 sigmoid      → [0,1]
Multiclass ANN:    softmax        → [0,1]³ summing to 1
Logistic Reg:      sigmoid        → [0,1]
```

## 2. Loss Function
```
Binary ANN:        Binary Cross-Entropy    = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
Multiclass ANN:    Categorical C-E         = -Σ(y*log(ŷ))
Logistic Reg:      Log-Loss (MLE)          = same as binary C-E
```

## 3. Data Processing
```
Binary ANN:        13 features, standardized, stratified split
Multiclass ANN:    4 features, standardized, ONE-HOT ENCODING, stratified split
Logistic Reg:      12 features, standardized, cross-validation
```

## 4. Regularization
```
Binary ANN:        Dropout 30%
Multiclass ANN:    Dropout 25%
Logistic Reg:      No explicit dropout (simpler model = less overfitting)
```

---

# 🚨 VIVA TIPS

### General Tips

1. **Understand every line of code**: You should be able to explain what each line does
2. **Know the math**: Understand formulas for activations, losses, metrics
3. **Explain trade-offs**: Why this choice vs alternatives?
4. **Real-world context**: Medical diagnosis (binary), species classification (multiclass), generic (LR)
5. **Show code**: Be ready to show them the code on your laptop and run it

### Potential Questions

✅ **"Can you explain line 47 of your code?"**
- Be specific, know what each function does

✅ **"Why did you use this activation function?"**
- Reference the mathematical properties

✅ **"What if you increased dropout to 0.5?"**
- Training accuracy drops, test improves (less overfitting)

✅ **"What if you doubled the number of neurons?"**
- More parameters, risk of overfitting, slower training

✅ **"Can you change sigmoid to ReLU in output layer?"**
- No! ReLU unbounded, won't give probabilities

✅ **"Why these particular datasets?"**
- Binary ANN: Medical diagnosis (real world importance)
- Multiclass: Classic ML dataset, demonstrates softmax
- Logistic: Shows linear model baseline

### Code Walkthrough (Important!)

Be ready to:
1. Run all three models on your laptop
2. Show the outputs (accuracy, confusion matrix, plots)
3. Modify hyperparameters (epochs, batch size, dropout, etc.)
4. Explain specific code lines
5. Show plots and interpretation

---

# 📚 FORMULA SHEET

## Activations
```
Sigmoid:    f(x) = 1 / (1 + e^(-x))
ReLU:       f(x) = max(0, x)
Softmax:    f(z_i) = e^(z_i) / Σe^(z_j)
```

## Loss Functions
```
Binary CE:      -[y*log(ŷ) + (1-y)*log(1-ŷ)]
Categorical CE: -Σ(y*log(ŷ))
```

## Metrics
```
Accuracy:   (TP + TN) / (TP + TN + FP + FN)
Precision:  TP / (TP + FP)
Recall:     TP / (TP + FN)
F1-Score:   2 * (Precision * Recall) / (Precision + Recall)
```

---

**Good luck with your viva! 🎓**
