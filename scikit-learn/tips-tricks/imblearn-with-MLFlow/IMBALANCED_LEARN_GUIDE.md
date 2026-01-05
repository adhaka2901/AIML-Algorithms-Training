# Imbalanced-Learn Integration Guide

## 🎯 Why We Didn't Use It in the Original Example

The original example used **`class_weight='balanced'`** in LogisticRegression, which is a **cost-sensitive** approach:
- Assigns higher penalty to minority class errors
- No change to data distribution
- Works well for mild-to-moderate imbalance
- Simpler, no external library needed

**Imbalanced-learn** provides **sampling techniques** that actually change the training data distribution, which can be more powerful for severe imbalance.

---

## 🔑 Critical Concept: imblearn.pipeline vs sklearn.pipeline

### ❌ WRONG: Using sklearn.pipeline.Pipeline

```python
from sklearn.pipeline import Pipeline  # ❌ Doesn't support samplers
from imblearn.over_sampling import SMOTE

# This will FAIL or cause data leakage!
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('sampler', SMOTE()),  # ❌ sklearn Pipeline doesn't know about samplers
    ('classifier', LogisticRegression())
])
```

**Problem**: sklearn's Pipeline only knows about transformers (fit_transform) and estimators (fit/predict). It doesn't understand samplers (fit_resample).

### ✅ CORRECT: Using imblearn.pipeline.Pipeline

```python
from imblearn.pipeline import Pipeline  # ✅ Supports samplers
from imblearn.over_sampling import SMOTE

# This works correctly!
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('sampler', SMOTE()),  # ✅ imblearn Pipeline handles samplers
    ('classifier', LogisticRegression())
])
```

**Why this matters**: When you use `cross_val_score` or `GridSearchCV`, sampling must happen **INSIDE each CV fold**, not before CV. imblearn's Pipeline ensures this.

---

## 📊 How Cross-Validation Works with Sampling

### Without Sampling (Original Data):

```
CV Fold 1:
├── Train: [1000 samples] → Fit model
└── Val:   [250 samples]  → Evaluate

CV Fold 2:
├── Train: [1000 samples] → Fit model
└── Val:   [250 samples]  → Evaluate
...
```

### With Sampling (Using imblearn.pipeline.Pipeline):

```
CV Fold 1:
├── Train: [1000 samples] 
│   ├── Class 0: 950 samples (95%)
│   └── Class 1: 50 samples (5%)
│   ↓ Apply SMOTE (INSIDE this fold only)
│   ├── Class 0: 950 samples
│   └── Class 1: 950 samples (generated 900 synthetic)
│   → Fit model on RESAMPLED train fold
└── Val: [250 samples] (ORIGINAL distribution)
    ├── Class 0: 238 samples (95%)
    └── Class 1: 12 samples (5%)
    → Evaluate on ORIGINAL val fold

CV Fold 2:
├── Train: [1000 samples] → RESAMPLE again (different samples!)
└── Val:   [250 samples]  → Evaluate on ORIGINAL distribution
...
```

**Key points**:
1. Sampling happens **independently** in each CV fold
2. Validation fold keeps **original** class distribution
3. Test set is **never** resampled
4. This prevents data leakage

---

## 🔬 Sampling Techniques Deep Dive

### 1. Over-Sampling (Increase Minority Class)

#### RandomOverSampler
```python
from imblearn.over_sampling import RandomOverSampler

sampler = RandomOverSampler(random_state=42)
```

**How it works**:
- Randomly duplicates minority class samples
- Simple: just copy existing samples

**Math**:
```
Original: [x1, x2, x3, ...]  (50 minority samples)
After:    [x1, x2, x3, ..., x1, x2, ...]  (950 samples)
                                ↑
                         Exact duplicates
```

**Pros**:
- Simple, fast
- Preserves all information
- No risk of creating noisy samples

**Cons**:
- Exact duplicates → risk of overfitting
- Model may memorize specific samples

**When to use**:
- Baseline comparison
- Very small datasets (< 100 minority samples)
- When data quality is critical

---

#### SMOTE (Synthetic Minority Over-sampling Technique)
```python
from imblearn.over_sampling import SMOTE

sampler = SMOTE(k_neighbors=5, random_state=42)
```

**How it works**:
1. For each minority sample `x`:
   - Find k nearest minority neighbors
   - Randomly pick one neighbor `x_neighbor`
   - Generate synthetic sample: `x_new = x + λ * (x_neighbor - x)` where λ ∈ [0, 1]

**Visual**:
```
    x1 ●─────────● x_neighbor
        │  λ=0.3  │
        ├─────●   │  ← x_new (synthetic)
```

**Math example**:
```
x1 = [1.0, 2.0, 3.0]
x2 = [2.0, 3.0, 4.0]  (nearest neighbor)

λ = 0.6 (random)

x_new = x1 + 0.6 * (x2 - x1)
      = [1.0, 2.0, 3.0] + 0.6 * [1.0, 1.0, 1.0]
      = [1.6, 2.6, 3.6]
```

**Pros**:
- No exact duplicates
- Smooths decision boundary
- Most widely used, good default

**Cons**:
- Can create noisy samples in overlap regions
- Assumes linear interpolation is valid
- Sensitive to outliers

**When to use**:
- **Most common choice**
- Moderate to severe imbalance (5%-30% minority)
- Clean data with clear minority patterns

---

#### ADASYN (Adaptive Synthetic Sampling)
```python
from imblearn.over_sampling import ADASYN

sampler = ADASYN(n_neighbors=5, random_state=42)
```

**How it works**:
1. Like SMOTE, but **adaptive density**
2. Calculates difficulty of learning each minority sample:
   - Difficulty = proportion of majority neighbors
3. Generates more synthetics where minority is harder to learn

**Math**:
```
For minority sample xi:
  Γi = # of majority neighbors / k
  
More synthetics generated where Γi is high
(i.e., minority sample surrounded by majority)
```

**Visual**:
```
Majority region:
    ○ ○ ○ ○
    ○ ● ○ ○  ← Minority sample (high Γ)
    ○ ○ ○ ○  → Generate MORE synthetics here
    
Minority region:
    ● ● ●
    ● ● ●    ← Minority samples (low Γ)
    ● ● ●    → Generate FEWER synthetics here
```

**Pros**:
- Adaptive to data density
- Focuses on decision boundary
- Better for complex boundaries

**Cons**:
- More complex than SMOTE
- Can amplify noise
- Slower computation

**When to use**:
- Complex decision boundaries
- Non-uniform minority distribution
- When you want to focus on hard-to-learn regions

---

#### SVMSMOTE
```python
from imblearn.over_sampling import SVMSMOTE

sampler = SVMSMOTE(random_state=42, k_neighbors=5)
```

**How it works**:
1. Trains SVM on current data
2. Identifies support vectors (samples near decision boundary)
3. Applies SMOTE only to support vectors

**Why this helps**:
- Support vectors define the decision boundary
- Generating synthetics there is most impactful
- Avoids creating synthetics in safe regions

**Pros**:
- Focuses on decision boundary
- Reduces noise in safe regions
- Often better performance than plain SMOTE

**Cons**:
- Slower (needs SVM training)
- More complex
- Sensitive to SVM parameters

**When to use**:
- When decision boundary quality is critical
- Moderate datasets (SVM is expensive for large data)
- Clean, well-separated classes

---

### 2. Under-Sampling (Decrease Majority Class)

#### RandomUnderSampler
```python
from imblearn.under_sampling import RandomUnderSampler

sampler = RandomUnderSampler(random_state=42)
```

**How it works**:
- Randomly removes majority class samples

**Math**:
```
Original: 950 majority, 50 minority
After:    50 majority, 50 minority (removed 900 randomly)
```

**Pros**:
- Fast
- Reduces training time
- No risk of creating noisy samples

**Cons**:
- **Loses information** (throws away data)
- May discard important samples
- Can underfit if majority has diverse patterns

**When to use**:
- Large datasets (> 100k samples)
- Majority class has redundant information
- Training time is critical

---

#### TomekLinks
```python
from imblearn.under_sampling import TomekLinks

sampler = TomekLinks()
```

**How it works**:
1. Tomek link = pair of samples from different classes that are each other's nearest neighbors
2. Removes majority samples in Tomek links

**Visual**:
```
Before:
    ● ─── ○  ← Tomek link (nearest neighbors from different classes)
    ●     ○
    ● ●   ○ ○
    
After:
    ●     
    ●     ○  ← Majority sample removed
    ● ●   ○ ○
```

**Why this helps**:
- Removes borderline/noisy majority samples
- Cleans decision boundary
- Makes classes more separable

**Pros**:
- Cleans noisy boundaries
- Removes ambiguous cases
- Often used with over-sampling

**Cons**:
- May not balance classes fully
- Only removes pairs
- Not aggressive enough for severe imbalance

**When to use**:
- **Combined with over-sampling** (see SMOTE-Tomek)
- Noisy data
- When decision boundary needs cleaning

---

#### NearMiss
```python
from imblearn.under_sampling import NearMiss

sampler = NearMiss(version=1)  # version: 1, 2, or 3
```

**How it works**:
- **Version 1**: Select majority samples with smallest average distance to k minority neighbors
- **Version 2**: Select majority samples with smallest average distance to k farthest minority neighbors
- **Version 3**: For each minority sample, select m majority neighbors

**Why this helps**:
- Keeps informative majority samples (close to boundary)
- Removes redundant majority samples far from boundary

**Pros**:
- Intelligent selection (not random)
- Preserves boundary information

**Cons**:
- Can create too clean separation
- Sensitive to k parameter
- May remove important samples

**When to use**:
- When majority class has clear clusters
- Need balanced dataset quickly
- Experimental (compare with other methods)

---

### 3. Hybrid Techniques (Combine Over & Under Sampling)

#### SMOTEENN (SMOTE + Edited Nearest Neighbours)
```python
from imblearn.combine import SMOTEENN

sampler = SMOTEENN(random_state=42)
```

**How it works**:
1. **Step 1**: Apply SMOTE (over-sample minority)
2. **Step 2**: Apply ENN (remove samples misclassified by k-NN)

**Process**:
```
Original:     950 majority, 50 minority

After SMOTE:  950 majority, 950 minority

After ENN:    800 majority, 900 minority
              (removed noisy samples from both classes)
```

**Why this helps**:
- SMOTE balances classes
- ENN removes noise and outliers
- Cleaner decision boundary

**Pros**:
- Balances classes AND cleans data
- Removes overlapping regions
- Often best for noisy data

**Cons**:
- Can be too aggressive
- May remove too many samples
- Slower than single technique

**When to use**:
- **Noisy data** with severe imbalance
- When data quality is poor
- Overlapping class regions

---

#### SMOTETomek (SMOTE + Tomek Links)
```python
from imblearn.combine import SMOTETomek

sampler = SMOTETomek(random_state=42)
```

**How it works**:
1. **Step 1**: Apply SMOTE (over-sample minority)
2. **Step 2**: Remove Tomek links (clean boundary)

**Process**:
```
Original:     950 majority, 50 minority

After SMOTE:  950 majority, 950 minority

After Tomek:  930 majority, 945 minority
              (removed borderline cases)
```

**Pros**:
- Balances classes
- Cleans decision boundary
- **Most robust choice** for severe imbalance
- Less aggressive than SMOTEENN

**Cons**:
- More complex
- Slower than single technique
- Two hyperparameter sets

**When to use**:
- **Most recommended** for severe imbalance
- Production systems
- When both balance and clean boundary matter

---

## 📋 Decision Tree: Which Technique to Use?

```
Is your data imbalanced?
├─ No (> 40% minority) → No sampling needed, use class_weight='balanced'
└─ Yes
    ├─ Is it severe (< 10% minority)?
    │   ├─ Yes, and data is noisy?
    │   │   └─ Use SMOTEENN or SMOTETomek
    │   └─ Yes, and data is clean?
    │       └─ Use SMOTE or ADASYN
    └─ Is it moderate (10-30% minority)?
        ├─ Large dataset (> 100k)?
        │   └─ Use RandomUnderSampler or NearMiss
        └─ Small dataset (< 100k)?
            └─ Use SMOTE

For most cases: Start with SMOTE, compare with SMOTETomek
```

---

## 🎯 Evaluation Metrics for Imbalanced Data

### ❌ DON'T Use: Accuracy

```python
# Example: 95% class 0, 5% class 1
y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #                                                           ↑
        #                                                  Always predict 0!

accuracy = 19/20 = 95%  # Looks great, but model is useless!
```

### ✅ DO Use: Imbalance-Aware Metrics

1. **ROC-AUC** (Receiver Operating Characteristic - Area Under Curve)
   ```python
   from sklearn.metrics import roc_auc_score
   
   roc_auc = roc_auc_score(y_test, y_pred_proba)
   # 0.5 = random, 1.0 = perfect
   ```
   - **Threshold-independent**: Considers all thresholds
   - **Good for ranking**: How well does model rank predictions?
   - **Limitation**: Can be misleading for severe imbalance

2. **PR-AUC** (Precision-Recall - Area Under Curve) ⭐
   ```python
   from sklearn.metrics import average_precision_score
   
   pr_auc = average_precision_score(y_test, y_pred_proba)
   # Compare to prevalence baseline
   ```
   - **Best for imbalanced data**
   - **Focuses on minority class**: More sensitive to minority performance
   - **Compare to baseline**: PR-AUC should be >> prevalence

3. **F1 Score**
   ```python
   from sklearn.metrics import f1_score
   
   f1 = f1_score(y_test, y_pred)
   # Harmonic mean of precision and recall
   ```
   - **Balances precision and recall**
   - **Single threshold**: Needs threshold selection
   - **Good for**: When both false positives and false negatives matter

4. **Balanced Accuracy**
   ```python
   from sklearn.metrics import balanced_accuracy_score
   
   bal_acc = balanced_accuracy_score(y_test, y_pred)
   # Average of recall for each class
   ```
   - **Not fooled by imbalance**: Treats classes equally
   - **Easy to interpret**: Similar to accuracy
   - **Good baseline metric**

---

## 🚨 Common Pitfalls

### 1. Resampling Test Set ❌

```python
# ❌ WRONG: Resampling test set
X_train, X_test, y_train, y_test = train_test_split(X, y)

# DON'T DO THIS!
sampler = SMOTE()
X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
X_test_res, y_test_res = sampler.fit_resample(X_test, y_test)  # ❌❌❌

# ✅ CORRECT: Only resample training set
X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
# Test set stays original!
```

**Why?** Test set should reflect real-world distribution.

---

### 2. Resampling Before Train-Test Split ❌

```python
# ❌ WRONG: Resample before split
sampler = SMOTE()
X_res, y_res = sampler.fit_resample(X, y)  # ❌ Resampling all data

# Then split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)
# Problem: Synthetic samples in test set!

# ✅ CORRECT: Split first, then resample
X_train, X_test, y_train, y_test = train_test_split(X, y)
sampler = SMOTE()
X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
```

**Why?** Synthetic samples shouldn't be in test set.

---

### 3. Resampling Outside CV Folds ❌

```python
# ❌ WRONG: Resample before CV
sampler = SMOTE()
X_res, y_res = sampler.fit_resample(X_train, y_train)

# Then CV
scores = cross_val_score(model, X_res, y_res, cv=5)  # ❌ Data leakage!

# ✅ CORRECT: Use imblearn.pipeline.Pipeline
from imblearn.pipeline import Pipeline

pipeline = Pipeline([
    ('sampler', SMOTE()),
    ('classifier', LogisticRegression())
])

# CV handles resampling inside each fold
scores = cross_val_score(pipeline, X_train, y_train, cv=5)
```

**Why?** Validation fold samples may be based on synthetic versions created from samples in the training fold.

---

## 📊 Performance Comparison Template

When comparing techniques, track:

| Technique | ROC-AUC | PR-AUC | F1 | Balanced Acc | Train Time |
|-----------|---------|--------|-----|--------------|------------|
| Baseline (class_weight) | 0.85 | 0.45 | 0.42 | 0.75 | 0.5s |
| RandomOverSampler | 0.87 | 0.48 | 0.46 | 0.78 | 0.6s |
| SMOTE | 0.89 | 0.52 | 0.50 | 0.82 | 0.7s |
| ADASYN | 0.88 | 0.51 | 0.49 | 0.81 | 0.9s |
| RandomUnderSampler | 0.84 | 0.43 | 0.40 | 0.73 | 0.3s |
| SMOTETomek | **0.91** | **0.55** | **0.53** | **0.85** | 1.2s |
| SMOTEENN | 0.90 | 0.54 | 0.52 | 0.84 | 1.5s |

**Winner**: SMOTETomek (best PR-AUC for imbalanced data)

---

## 🎓 Key Takeaways

1. **Use imblearn.pipeline.Pipeline**, not sklearn's
2. **Only resample training data**, never test set
3. **Resample inside CV folds**, not before
4. **Use PR-AUC** as primary metric for imbalanced data
5. **Start with SMOTE**, compare with SMOTETomek
6. **For severe imbalance** (< 5% minority): Use hybrid methods
7. **For large datasets**: Consider under-sampling
8. **Always compare** to baseline (`class_weight='balanced'`)

---

## 📚 Further Reading

- [Imbalanced-Learn Documentation](https://imbalanced-learn.org/)
- [SMOTE Paper](https://arxiv.org/abs/1106.1813)
- [Learning from Imbalanced Data (Survey)](https://dl.acm.org/doi/10.1145/1007730.1007735)
