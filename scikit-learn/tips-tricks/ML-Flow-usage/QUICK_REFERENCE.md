# ML Workflow Quick Reference Guide

This is a condensed reference for the complete ML workflow. Use this as a checklist when building your own ML projects.

## 🎯 The 8-Step Workflow

```
1. Data & Split          → Train-test split (hold out test!)
2. Preprocessing         → Pipeline with imputation, scaling
3. Hyperparameter Tuning → GridSearchCV with CV
4. Calibration          → CalibratedClassifierCV
5. Test Evaluation      → All metrics + plots
6. Learning Curve       → Diagnose bias vs variance
7. Validation Curve     → Find optimal complexity
8. Runtime Analysis     → Speed vs accuracy tradeoff
```

---

## 📊 When to Use What

### Metrics

| Metric | Use When | Interpretation |
|--------|----------|----------------|
| **ROC-AUC** | General binary classification | 0.5 = random, 1.0 = perfect |
| **PR-AUC** | Imbalanced classes | Compare to prevalence baseline |
| **F1** | Need single threshold | Harmonic mean of precision/recall |
| **Accuracy** | Balanced classes | Simple, but misleading for imbalance |
| **Log Loss** | Probability quality matters | Lower is better, penalizes confidence |

### Cross-Validation

| Method | Use When | Training Cost |
|--------|----------|---------------|
| **KFold** | Balanced data, general use | N models |
| **StratifiedKFold** | Classification, preserve class balance | N models |
| **GroupKFold** | Grouped data (time series, patients) | N models |
| **LeaveOneOut** | Very small datasets (< 100 samples) | N models (expensive!) |
| **ShuffleSplit** | Want bootstrap-like splits | N models |

**Default choice**: `StratifiedKFold(n_splits=5)` for classification

### Hyperparameter Tuning

| Method | Use When | Search Strategy |
|--------|----------|-----------------|
| **GridSearchCV** | Small search space (< 100 combos) | Exhaustive |
| **RandomizedSearchCV** | Large search space (> 100 combos) | Random sampling |
| **HalvingGridSearchCV** | Very large search space | Successive halving |
| **BayesSearchCV** | Expensive models | Bayesian optimization |

**Default choice**: `GridSearchCV` for < 50 combinations, `RandomizedSearchCV` otherwise

---

## 🔍 Learning Curve Interpretation

```python
# Pattern 1: High Bias (Underfitting)
Train: ████──── 0.75
Val:   ███───── 0.72
Gap:   Small
→ Both low, plateau early
→ Solution: More complex model

# Pattern 2: High Variance (Overfitting)
Train: █████████ 0.95
Val:   ██████─── 0.70
Gap:   Large
→ Large gap, train high, val low
→ Solution: More data, regularization

# Pattern 3: Good Fit
Train: ████████ 0.92
Val:   ███████─ 0.89
Gap:   Small
→ Both high, small gap
→ Solution: You're done! 🎉

# Pattern 4: Need More Data
Train: ████████↗ (rising)
Val:   ███████─↗ (rising)
Gap:   Decreasing
→ Curves haven't plateaued
→ Solution: Collect more data
```

---

## 🎚️ Validation Curve Interpretation

For **LogisticRegression's C parameter** (inverse regularization strength):

```
C Value:   0.001      0.1       1.0       10.0      1000.0
           ←─────── Simpler         Complex ───────→
           ←─────── More regularization ───────→

Scenario 1: Underfitting (Need more complexity)
Train:     ████       ████      ████      ████      ████
Val:       ███        ███       ███       ███       ███
→ Increasing C doesn't help → Try more features/polynomial features

Scenario 2: Optimal
Train:     ████       ██████    ████████  ████████  ████████
Val:       ███        █████     ███████   ██████    ████
           ↑                    ↑         
           Underfit             Sweet spot  
→ C=1.0 is optimal

Scenario 3: Overfitting (Need regularization)
Train:     ████       ██████    ████████  █████████ ██████████
Val:       ███        █████     ███████   ██████    ████
→ Smaller C (more regularization) needed
```

---

## 🔄 Pipeline Best Practices

### ✅ DO

```python
# ✅ Correct: Pipeline prevents data leakage
pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Fit on training data only
pipeline.fit(X_train, y_train)

# Transform test data using training statistics
predictions = pipeline.predict(X_test)
```

### ❌ DON'T

```python
# ❌ Wrong: Data leakage!
# Scaling on all data before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # BAD: Test data influences scaling

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# ❌ Wrong: Refitting on test data!
scaler.fit(X_test)  # BAD: Never fit on test data
X_test_scaled = scaler.transform(X_test)
```

---

## 📈 Calibration Check

**When to calibrate:**
- Probability values matter (not just rankings)
- Using probabilities for decision thresholds
- Comparing models across different datasets
- Risk scores, medical diagnosis, etc.

**How to check if calibration is needed:**
```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

# Perfect calibration: prob_true ≈ prob_pred
# If plot deviates from y=x line, calibrate!
```

**Calibration methods:**
- `'sigmoid'` (Platt scaling): Good for LogReg, SVM, trees
- `'isotonic'`: More flexible, needs more data (> 1000 samples)

---

## ⚡ Performance Optimization

### Speed Up Training

```python
# 1. Use fewer CV folds
GridSearchCV(..., cv=3)  # Instead of cv=5

# 2. Use RandomizedSearchCV
RandomizedSearchCV(..., n_iter=20)  # Instead of GridSearchCV

# 3. Use parallel processing
GridSearchCV(..., n_jobs=-1)  # Use all CPU cores

# 4. Reduce hyperparameter grid
param_grid = {
    'C': [0.1, 1.0, 10.0],  # Instead of [0.01, 0.1, 1.0, 10.0, 100.0]
}
```

### Speed Up Prediction

```python
# 1. Use simpler model
LogisticRegression()  # Fast
# vs
RandomForestClassifier(n_estimators=1000)  # Slow

# 2. Use fewer features
SelectKBest(k=10)  # Select top 10 features

# 3. Reduce tree depth
RandomForestClassifier(max_depth=5)  # Instead of max_depth=None

# 4. Use fewer trees
RandomForestClassifier(n_estimators=50)  # Instead of 100
```

---

## 🎯 MLflow Tracking Pattern

### Standard Tracking Template

```python
import mlflow

mlflow.set_experiment('my_experiment')

with mlflow.start_run(run_name='experiment_v1'):
    
    # 1. Log parameters (configurations)
    mlflow.log_param('model_type', 'LogisticRegression')
    mlflow.log_param('C', 1.0)
    mlflow.log_param('cv_folds', 5)
    
    # 2. Train model
    model = train_model(X_train, y_train)
    
    # 3. Log metrics (results)
    mlflow.log_metric('train_roc_auc', 0.95)
    mlflow.log_metric('test_roc_auc', 0.92)
    mlflow.log_metric('train_time', 2.5)
    
    # 4. Log artifacts (files)
    mlflow.log_artifact('plots/roc_curve.png')
    mlflow.log_artifact('results/metrics.csv')
    
    # 5. Log model
    mlflow.sklearn.log_model(model, 'model')
```

### Querying Logged Runs

```python
from mlflow import MlflowClient

client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name('my_experiment')

# Find best run by metric
runs = client.search_runs(
    experiment.experiment_id,
    order_by=['metrics.test_roc_auc DESC'],
    max_results=5
)

# Load best model
best_run_id = runs[0].info.run_id
model = mlflow.sklearn.load_model(f'runs:/{best_run_id}/model')
```

---

## 🚨 Common Pitfalls

### 1. Data Leakage
```python
# ❌ Wrong
scaler.fit(X)  # Fit on all data
X_train, X_test = train_test_split(X)

# ✅ Correct
X_train, X_test = train_test_split(X)
scaler.fit(X_train)  # Fit only on training data
```

### 2. Using Test Set for Tuning
```python
# ❌ Wrong
for C in [0.1, 1.0, 10.0]:
    model = LogisticRegression(C=C)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # BAD: Tuning on test set!

# ✅ Correct
grid_search = GridSearchCV(
    LogisticRegression(),
    {'C': [0.1, 1.0, 10.0]},
    cv=5  # Use CV on training data
)
grid_search.fit(X_train, y_train)
final_score = grid_search.score(X_test, y_test)  # Test set only for final eval
```

### 3. Ignoring Class Imbalance
```python
# ❌ Wrong
# Dataset: 95% class 0, 5% class 1
# Model that always predicts class 0: 95% accuracy!

# ✅ Correct
# Use appropriate metrics
roc_auc_score(y_test, y_pred_proba)  # Threshold-independent
average_precision_score(y_test, y_pred_proba)  # PR-AUC

# Or balance classes
LogisticRegression(class_weight='balanced')
```

### 4. Not Setting Random State
```python
# ❌ Wrong
model = RandomForestClassifier()  # Different results each run

# ✅ Correct
model = RandomForestClassifier(random_state=42)  # Reproducible
```

---

## 📋 Pre-Flight Checklist

Before deploying your model, verify:

- [ ] Test set was never used for training or tuning
- [ ] All preprocessing steps are in the pipeline
- [ ] Cross-validation was used for hyperparameter tuning
- [ ] Model is calibrated (if probabilities matter)
- [ ] Learning curves show no severe overfitting
- [ ] Validation curves show optimal hyperparameter
- [ ] Runtime is acceptable for production use
- [ ] All metrics are logged to MLflow
- [ ] Model is saved and versioned
- [ ] Documentation explains model decisions

---

## 🔑 Key Formulas

### Precision
```
Precision = TP / (TP + FP)
"Of all positive predictions, how many were correct?"
```

### Recall (Sensitivity, TPR)
```
Recall = TP / (TP + FN)
"Of all actual positives, how many did we catch?"
```

### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Harmonic mean of precision and recall
```

### Specificity (TNR)
```
Specificity = TN / (TN + FP)
"Of all actual negatives, how many did we correctly identify?"
```

### False Positive Rate
```
FPR = FP / (FP + TN) = 1 - Specificity
Used in ROC curve (x-axis)
```

---

## 🎓 Decision Rules Summary

| Question | Answer |
|----------|--------|
| **What metric?** | Balanced → ROC-AUC; Imbalanced → PR-AUC |
| **What CV?** | Classification → StratifiedKFold(5) |
| **What search?** | < 50 combos → Grid; > 50 → Randomized |
| **What preprocessing?** | Always: Impute → Scale → Encode |
| **Calibrate?** | If probabilities matter: Yes |
| **How many folds?** | Default: 5; Small data: 10; Large data: 3 |
| **Train-test split?** | Default: 80/20 or 70/30 |
| **Random state?** | Always set for reproducibility |

---

**Reference this guide when building ML workflows!** 🚀
