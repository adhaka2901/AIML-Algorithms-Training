# Complete ML Workflow with MLflow

A comprehensive, production-ready machine learning workflow demonstrating best practices from data loading through model deployment, with full MLflow experiment tracking.

## 🎯 What This Demonstrates

This workflow covers **all 8 essential steps** of a complete ML project:

1. **Data & Split**: Load data and create train-test split
2. **Preprocessing Pipeline**: Build reproducible preprocessing with `Pipeline`
3. **Hyperparameter Tuning**: Use `GridSearchCV` with cross-validation
4. **Model Calibration**: Improve probability estimates with `CalibratedClassifierCV`
5. **Test Evaluation**: Comprehensive metrics and visualization
6. **Learning Curves**: Diagnose bias vs variance
7. **Validation Curves**: Understand model complexity
8. **Runtime Analysis**: Compare model performance vs speed

**Plus**: Full MLflow integration for experiment tracking, artifact logging, and model versioning.

## 📋 Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install numpy pandas scikit-learn matplotlib seaborn mlflow
```

## 🚀 Quick Start

```bash
# Run the complete workflow
python ml_workflow_with_mlflow.py

# View MLflow UI (after running the workflow)
mlflow ui
# Then open: http://localhost:5000
```

## 📁 Output Structure

After running, you'll get:

```
outputs/
├── test_evaluation.png       # ROC, PR, confusion matrix, calibration
├── learning_curve.png         # Training vs validation performance
├── validation_curve.png       # Model complexity analysis
├── runtime_comparison.csv     # Speed comparison table
└── cv_results.csv            # Grid search detailed results

mlruns/                        # MLflow tracking data
└── [experiment_id]/
    └── [run_id]/
        ├── metrics/          # All logged metrics
        ├── params/           # All logged parameters
        ├── artifacts/        # Plots, models, tables
        └── meta.yaml         # Run metadata
```

## 🔍 Step-by-Step Breakdown

### Step 1: Data & Split

**What happens:**
- Generates synthetic classification data (5000 samples, 20 features)
- Adds 5% missing values (realistic scenario)
- Adds 5% label noise (realistic scenario)
- Splits into train (80%) and test (20%) sets

**Why:**
- Test set is completely held out for final evaluation
- Simulates real-world deployment scenario
- Prevents data leakage

**MLflow tracks:**
- Dataset parameters (n_samples, n_features, etc.)
- Train/test split sizes

---

### Step 2: Preprocessing Pipeline

**What happens:**
- Creates a `Pipeline` with three steps:
  1. `SimpleImputer`: Handle missing values (median strategy)
  2. `StandardScaler`: Normalize features (zero mean, unit variance)
  3. `LogisticRegression`: Classifier

**Why:**
- **Pipeline prevents data leakage**: Fit and transform are kept separate
- **Reproducible**: Same preprocessing for train, CV, and test
- **Clean code**: All steps in one object
- **Tunable**: Can include preprocessing params in hyperparameter tuning

**Why these preprocessing steps?**
- **Imputation**: Real data has missing values
- **Scaling**: Gradient descent converges faster with scaled features
- **Regularization**: L2 penalty treats features equally when scaled

**MLflow tracks:**
- Preprocessing strategy choices
- Classifier choice

---

### Step 3: Hyperparameter Tuning

**What happens:**
- `GridSearchCV` tries all combinations of:
  - `C`: [0.01, 0.1, 1.0, 10.0] (regularization strength)
  - `solver`: ['lbfgs', 'liblinear'] (optimization algorithm)
  - `class_weight`: [None, 'balanced'] (handle imbalance)
- Total: 4 × 2 × 2 = 16 combinations
- Each combination evaluated with 5-fold CV
- Total models trained: 5 × 16 = 80 models

**Why GridSearchCV?**
- Systematic: tries all combinations
- Unbiased: uses CV for each evaluation
- Automatic: selects best and retrains on full training data

**Why cross-validation?**
- More robust than single train-val split
- Uses all training data for both training and validation
- Reduces variance in performance estimates

**Why ROC-AUC?**
- Threshold-independent metric
- Good for imbalanced classes
- Easy to interpret (0.5 = random, 1.0 = perfect)

**MLflow tracks:**
- Best hyperparameters
- Best CV score
- All CV results (16 rows showing each combination)

---

### Step 4: Calibrated Model

**What happens:**
- Takes best estimator from GridSearchCV
- Wraps it in `CalibratedClassifierCV`
- Uses Platt scaling (sigmoid method)
- Fits on full training data with 5-fold CV

**Why calibration?**
- Many classifiers output poorly calibrated probabilities
- Calibration fixes this using isotonic regression or Platt scaling
- Important when probability values matter (not just rankings)

**How CalibratedClassifierCV works:**
1. Takes a fitted or unfitted estimator
2. Uses CV to train calibration on held-out folds
3. Final model = base estimator + calibration mapping
4. Averages predictions from all folds

**MLflow tracks:**
- Calibration method
- Number of calibration folds

---

### Step 5: Test Set Evaluation

**What happens:**
- Predicts on held-out test set
- Computes comprehensive metrics:
  - **ROC-AUC**: Overall ranking quality
  - **PR-AUC**: Precision-recall tradeoff (better for imbalance)
  - **F1 Score**: Harmonic mean of precision and recall
  - **Optimal threshold**: Value that maximizes F1

**Four key plots:**

1. **ROC Curve**
   - X-axis: False Positive Rate (FPR)
   - Y-axis: True Positive Rate (TPR)
   - Shows classifier performance at all thresholds
   - Diagonal = random classifier

2. **Precision-Recall Curve**
   - X-axis: Recall (sensitivity)
   - Y-axis: Precision (positive predictive value)
   - Better than ROC for imbalanced datasets
   - Horizontal line = prevalence baseline

3. **Confusion Matrix**
   - Shows actual vs predicted at optimal threshold
   - Reveals error types (false positives vs false negatives)
   - Uses threshold that maximizes F1

4. **Calibration Curve**
   - X-axis: Mean predicted probability
   - Y-axis: Fraction of positives (true probability)
   - Diagonal = perfect calibration
   - Shows if probabilities are reliable

**Why test set?**
- Final, unbiased estimate of performance
- Never seen during training/tuning
- Simulates real-world deployment

**MLflow tracks:**
- All test metrics
- Optimal threshold
- Test evaluation plots

---

### Step 6: Learning Curve

**What happens:**
- Trains model with different training set sizes
- For each size, computes train and validation scores (CV)
- Plots both curves vs training size

**How to interpret:**

| Pattern | Interpretation | Solution |
|---------|---------------|----------|
| Both curves low, plateaued | High bias (underfitting) | More complex model |
| Large gap between curves | High variance (overfitting) | More data, regularization |
| Curves converge to high score | Good fit | ✅ Ready for production |
| Curves still rising | More data would help | Collect more data |

**Example interpretations:**
```
Train: ████████ 0.95
Val:   ██████── 0.70
→ High variance: Model overfitting, needs regularization

Train: ████──── 0.75
Val:   ███───── 0.72
→ High bias: Model underfitting, needs more complexity

Train: ████████ 0.92
Val:   ███████─ 0.89
→ Good fit: Small gap, high performance
```

**MLflow tracks:**
- Final train score
- Final validation score
- Train-val gap (overfitting indicator)
- Learning curve plot

---

### Step 7: Validation Curve (Model Complexity)

**What happens:**
- Varies one hyperparameter (C for LogReg)
- For each value, computes train and validation scores (CV)
- Plots both curves vs hyperparameter value

**For LogisticRegression's C parameter:**
- **Small C** (e.g., 0.01): High regularization → Simple model
- **Large C** (e.g., 100): Low regularization → Complex model

**How to interpret:**

```
           Low C              Optimal C           High C
           (simple)                              (complex)
Train:     ████────          ████████          ██████████
Val:       ████────          ████████          ███████───
           Underfitting      Sweet spot        Overfitting
```

**Sweet spot:**
- Where validation score is highest
- Balance between bias and variance
- Should match value found in GridSearchCV

**MLflow tracks:**
- Optimal C value
- Optimal validation score
- Validation curve plot

---

### Step 8: Runtime Comparison

**What happens:**
- Trains multiple models on same data
- Measures:
  - **Fit time**: Time to train
  - **Predict time**: Time for all test predictions
  - **Per-sample time**: Predict time / n_samples (in ms)
  - **ROC-AUC**: Performance

**Models compared:**
1. **Logistic Regression**
   - Fast training
   - Fast prediction
   - Linear decision boundary
   - Good baseline

2. **Random Forest**
   - Slower training
   - Slower prediction
   - Non-linear decision boundary
   - Often higher accuracy

**Example output:**
```
Model                  Fit Time  Predict Time  Per-Sample  ROC-AUC
Logistic Regression    0.05s     0.002s        0.002ms     0.92
Random Forest          2.50s     0.050s        0.050ms     0.95
```

**Speed vs Accuracy tradeoff:**
- **LogReg**: 50x faster to train, 25x faster to predict, -0.03 AUC
- **RF**: Better accuracy but much slower

**Use case guidance:**
- **Real-time systems** (< 10ms latency): Use LogReg
- **Batch processing**: Use RF if accuracy matters
- **Large datasets**: LogReg scales better

**MLflow tracks:**
- Runtime comparison table
- Hardware note for reproducibility

---

## 🔬 MLflow Integration

### What MLflow Tracks

**Parameters** (configurations):
```python
n_samples=5000
n_features=20
test_size=0.2
imputation_strategy='median'
scaling='StandardScaler'
best_classifier__C=1.0
best_classifier__solver='lbfgs'
calibration_method='sigmoid'
```

**Metrics** (results):
```python
best_cv_roc_auc=0.9234
test_roc_auc=0.9187
test_pr_auc=0.8945
test_f1=0.8512
optimal_threshold=0.42
final_train_score=0.9456
final_val_score=0.9234
train_val_gap=0.0222
```

**Artifacts** (files):
```
artifacts/
├── test_evaluation.png
├── learning_curve.png
├── validation_curve.png
├── runtime_comparison.csv
├── cv_results.csv
└── model/
    ├── model.pkl
    ├── conda.yaml
    └── requirements.txt
```

### Using MLflow UI

1. **Start the UI:**
   ```bash
   mlflow ui
   ```

2. **Compare experiments:**
   - Select multiple runs
   - Compare parameters side-by-side
   - View metric trends
   - Download artifacts

3. **Load a saved model:**
   ```python
   import mlflow
   
   # Load by run ID
   model = mlflow.sklearn.load_model('runs:/[RUN_ID]/model')
   
   # Or load from model registry
   model = mlflow.sklearn.load_model('models:/calibrated_logistic_regression/latest')
   
   # Make predictions
   predictions = model.predict(X_new)
   ```

4. **Query runs programmatically:**
   ```python
   from mlflow import MlflowClient
   
   client = MlflowClient()
   
   # Get best run by metric
   experiment = client.get_experiment_by_name('ml_workflow_demo')
   runs = client.search_runs(
       experiment.experiment_id,
       order_by=['metrics.test_roc_auc DESC'],
       max_results=1
   )
   best_run = runs[0]
   print(f"Best ROC-AUC: {best_run.data.metrics['test_roc_auc']}")
   ```

---

## 🎓 Key Learning Points

### 1. Pipeline Architecture
- **Everything follows fit/transform/predict pattern**
- Prevents data leakage by separating fit and transform
- Enables hyperparameter tuning of preprocessing steps
- Makes code reproducible and production-ready

### 2. Cross-Validation Strategy
- GridSearchCV trains N_folds × N_combinations models
- All are discarded except the best
- Best model is retrained on full training data
- Test set is completely independent

### 3. Calibration Importance
- Raw classifier probabilities may be poorly calibrated
- Calibration maps scores to true probabilities
- Essential when probability values matter (not just rankings)
- Uses CV to avoid overfitting the calibration

### 4. Comprehensive Evaluation
- **ROC-AUC**: Threshold-independent, good for ranking
- **PR-AUC**: Better for imbalanced data
- **Learning curves**: Diagnose bias vs variance
- **Validation curves**: Find optimal complexity
- **Calibration curves**: Assess probability quality

### 5. Production Considerations
- **Runtime analysis**: Speed vs accuracy tradeoff
- **Model serialization**: Save with MLflow
- **Reproducibility**: Log all parameters
- **Monitoring**: Track metrics over time

---

## 🔧 Customization Guide

### Change the Dataset

```python
# Use a real dataset instead of synthetic
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Try Different Models

```python
# In Step 2, replace LogisticRegression with:
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Update param_grid in Step 3:
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5, 10]
}
```

### Add Feature Engineering

```python
from sklearn.preprocessing import PolynomialFeatures

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
```

### Change Metrics

```python
# In GridSearchCV, use F1 instead of ROC-AUC
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',  # Changed from 'roc_auc'
    refit=True,
    n_jobs=-1
)
```

---

## 🐛 Troubleshooting

### MLflow UI Not Starting
```bash
# Check if port 5000 is in use
lsof -i :5000

# Use a different port
mlflow ui --port 5001
```

### Import Errors
```bash
# Ensure all dependencies are installed
pip install --upgrade -r requirements.txt

# Check versions
python -c "import sklearn; print(sklearn.__version__)"
python -c "import mlflow; print(mlflow.__version__)"
```

### Slow Performance
```python
# Reduce dataset size
X_train, X_test, y_train, y_test = load_and_split_data(
    n_samples=1000,  # Reduced from 5000
    n_features=10    # Reduced from 20
)

# Reduce grid search space
param_grid = {
    'classifier__C': [0.1, 1.0],  # Reduced from 4 values
    'classifier__solver': ['lbfgs']  # Reduced from 2 values
}

# Reduce CV folds
grid_search = GridSearchCV(..., cv=3)  # Reduced from 5
```

---

## 📚 Further Reading

**Scikit-learn Documentation:**
- [Pipelines](https://scikit-learn.org/stable/modules/compose.html#pipeline)
- [GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html)
- [Model Calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Learning Curves](https://scikit-learn.org/stable/modules/learning_curve.html)

**MLflow Documentation:**
- [Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Models](https://mlflow.org/docs/latest/models.html)
- [Projects](https://mlflow.org/docs/latest/projects.html)

**Machine Learning Best Practices:**
- [Google's Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Scikit-learn's Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)

---

## 📝 License

This code is provided as a learning resource and can be freely used and modified.

## 🤝 Contributing

Feel free to:
- Add more preprocessing steps
- Try different algorithms
- Add more evaluation metrics
- Improve visualization

---

**Happy Learning! 🚀**
