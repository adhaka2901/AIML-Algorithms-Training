"""
Complete ML Workflow with MLflow Tracking
==========================================

This script demonstrates a full machine learning workflow from data loading
through model evaluation, with MLflow experiment tracking throughout.

Big-picture structure:
1. Data & split
2. Preprocessing & base pipeline
3. Hyperparameter tuning with cross-validation
4. Fit final calibrated model
5. Evaluate on test set (all classification metrics & plots)
6. Learning curve
7. Model complexity (validation) curve
8. Runtime comparison table

MLflow tracks:
- Parameters (hyperparameters, preprocessing choices)
- Metrics (ROC-AUC, PR-AUC, F1, etc.)
- Artifacts (plots, models, data splits)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# Scikit-learn imports
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score,
    learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)
from sklearn.calibration import calibration_curve

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# =============================================================================
# STEP 1: DATA & SPLIT
# =============================================================================
def load_and_split_data(n_samples=5000, n_features=20, n_informative=15,
                        random_state=42, test_size=0.2):
    """
    Load/generate data and perform train-test split.
    
    Why generate synthetic data?
    - Fully controlled: we know the true relationship
    - Reproducible: same data every time
    - Customizable: adjust difficulty, imbalance, etc.
    
    Why hold out test set now?
    - Test set must NEVER be seen during training or tuning
    - Acts as final "real-world" evaluation
    - All CV, tuning, and preprocessing fit only on training data
    """
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING & TRAIN-TEST SPLIT")
    print("="*80)
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,
        n_classes=2,
        flip_y=0.05,  # 5% label noise
        class_sep=0.8,  # moderate separation
        random_state=random_state
    )
    
    # Add some missing values to make preprocessing more realistic
    # In real data, missing values are common
    rng = np.random.RandomState(random_state)
    mask = rng.rand(*X.shape) < 0.05  # 5% missing
    X = X.astype(float)
    X[mask] = np.nan
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Class balance: {y.mean():.2%} positive class")
    print(f"Missing values: {np.isnan(X).sum()} ({np.isnan(X).mean():.2%})")
    
    # Train-test split
    # WHY: Test set is completely held out for final evaluation
    # It simulates "future unseen data"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain size: {X_train.shape[0]} samples")
    print(f"Test size: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 2: PREPROCESSING & BASE PIPELINE
# =============================================================================
def create_preprocessing_pipeline():
    """
    Create preprocessing steps and wrap with classifier in Pipeline.
    
    Why Pipeline?
    - Prevents data leakage: fit/transform operations stay separate
    - Reproducible: same preprocessing for train, CV, and test
    - Clean: all steps in one object
    - Hyperparameter tuning: can tune preprocessing params too
    
    Why this preprocessing?
    - SimpleImputer: handle missing values (common in real data)
    - StandardScaler: zero mean, unit variance (helps LogReg convergence)
    
    Why LogisticRegression?
    - Good baseline for binary classification
    - Fast to train
    - Interpretable coefficients
    - Outputs calibrated probabilities (when trained properly)
    """
    print("\n" + "="*80)
    print("STEP 2: PREPROCESSING & BASE PIPELINE")
    print("="*80)
    
    pipeline = Pipeline([
        # Step 1: Impute missing values with median
        # WHY median? Robust to outliers
        ('imputer', SimpleImputer(strategy='median')),
        
        # Step 2: Standardize features
        # WHY? LogReg uses gradient descent, which converges faster with scaled features
        # Also, regularization (L2 penalty) treats all features equally when scaled
        ('scaler', StandardScaler()),
        
        # Step 3: Classifier
        # WHY LogReg? Good baseline, fast, interpretable
        # max_iter=1000 to ensure convergence
        # random_state for reproducibility
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    print("Pipeline steps:")
    for name, step in pipeline.steps:
        print(f"  - {name}: {step.__class__.__name__}")
    
    return pipeline


# =============================================================================
# STEP 3: HYPERPARAMETER TUNING WITH CROSS-VALIDATION
# =============================================================================
def tune_hyperparameters(pipeline, X_train, y_train, cv=5):
    """
    Use GridSearchCV to find best hyperparameters.
    
    Why GridSearchCV?
    - Systematic: tries all combinations
    - Unbiased: uses CV to evaluate each combination
    - Automatic: selects best and retrains on full training data
    
    Why cross-validation?
    - More robust estimate than single train-val split
    - Uses all training data for both training and validation
    - Reduces variance in performance estimates
    
    Why ROC-AUC for scoring?
    - Threshold-independent metric
    - Good for imbalanced classes
    - Easy to interpret (0.5 = random, 1.0 = perfect)
    
    What hyperparameters to tune?
    - C: inverse regularization strength (smaller = more regularization)
    - solver: optimization algorithm
    - class_weight: handle class imbalance
    """
    print("\n" + "="*80)
    print("STEP 3: HYPERPARAMETER TUNING WITH CROSS-VALIDATION")
    print("="*80)
    
    # Define hyperparameter grid
    param_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],  # Regularization strength
        'classifier__solver': ['lbfgs', 'liblinear'],  # Optimization algorithm
        'classifier__class_weight': [None, 'balanced']  # Handle imbalance
    }
    
    print(f"Parameter grid: {len(param_grid['classifier__C']) * len(param_grid['classifier__solver']) * len(param_grid['classifier__class_weight'])} combinations")
    print(f"Cross-validation folds: {cv}")
    print(f"Total models to train: {cv * len(param_grid['classifier__C']) * len(param_grid['classifier__solver']) * len(param_grid['classifier__class_weight'])}")
    
    # GridSearchCV
    # WHY refit=True? Automatically retrain best model on full training data
    # WHY return_train_score=True? See if we're overfitting
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='roc_auc',  # Metric to optimize
        refit=True,  # Retrain best model on full training data
        return_train_score=True,  # Track training scores too
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    
    print("\nStarting grid search...")
    start_time = time()
    grid_search.fit(X_train, y_train)
    elapsed_time = time() - start_time
    
    print(f"\nGrid search completed in {elapsed_time:.2f} seconds")
    print(f"Best ROC-AUC (CV): {grid_search.best_score_:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Analyze overfitting
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results['overfit_gap'] = cv_results['mean_train_score'] - cv_results['mean_test_score']
    
    print(f"\nOverfitting analysis:")
    print(f"  Average train-val gap: {cv_results['overfit_gap'].mean():.4f}")
    print(f"  Max train-val gap: {cv_results['overfit_gap'].max():.4f}")
    
    return grid_search, cv_results


# =============================================================================
# STEP 4: FIT FINAL CALIBRATED MODEL
# =============================================================================
def fit_calibrated_model(best_estimator, X_train, y_train, cv=5):
    """
    Wrap best model in CalibratedClassifierCV for better probability estimates.
    
    Why calibration?
    - Many classifiers output poorly calibrated probabilities
    - Calibration fixes this using isotonic regression or Platt scaling
    - Important when probability values matter (not just rankings)
    
    Why fit on full training data?
    - GridSearchCV already found best hyperparameters
    - Now we want to use ALL training data for final model
    - Test set is still held out
    
    How does CalibratedClassifierCV work?
    - Takes a fitted or unfitted estimator
    - Uses CV to train calibration on held-out folds
    - Final model: base estimator + calibration mapping
    """
    print("\n" + "="*80)
    print("STEP 4: FIT FINAL CALIBRATED MODEL")
    print("="*80)
    
    print("Wrapping best estimator in CalibratedClassifierCV...")
    print(f"Calibration method: sigmoid (Platt scaling)")
    print(f"Calibration CV folds: {cv}")
    
    # CalibratedClassifierCV
    # WHY sigmoid? Good for LogReg, fast
    # WHY cv=5? Same as tuning CV for consistency
    calibrated_model = CalibratedClassifierCV(
        best_estimator,
        method='sigmoid',  # Platt scaling
        cv=cv,
        n_jobs=-1
    )
    
    start_time = time()
    calibrated_model.fit(X_train, y_train)
    elapsed_time = time() - start_time
    
    print(f"Calibrated model fitted in {elapsed_time:.2f} seconds")
    print(f"Number of calibrated classifiers: {len(calibrated_model.calibrated_classifiers_)}")
    
    return calibrated_model


# =============================================================================
# STEP 5: EVALUATE ON TEST SET (ALL CLASSIFICATION METRICS & PLOTS)
# =============================================================================
def evaluate_on_test_set(model, X_test, y_test, output_dir='outputs'):
    """
    Comprehensive evaluation on held-out test set.
    
    Why test set evaluation?
    - Final, unbiased estimate of model performance
    - Simulates real-world deployment
    - Test set was never seen during training/tuning
    
    What metrics?
    - ROC-AUC: Overall ranking quality
    - PR-AUC: Precision-recall tradeoff (better for imbalanced data)
    - F1: Harmonic mean of precision and recall
    - Confusion matrix: Detailed error breakdown
    
    What plots?
    - ROC curve: TPR vs FPR at all thresholds
    - PR curve: Precision vs Recall at all thresholds
    - Confusion matrix: Visual error breakdown
    - Calibration curve: Are probabilities well-calibrated?
    """
    print("\n" + "="*80)
    print("STEP 5: EVALUATE ON TEST SET")
    print("="*80)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class
    y_pred = model.predict(X_test)  # Hard predictions (default threshold=0.5)
    
    # Compute metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    # Prevalence baseline for PR-AUC
    # WHY? PR-AUC should be compared to prevalence (random baseline)
    prevalence = y_test.mean()
    
    print(f"\nTest Set Metrics:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f} (prevalence baseline: {prevalence:.4f})")
    print(f"  F1 Score: {f1:.4f} (at threshold=0.5)")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    axes[0, 1].plot(recall, precision, linewidth=2, label=f'PR (AUC={pr_auc:.4f})')
    axes[0, 1].axhline(y=prevalence, color='k', linestyle='--', linewidth=1,
                       label=f'Prevalence={prevalence:.4f}')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix
    # Find threshold that maximizes F1
    f1_scores = []
    thresholds_test = np.linspace(0.1, 0.9, 50)
    for thresh in thresholds_test:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        f1_scores.append(f1_score(y_test, y_pred_thresh))
    
    optimal_threshold = thresholds_test[np.argmax(f1_scores)]
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred_optimal)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(ax=axes[1, 0], cmap='Blues', values_format='d')
    axes[1, 0].set_title(f'Confusion Matrix (threshold={optimal_threshold:.2f}, max F1)')
    
    # Plot 4: Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy='uniform')
    axes[1, 1].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Calibrated Model')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
    axes[1, 1].set_xlabel('Mean Predicted Probability')
    axes[1, 1].set_ylabel('Fraction of Positives')
    axes[1, 1].set_title('Calibration Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/test_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to: {output_dir}/test_evaluation.png")
    plt.close()
    
    # Return metrics for MLflow
    return {
        'test_roc_auc': roc_auc,
        'test_pr_auc': pr_auc,
        'test_f1': f1,
        'test_prevalence': prevalence,
        'optimal_threshold': optimal_threshold
    }


# =============================================================================
# STEP 6: LEARNING CURVE
# =============================================================================
def plot_learning_curve(pipeline, X_train, y_train, cv=5, output_dir='outputs'):
    """
    Plot learning curves: training and validation scores vs training size.
    
    Why learning curves?
    - Diagnose bias vs variance issues
    - See if more data would help
    - Understand training dynamics
    
    How to interpret?
    - High bias: both curves plateau at low performance (underfitting)
    - High variance: large gap between train and val curves (overfitting)
    - Good fit: curves converge to high performance
    - More data helps: curves haven't plateaued yet
    """
    print("\n" + "="*80)
    print("STEP 6: LEARNING CURVE")
    print("="*80)
    
    # Use exponentially spaced training sizes for better visualization
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    print("Computing learning curves...")
    print(f"Training sizes: {len(train_sizes)} points")
    print(f"CV folds: {cv}")
    
    # learning_curve computes train/val scores for different training sizes
    # WHY? To see how performance changes with more data
    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipeline,
        X_train,
        y_train,
        train_sizes=train_sizes,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )
    
    # Compute mean and std
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', linewidth=2, label='Training score')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(train_sizes_abs, val_mean, 'o-', linewidth=2, label='Validation score (CV)')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.xlabel('Training Set Size')
    plt.ylabel('ROC-AUC Score')
    plt.title('Learning Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Add interpretation text
    final_gap = train_mean[-1] - val_mean[-1]
    interpretation = "Overfitting (high variance)" if final_gap > 0.05 else "Good fit"
    plt.text(0.02, 0.98, f"Train-Val Gap: {final_gap:.4f}\nInterpretation: {interpretation}",
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curve.png', dpi=150, bbox_inches='tight')
    print(f"Learning curve saved to: {output_dir}/learning_curve.png")
    plt.close()
    
    return {
        'final_train_score': train_mean[-1],
        'final_val_score': val_mean[-1],
        'train_val_gap': final_gap
    }


# =============================================================================
# STEP 7: MODEL COMPLEXITY (VALIDATION) CURVE
# =============================================================================
def plot_validation_curve(pipeline, X_train, y_train, cv=5, output_dir='outputs'):
    """
    Plot validation curves: training and validation scores vs hyperparameter.
    
    Why validation curves?
    - See how model complexity affects performance
    - Identify optimal hyperparameter value
    - Diagnose underfitting vs overfitting
    
    What hyperparameter?
    - For LogReg: C (inverse regularization strength)
    - Smaller C = more regularization = simpler model
    - Larger C = less regularization = more complex model
    
    How to interpret?
    - Sweet spot: where val score is highest
    - Too simple (low C): both curves low (underfitting)
    - Too complex (high C): train high, val low (overfitting)
    """
    print("\n" + "="*80)
    print("STEP 7: MODEL COMPLEXITY (VALIDATION) CURVE")
    print("="*80)
    
    # Range of C values to test
    param_range = np.logspace(-3, 3, 10)  # 0.001 to 1000
    param_name = 'classifier__C'
    
    print(f"Hyperparameter: {param_name}")
    print(f"Value range: {param_range[0]:.4f} to {param_range[-1]:.4f}")
    print(f"Number of values: {len(param_range)}")
    print(f"CV folds: {cv}")
    
    # validation_curve computes train/val scores for different param values
    # WHY? To see optimal complexity level
    train_scores, val_scores = validation_curve(
        pipeline,
        X_train,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    # Compute mean and std
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Find optimal C
    optimal_idx = np.argmax(val_mean)
    optimal_C = param_range[optimal_idx]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(param_range, train_mean, 'o-', linewidth=2, label='Training score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.semilogx(param_range, val_mean, 'o-', linewidth=2, label='Validation score (CV)')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.axvline(x=optimal_C, color='red', linestyle='--', linewidth=1, label=f'Optimal C={optimal_C:.3f}')
    plt.xlabel('C (Inverse Regularization Strength)')
    plt.ylabel('ROC-AUC Score')
    plt.title('Validation Curve (Model Complexity)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Add interpretation text
    plt.text(0.02, 0.98, f"Optimal C: {optimal_C:.4f}\nBest Val Score: {val_mean[optimal_idx]:.4f}",
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/validation_curve.png', dpi=150, bbox_inches='tight')
    print(f"Validation curve saved to: {output_dir}/validation_curve.png")
    plt.close()
    
    return {
        'optimal_C': optimal_C,
        'optimal_val_score': val_mean[optimal_idx]
    }


# =============================================================================
# STEP 8: RUNTIME COMPARISON TABLE
# =============================================================================
def compare_model_runtimes(X_train, y_train, X_test, y_test, output_dir='outputs'):
    """
    Compare training and prediction times for multiple models.
    
    Why runtime analysis?
    - Different models have different speed/accuracy tradeoffs
    - Important for production deployment
    - Helps choose model based on latency requirements
    
    What to measure?
    - Fit time: time to train model
    - Predict time: time to make predictions on test set
    - Per-sample predict time: useful for real-time systems
    """
    print("\n" + "="*80)
    print("STEP 8: RUNTIME COMPARISON TABLE")
    print("="*80)
    
    # Define models to compare
    models = {
        'Logistic Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTiming {name}...")
        
        # Measure fit time
        start = time()
        model.fit(X_train, y_train)
        fit_time = time() - start
        
        # Measure predict time
        start = time()
        y_pred = model.predict(X_test)
        predict_time = time() - start
        
        # Per-sample predict time
        per_sample_time = predict_time / len(X_test) * 1000  # Convert to milliseconds
        
        # Compute accuracy
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        results.append({
            'Model': name,
            'Fit Time (s)': fit_time,
            'Predict Time (s)': predict_time,
            'Per-Sample Time (ms)': per_sample_time,
            'ROC-AUC': roc_auc
        })
        
        print(f"  Fit time: {fit_time:.4f}s")
        print(f"  Predict time: {predict_time:.4f}s")
        print(f"  Per-sample time: {per_sample_time:.4f}ms")
        print(f"  ROC-AUC: {roc_auc:.4f}")
    
    # Create DataFrame
    runtime_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("RUNTIME COMPARISON TABLE")
    print("="*60)
    print(runtime_df.to_string(index=False))
    print("="*60)
    
    # Save table
    runtime_df.to_csv(f'{output_dir}/runtime_comparison.csv', index=False)
    print(f"\nRuntime table saved to: {output_dir}/runtime_comparison.csv")
    
    # Hardware note
    print("\nNote: Runtime measurements depend on hardware and system load.")
    print("These are relative comparisons, not absolute benchmarks.")
    
    return runtime_df


# =============================================================================
# MLFLOW EXPERIMENT TRACKING
# =============================================================================
def run_experiment_with_mlflow(experiment_name='ml_workflow_demo'):
    """
    Run the complete ML workflow with MLflow tracking.
    
    Why MLflow?
    - Tracks all experiments in one place
    - Logs parameters, metrics, and artifacts
    - Enables experiment comparison
    - Reproducible: can reload models and data
    
    What does MLflow track?
    - Parameters: hyperparameters, preprocessing choices
    - Metrics: ROC-AUC, PR-AUC, F1, etc.
    - Artifacts: plots, models, data splits, tables
    - Code version, timestamps, etc.
    """
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name='complete_workflow'):
        
        print("\n" + "="*80)
        print("STARTING ML WORKFLOW WITH MLFLOW TRACKING")
        print(f"Experiment: {experiment_name}")
        print("="*80)
        
        # Create output directory
        import os
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        # =====================================================================
        # STEP 1: Load and split data
        # =====================================================================
        X_train, X_test, y_train, y_test = load_and_split_data(
            n_samples=5000,
            n_features=20,
            n_informative=15,
            random_state=42,
            test_size=0.2
        )
        
        # Log data parameters
        mlflow.log_param('n_samples', 5000)
        mlflow.log_param('n_features', 20)
        mlflow.log_param('n_informative', 15)
        mlflow.log_param('test_size', 0.2)
        mlflow.log_param('train_samples', len(X_train))
        mlflow.log_param('test_samples', len(X_test))
        
        # =====================================================================
        # STEP 2: Create preprocessing pipeline
        # =====================================================================
        pipeline = create_preprocessing_pipeline()
        
        # Log preprocessing parameters
        mlflow.log_param('imputation_strategy', 'median')
        mlflow.log_param('scaling', 'StandardScaler')
        mlflow.log_param('classifier', 'LogisticRegression')
        
        # =====================================================================
        # STEP 3: Hyperparameter tuning
        # =====================================================================
        grid_search, cv_results = tune_hyperparameters(
            pipeline, X_train, y_train, cv=5
        )
        
        # Log tuning parameters
        mlflow.log_param('cv_folds', 5)
        mlflow.log_param('tuning_method', 'GridSearchCV')
        mlflow.log_param('tuning_metric', 'roc_auc')
        
        # Log best parameters
        for param_name, param_value in grid_search.best_params_.items():
            mlflow.log_param(f'best_{param_name}', param_value)
        
        # Log best CV score
        mlflow.log_metric('best_cv_roc_auc', grid_search.best_score_)
        
        # Save CV results
        cv_results.to_csv(f'{output_dir}/cv_results.csv', index=False)
        mlflow.log_artifact(f'{output_dir}/cv_results.csv')
        
        # =====================================================================
        # STEP 4: Fit final calibrated model
        # =====================================================================
        calibrated_model = fit_calibrated_model(
            grid_search.best_estimator_, X_train, y_train, cv=5
        )
        
        # Log calibration parameters
        mlflow.log_param('calibration_method', 'sigmoid')
        mlflow.log_param('calibration_cv_folds', 5)
        
        # =====================================================================
        # STEP 5: Evaluate on test set
        # =====================================================================
        test_metrics = evaluate_on_test_set(
            calibrated_model, X_test, y_test, output_dir
        )
        
        # Log test metrics
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log test evaluation plots
        mlflow.log_artifact(f'{output_dir}/test_evaluation.png')
        
        # =====================================================================
        # STEP 6: Learning curve
        # =====================================================================
        learning_metrics = plot_learning_curve(
            pipeline, X_train, y_train, cv=5, output_dir=output_dir
        )
        
        # Log learning curve metrics
        for metric_name, metric_value in learning_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log learning curve plot
        mlflow.log_artifact(f'{output_dir}/learning_curve.png')
        
        # =====================================================================
        # STEP 7: Validation curve
        # =====================================================================
        validation_metrics = plot_validation_curve(
            pipeline, X_train, y_train, cv=5, output_dir=output_dir
        )
        
        # Log validation curve metrics
        for metric_name, metric_value in validation_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log validation curve plot
        mlflow.log_artifact(f'{output_dir}/validation_curve.png')
        
        # =====================================================================
        # STEP 8: Runtime comparison
        # =====================================================================
        runtime_df = compare_model_runtimes(
            X_train, y_train, X_test, y_test, output_dir=output_dir
        )
        
        # Log runtime table
        mlflow.log_artifact(f'{output_dir}/runtime_comparison.csv')
        
        # =====================================================================
        # Save final model
        # =====================================================================
        print("\n" + "="*80)
        print("SAVING FINAL MODEL TO MLFLOW")
        print("="*80)
        
        # Save model
        mlflow.sklearn.log_model(
            calibrated_model,
            artifact_path='model',
            registered_model_name='calibrated_logistic_regression'
        )
        
        print("Model saved to MLflow")
        
        # =====================================================================
        # Final summary
        # =====================================================================
        print("\n" + "="*80)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nFinal Test Performance:")
        print(f"  ROC-AUC: {test_metrics['test_roc_auc']:.4f}")
        print(f"  PR-AUC: {test_metrics['test_pr_auc']:.4f}")
        print(f"  F1 Score: {test_metrics['test_f1']:.4f}")
        print(f"  Optimal Threshold: {test_metrics['optimal_threshold']:.4f}")
        
        print(f"\nAll artifacts saved to: {output_dir}/")
        print(f"MLflow experiment: {experiment_name}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        
        print("\n" + "="*80)
        print("To view MLflow UI, run:")
        print("  mlflow ui")
        print("Then open: http://localhost:5000")
        print("="*80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    """
    Main entry point for the ML workflow.
    
    This executes all 8 steps and tracks everything with MLflow.
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Run the complete workflow
    run_experiment_with_mlflow(experiment_name='ml_workflow_demo')
    
    print("\n✅ Complete ML workflow finished successfully!")
    print("Check the 'outputs/' directory for all plots and tables.")
    print("Run 'mlflow ui' to view tracked experiments.")
