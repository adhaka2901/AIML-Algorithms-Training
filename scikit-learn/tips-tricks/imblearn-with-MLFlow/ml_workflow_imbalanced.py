"""
Complete ML Workflow with Imbalanced-Learn Integration
========================================================

This script demonstrates handling imbalanced datasets using the imbalanced-learn library.

KEY CONCEPTS:
1. Imbalanced data: When one class is significantly less frequent than others
2. Sampling strategies: Over-sampling, under-sampling, and hybrid approaches
3. Pipeline integration: Why we need imblearn.pipeline, not sklearn.pipeline
4. Evaluation: Metrics that aren't fooled by class imbalance

CRITICAL PIPELINE RULE:
- Use imblearn.pipeline.Pipeline, NOT sklearn.pipeline.Pipeline
- Why? Sampling must happen INSIDE each CV fold to avoid data leakage
- sklearn's Pipeline doesn't know about samplers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# Scikit-learn imports
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay, classification_report, balanced_accuracy_score
)

# CRITICAL: Use imblearn.pipeline, NOT sklearn.pipeline
# WHY? sklearn's Pipeline doesn't support samplers
from imblearn.pipeline import Pipeline  # NOTE: imblearn, not sklearn!

# Imbalanced-learn sampling techniques
from imblearn.over_sampling import (
    SMOTE,           # Synthetic Minority Over-sampling Technique
    ADASYN,          # Adaptive Synthetic Sampling
    RandomOverSampler,  # Simple duplication
    SVMSMOTE         # SVM-based SMOTE
)
from imblearn.under_sampling import (
    RandomUnderSampler,   # Random deletion
    TomekLinks,           # Remove borderline cases
    EditedNearestNeighbours,  # Clean based on neighbors
    NearMiss              # Various undersampling strategies
)
from imblearn.combine import (
    SMOTEENN,        # SMOTE + Edited Nearest Neighbors
    SMOTETomek        # SMOTE + Tomek Links
)

# MLflow
import mlflow
import mlflow.sklearn

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# =============================================================================
# UNDERSTANDING IMBALANCED DATA
# =============================================================================
"""
WHY IS CLASS IMBALANCE A PROBLEM?

1. Majority Class Bias:
   - Model learns to predict majority class to maximize accuracy
   - Example: 95% class 0, 5% class 1
   - Model that always predicts 0: 95% accuracy! (but useless)

2. Gradient Bias:
   - In gradient descent, majority class dominates the loss
   - Minority class errors have less impact on weight updates
   - Model ignores minority class patterns

3. Decision Boundary Issues:
   - Decision boundary shifts toward minority class
   - Minority class region becomes smaller
   - Harder to learn minority class patterns

WHEN IS DATA CONSIDERED IMBALANCED?

- Mild: 60/40 to 70/30 ratio
- Moderate: 80/20 to 90/10 ratio  
- Severe: 95/5 to 99/1 ratio

SOLUTIONS:

1. Cost-Sensitive Learning:
   - class_weight='balanced' in sklearn models
   - Penalizes majority class errors more
   - No change to data distribution

2. Sampling Techniques (imbalanced-learn):
   - Over-sampling: Increase minority class samples
   - Under-sampling: Decrease majority class samples
   - Hybrid: Combination of both
"""


# =============================================================================
# STEP 1: CREATE IMBALANCED DATASET
# =============================================================================
def create_imbalanced_data(imbalance_ratio=0.05, n_samples=5000, random_state=42):
    """
    Create imbalanced dataset for demonstration.
    
    Parameters:
    -----------
    imbalance_ratio : float
        Proportion of minority class (default: 0.05 = 5%)
        
    Why create synthetic imbalanced data?
    - Controlled: We know the true imbalance ratio
    - Reproducible: Same data every time
    - Adjustable: Easy to test different imbalance levels
    """
    print("\n" + "="*80)
    print("STEP 1: CREATE IMBALANCED DATASET")
    print("="*80)
    
    # Calculate weights for make_classification
    # weights = [majority_weight, minority_weight]
    minority_weight = imbalance_ratio
    majority_weight = 1 - imbalance_ratio
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=2,
        weights=[majority_weight, minority_weight],  # Create imbalance
        flip_y=0.02,  # Small label noise
        class_sep=0.8,
        random_state=random_state
    )
    
    # Add missing values
    rng = np.random.RandomState(random_state)
    mask = rng.rand(*X.shape) < 0.05
    X = X.astype(float)
    X[mask] = np.nan
    
    # Print class distribution
    class_counts = np.bincount(y)
    class_ratio = class_counts[1] / len(y)
    
    print(f"Total samples: {len(y)}")
    print(f"Class 0 (majority): {class_counts[0]} ({class_counts[0]/len(y):.2%})")
    print(f"Class 1 (minority): {class_counts[1]} ({class_counts[1]/len(y):.2%})")
    print(f"Imbalance ratio: {class_counts[1]/class_counts[0]:.4f} (1:{class_counts[0]/class_counts[1]:.1f})")
    
    # Train-test split (stratified to preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set:")
    train_counts = np.bincount(y_train)
    print(f"  Class 0: {train_counts[0]} ({train_counts[0]/len(y_train):.2%})")
    print(f"  Class 1: {train_counts[1]} ({train_counts[1]/len(y_train):.2%})")
    
    print(f"\nTest set:")
    test_counts = np.bincount(y_test)
    print(f"  Class 0: {test_counts[0]} ({test_counts[0]/len(y_test):.2%})")
    print(f"  Class 1: {test_counts[1]} ({test_counts[1]/len(y_test):.2%})")
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# SAMPLING STRATEGIES EXPLAINED
# =============================================================================
"""
OVER-SAMPLING TECHNIQUES (Increase minority class):

1. RandomOverSampler:
   - Strategy: Duplicate minority class samples randomly
   - Pros: Simple, fast, preserves all information
   - Cons: Risk of overfitting (exact duplicates)
   - When to use: Baseline, or when data is very limited

2. SMOTE (Synthetic Minority Over-sampling Technique):
   - Strategy: Create synthetic samples by interpolating between neighbors
   - Process:
     a. Pick minority sample
     b. Find k nearest minority neighbors
     c. Create synthetic sample along line between them
   - Pros: No exact duplicates, smooths decision boundary
   - Cons: Can create noisy samples in overlap regions
   - When to use: Most common choice, good default

3. ADASYN (Adaptive Synthetic Sampling):
   - Strategy: Like SMOTE, but focuses on hard-to-learn regions
   - Process: Generate more synthetics where minority class is harder to learn
   - Pros: Adapts to data density, focuses on decision boundary
   - Cons: More complex, can amplify noise
   - When to use: When decision boundary is complex

4. SVMSMOTE:
   - Strategy: SMOTE + SVM to identify support vectors
   - Process: Generate synthetics only near support vectors (boundary)
   - Pros: Focuses on decision boundary, reduces noise
   - Cons: Slower due to SVM computation
   - When to use: When clean decision boundary is critical


UNDER-SAMPLING TECHNIQUES (Decrease majority class):

1. RandomUnderSampler:
   - Strategy: Randomly remove majority class samples
   - Pros: Fast, reduces training time
   - Cons: Loses information, may discard important samples
   - When to use: When majority class has redundant information

2. TomekLinks:
   - Strategy: Remove pairs where classes are nearest neighbors
   - Process: Remove borderline/noisy majority samples
   - Pros: Cleans decision boundary, removes ambiguous cases
   - Cons: May not balance classes fully
   - When to use: Combined with over-sampling (see SMOTE-Tomek)

3. EditedNearestNeighbours:
   - Strategy: Remove samples misclassified by k-NN
   - Process: Remove majority samples whose neighbors are mostly minority
   - Pros: Removes noisy samples, cleans data
   - Cons: May remove too many samples
   - When to use: When data quality is poor

4. NearMiss:
   - Strategy: Select majority samples closest to minority samples
   - Pros: Keeps informative majority samples
   - Cons: May create too clean separation
   - When to use: When majority class has clear clusters


HYBRID TECHNIQUES (Combine over-sampling and under-sampling):

1. SMOTEENN:
   - Strategy: SMOTE + Edited Nearest Neighbours
   - Process: Over-sample with SMOTE, then clean with ENN
   - Pros: Balances classes and removes noise
   - Cons: May be too aggressive in cleaning
   - When to use: Noisy data with severe imbalance

2. SMOTETomek:
   - Strategy: SMOTE + Tomek Links
   - Process: Over-sample with SMOTE, then remove Tomek links
   - Pros: Balances classes and cleans boundary
   - Cons: More complex, slower
   - When to use: Most robust choice for severe imbalance
"""


# =============================================================================
# STEP 2: CREATE PIPELINES WITH DIFFERENT SAMPLING STRATEGIES
# =============================================================================
def create_sampling_pipelines():
    """
    Create multiple pipelines with different sampling strategies.
    
    CRITICAL: We use imblearn.pipeline.Pipeline, NOT sklearn.pipeline.Pipeline
    
    WHY?
    - sklearn's Pipeline doesn't know about samplers
    - When we do cross-validation, sampling must happen INSIDE each fold
    - If we sample before CV, we have data leakage (test fold influenced by train fold)
    - imblearn's Pipeline handles this correctly
    
    HOW IT WORKS:
    1. CV splits data into train and validation folds
    2. INSIDE each fold:
       a. Fit imputer on train fold
       b. Transform both train and val folds with imputer
       c. Fit scaler on train fold
       d. Transform both train and val folds with scaler
       e. FIT SAMPLER on train fold (creates balanced training set)
       f. Train classifier on RESAMPLED train fold
       g. Evaluate on ORIGINAL val fold (not resampled!)
    3. Test set is NEVER resampled
    """
    print("\n" + "="*80)
    print("STEP 2: CREATE PIPELINES WITH SAMPLING STRATEGIES")
    print("="*80)
    
    pipelines = {}
    
    # Baseline: No sampling, just class_weight='balanced'
    pipelines['Baseline (class_weight)'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight='balanced',  # Cost-sensitive approach
            max_iter=1000,
            random_state=42
        ))
    ])
    
    # Random Over-sampling
    pipelines['RandomOverSampler'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('sampler', RandomOverSampler(random_state=42)),  # Duplicate minority
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # SMOTE
    pipelines['SMOTE'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('sampler', SMOTE(random_state=42)),  # Synthetic over-sampling
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # ADASYN
    pipelines['ADASYN'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('sampler', ADASYN(random_state=42)),  # Adaptive synthetic
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Random Under-sampling
    pipelines['RandomUnderSampler'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('sampler', RandomUnderSampler(random_state=42)),  # Remove majority
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # SMOTE + Tomek Links (Hybrid)
    pipelines['SMOTETomek'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('sampler', SMOTETomek(random_state=42)),  # Over-sample + clean
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # SMOTE + ENN (Hybrid)
    pipelines['SMOTEENN'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('sampler', SMOTEENN(random_state=42)),  # Over-sample + remove noise
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    print(f"Created {len(pipelines)} pipelines with different sampling strategies:")
    for name in pipelines.keys():
        print(f"  - {name}")
    
    return pipelines


# =============================================================================
# STEP 3: EVALUATE ALL STRATEGIES WITH CROSS-VALIDATION
# =============================================================================
def compare_sampling_strategies(pipelines, X_train, y_train, cv=5):
    """
    Compare all sampling strategies using cross-validation.
    
    IMPORTANT METRICS FOR IMBALANCED DATA:
    
    1. ROC-AUC: Threshold-independent, good for ranking
    2. PR-AUC: Better than ROC-AUC for imbalanced data
    3. F1: Harmonic mean of precision and recall
    4. Balanced Accuracy: Average of recall for each class
    
    WHY NOT ACCURACY?
    - Misleading for imbalanced data
    - Example: 95% class 0, 5% class 1
    - Always predicting class 0: 95% accuracy (but useless!)
    """
    print("\n" + "="*80)
    print("STEP 3: COMPARE SAMPLING STRATEGIES (CROSS-VALIDATION)")
    print("="*80)
    
    results = []
    
    # Metrics to evaluate
    scoring_metrics = {
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision',
        'f1': 'f1',
        'balanced_accuracy': 'balanced_accuracy'
    }
    
    for name, pipeline in pipelines.items():
        print(f"\nEvaluating: {name}")
        
        metric_scores = {}
        
        for metric_name, scorer in scoring_metrics.items():
            # Cross-validation score
            # IMPORTANT: Sampling happens INSIDE each fold
            scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=cv,
                scoring=scorer,
                n_jobs=-1
            )
            
            mean_score = scores.mean()
            std_score = scores.std()
            metric_scores[metric_name] = mean_score
            metric_scores[f'{metric_name}_std'] = std_score
            
            print(f"  {metric_name}: {mean_score:.4f} (+/- {std_score:.4f})")
        
        results.append({
            'Strategy': name,
            **metric_scores
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by PR-AUC (best metric for imbalanced data)
    results_df = results_df.sort_values('pr_auc', ascending=False)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY (sorted by PR-AUC)")
    print("="*80)
    print(results_df[['Strategy', 'roc_auc', 'pr_auc', 'f1', 'balanced_accuracy']].to_string(index=False))
    
    return results_df


# =============================================================================
# STEP 4: VISUALIZE STRATEGY COMPARISON
# =============================================================================
def visualize_strategy_comparison(results_df, output_dir='outputs'):
    """
    Visualize performance of different sampling strategies.
    """
    print("\n" + "="*80)
    print("STEP 4: VISUALIZE STRATEGY COMPARISON")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['roc_auc', 'pr_auc', 'f1', 'balanced_accuracy']
    titles = ['ROC-AUC', 'PR-AUC (Best for Imbalance)', 'F1 Score', 'Balanced Accuracy']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        # Sort by current metric
        data = results_df.sort_values(metric, ascending=True)
        
        # Bar plot
        bars = ax.barh(data['Strategy'], data[metric])
        
        # Color code: green if > baseline, red if < baseline
        baseline_score = data[data['Strategy'] == 'Baseline (class_weight)'][metric].values[0]
        colors = ['green' if score > baseline_score else 'red' for score in data[metric]]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        # Add error bars if std available
        if f'{metric}_std' in data.columns:
            ax.errorbar(
                data[metric], 
                range(len(data)),
                xerr=data[f'{metric}_std'],
                fmt='none',
                ecolor='black',
                capsize=5,
                alpha=0.5
            )
        
        # Formatting
        ax.set_xlabel(title)
        ax.set_title(f'{title} by Sampling Strategy')
        ax.axvline(x=baseline_score, color='blue', linestyle='--', 
                   linewidth=2, label='Baseline', alpha=0.7)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sampling_strategy_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_dir}/sampling_strategy_comparison.png")
    plt.close()


# =============================================================================
# STEP 5: DETAILED EVALUATION OF BEST STRATEGY
# =============================================================================
def evaluate_best_strategy(best_pipeline, X_train, y_train, X_test, y_test, 
                          strategy_name, output_dir='outputs'):
    """
    Detailed evaluation of the best sampling strategy on test set.
    
    CRITICAL: Test set is NEVER resampled!
    
    Why?
    - Resampling is only for training
    - Test set should reflect real-world class distribution
    - We want to know: "How well does this model perform on imbalanced data?"
    """
    print("\n" + "="*80)
    print(f"STEP 5: DETAILED EVALUATION - {strategy_name}")
    print("="*80)
    
    # Fit on training data (with sampling)
    print("Training model...")
    start_time = time()
    best_pipeline.fit(X_train, y_train)
    train_time = time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Check if sampler was used and show resampling effect
    if 'sampler' in best_pipeline.named_steps:
        print("\nResampling Effect (Training Set Only):")
        # Get sampler
        sampler = best_pipeline.named_steps['sampler']
        
        # We need to transform data through pipeline up to sampler
        # This is a bit hacky, but shows the concept
        X_temp = X_train.copy()
        for name, step in best_pipeline.steps[:-2]:  # All steps before sampler
            if hasattr(step, 'transform'):
                X_temp = step.transform(X_temp)
        
        # Now apply sampler
        X_resampled, y_resampled = sampler.fit_resample(X_temp, y_train)
        
        original_counts = np.bincount(y_train)
        resampled_counts = np.bincount(y_resampled)
        
        print(f"  Original training set:")
        print(f"    Class 0: {original_counts[0]}")
        print(f"    Class 1: {original_counts[1]}")
        print(f"    Ratio: {original_counts[1]/original_counts[0]:.4f}")
        
        print(f"  After resampling:")
        print(f"    Class 0: {resampled_counts[0]}")
        print(f"    Class 1: {resampled_counts[1]}")
        print(f"    Ratio: {resampled_counts[1]/resampled_counts[0]:.4f}")
    
    # Predict on test set (NO resampling)
    print("\nPredicting on test set (original distribution)...")
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = best_pipeline.predict(X_test)
    
    # Compute metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\nTest Set Performance:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
    
    # Create evaluation plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    prevalence = y_test.mean()
    axes[0, 1].plot(recall, precision, linewidth=2, label=f'PR (AUC={pr_auc:.4f})')
    axes[0, 1].axhline(y=prevalence, color='k', linestyle='--', linewidth=1,
                       label=f'Prevalence={prevalence:.4f}')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(ax=axes[1, 0], cmap='Blues', values_format='d')
    axes[1, 0].set_title(f'Confusion Matrix\n{strategy_name}')
    
    # Plot 4: Class Distribution Comparison
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    
    x = np.arange(2)
    width = 0.35
    
    axes[1, 1].bar(x - width/2, train_counts, width, label='Train', alpha=0.7)
    axes[1, 1].bar(x + width/2, test_counts, width, label='Test', alpha=0.7)
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Class Distribution (Train vs Test)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Class 0', 'Class 1'])
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/best_strategy_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"\nEvaluation plots saved to: {output_dir}/best_strategy_evaluation.png")
    plt.close()
    
    return {
        'test_roc_auc': roc_auc,
        'test_pr_auc': pr_auc,
        'test_f1': f1,
        'test_balanced_accuracy': bal_acc,
        'train_time': train_time
    }


# =============================================================================
# MAIN WORKFLOW WITH MLFLOW
# =============================================================================
def run_imbalanced_workflow_with_mlflow(imbalance_ratio=0.05, experiment_name='imbalanced_ml_workflow'):
    """
    Run complete workflow for imbalanced data with MLflow tracking.
    """
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f'imbalance_{imbalance_ratio:.3f}'):
        
        print("\n" + "="*80)
        print("ML WORKFLOW FOR IMBALANCED DATA WITH MLFLOW")
        print("="*80)
        
        # Create output directory
        import os
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        # Log imbalance ratio
        mlflow.log_param('imbalance_ratio', imbalance_ratio)
        
        # Step 1: Create imbalanced data
        X_train, X_test, y_train, y_test = create_imbalanced_data(
            imbalance_ratio=imbalance_ratio,
            n_samples=5000,
            random_state=42
        )
        
        mlflow.log_param('n_samples', 5000)
        mlflow.log_param('train_samples', len(y_train))
        mlflow.log_param('test_samples', len(y_test))
        
        # Step 2: Create pipelines with different sampling strategies
        pipelines = create_sampling_pipelines()
        
        # Step 3: Compare strategies
        results_df = compare_sampling_strategies(pipelines, X_train, y_train, cv=5)
        
        # Save results
        results_df.to_csv(f'{output_dir}/sampling_comparison.csv', index=False)
        mlflow.log_artifact(f'{output_dir}/sampling_comparison.csv')
        
        # Log best strategy metrics
        best_row = results_df.iloc[0]
        mlflow.log_param('best_strategy', best_row['Strategy'])
        mlflow.log_metric('best_cv_pr_auc', best_row['pr_auc'])
        mlflow.log_metric('best_cv_roc_auc', best_row['roc_auc'])
        
        # Step 4: Visualize comparison
        visualize_strategy_comparison(results_df, output_dir)
        mlflow.log_artifact(f'{output_dir}/sampling_strategy_comparison.png')
        
        # Step 5: Detailed evaluation of best strategy
        best_strategy_name = best_row['Strategy']
        best_pipeline = pipelines[best_strategy_name]
        
        test_metrics = evaluate_best_strategy(
            best_pipeline, X_train, y_train, X_test, y_test,
            best_strategy_name, output_dir
        )
        
        # Log test metrics
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log plots
        mlflow.log_artifact(f'{output_dir}/best_strategy_evaluation.png')
        
        # Save best model
        mlflow.sklearn.log_model(
            best_pipeline,
            artifact_path='model',
            registered_model_name=f'imbalanced_{best_strategy_name.replace(" ", "_")}'
        )
        
        print("\n" + "="*80)
        print("WORKFLOW COMPLETED")
        print("="*80)
        print(f"\nBest Strategy: {best_strategy_name}")
        print(f"Test PR-AUC: {test_metrics['test_pr_auc']:.4f}")
        print(f"Test ROC-AUC: {test_metrics['test_roc_auc']:.4f}")
        print(f"Test F1: {test_metrics['test_f1']:.4f}")
        print(f"Test Balanced Accuracy: {test_metrics['test_balanced_accuracy']:.4f}")
        
        print(f"\nAll artifacts saved to: {output_dir}/")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    """
    Run the imbalanced data workflow.
    
    Try different imbalance ratios to see how strategies perform:
    - 0.30: Mild imbalance (30% minority)
    - 0.10: Moderate imbalance (10% minority)
    - 0.05: Severe imbalance (5% minority)
    - 0.01: Extreme imbalance (1% minority)
    """
    np.random.seed(42)
    
    # Run with severe imbalance (5% minority class)
    run_imbalanced_workflow_with_mlflow(
        imbalance_ratio=0.05,
        experiment_name='imbalanced_ml_workflow'
    )
    
    print("\n✅ Imbalanced data workflow completed!")
    print("Run 'mlflow ui' to view tracked experiments.")
