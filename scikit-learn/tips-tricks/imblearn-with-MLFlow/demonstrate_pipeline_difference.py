"""
CRITICAL DEMONSTRATION: sklearn.pipeline vs imblearn.pipeline
==============================================================

This script demonstrates WHY you MUST use imblearn.pipeline.Pipeline
when working with samplers, not sklearn.pipeline.Pipeline.

It shows the DATA LEAKAGE problem that occurs when sampling is done
incorrectly with cross-validation.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

print("="*80)
print("DEMONSTRATION: Data Leakage with Incorrect Sampling")
print("="*80)

# Create imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    weights=[0.95, 0.05],  # 5% minority class
    random_state=42
)

print(f"\nOriginal dataset:")
print(f"  Total samples: {len(y)}")
print(f"  Class 0 (majority): {np.sum(y == 0)} ({np.sum(y == 0)/len(y):.1%})")
print(f"  Class 1 (minority): {np.sum(y == 1)} ({np.sum(y == 1)/len(y):.1%})")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set:")
print(f"  Total samples: {len(y_train)}")
print(f"  Class 0: {np.sum(y_train == 0)}")
print(f"  Class 1: {np.sum(y_train == 1)}")

# =============================================================================
# ❌ WRONG APPROACH 1: Resample BEFORE cross-validation
# =============================================================================
print("\n" + "="*80)
print("❌ WRONG APPROACH 1: Resample BEFORE cross-validation")
print("="*80)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE (before CV):")
print(f"  Total samples: {len(y_train_resampled)}")
print(f"  Class 0: {np.sum(y_train_resampled == 0)}")
print(f"  Class 1: {np.sum(y_train_resampled == 1)}")

# Now do cross-validation
model = LogisticRegression(max_iter=1000, random_state=42)

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)

scores_wrong = cross_val_score(model, X_train_scaled, y_train_resampled, 
                                cv=5, scoring='roc_auc')

print(f"\n❌ Cross-validation scores (WRONG - resampled before CV):")
print(f"  Scores: {scores_wrong}")
print(f"  Mean ROC-AUC: {scores_wrong.mean():.4f} (+/- {scores_wrong.std():.4f})")

print("\n⚠️  PROBLEM: Data leakage!")
print("   - Synthetic samples were created using ALL training data")
print("   - Validation fold may contain synthetic versions of training samples")
print("   - CV score is OPTIMISTICALLY BIASED (too high)")
print("   - Model performance will be WORSE on real test data")

# =============================================================================
# ❌ WRONG APPROACH 2: Try to use sklearn.pipeline (will fail)
# =============================================================================
print("\n" + "="*80)
print("❌ WRONG APPROACH 2: Using sklearn.pipeline.Pipeline with SMOTE")
print("="*80)

from sklearn.pipeline import Pipeline as SklearnPipeline

try:
    # This will FAIL because sklearn's Pipeline doesn't support samplers
    pipeline_sklearn = SklearnPipeline([
        ('scaler', StandardScaler()),
        ('sampler', SMOTE(random_state=42)),  # ❌ sklearn doesn't know about this
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    scores_sklearn = cross_val_score(pipeline_sklearn, X_train, y_train, 
                                     cv=5, scoring='roc_auc')
    
    print("Unexpectedly succeeded (sklearn version may have changed)")
    
except Exception as e:
    print(f"❌ ERROR (as expected): {type(e).__name__}")
    print(f"   Message: {str(e)[:100]}...")
    print("\n⚠️  PROBLEM: sklearn.pipeline.Pipeline doesn't support samplers!")
    print("   - Pipeline only knows about transformers and estimators")
    print("   - Samplers have fit_resample(), not fit_transform()")
    print("   - We MUST use imblearn.pipeline.Pipeline instead")

# =============================================================================
# ✅ CORRECT APPROACH: Use imblearn.pipeline.Pipeline
# =============================================================================
print("\n" + "="*80)
print("✅ CORRECT APPROACH: Using imblearn.pipeline.Pipeline")
print("="*80)

from imblearn.pipeline import Pipeline as ImbPipeline

pipeline_correct = ImbPipeline([
    ('scaler', StandardScaler()),
    ('sampler', SMOTE(random_state=42)),  # ✅ imblearn handles samplers
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

scores_correct = cross_val_score(pipeline_correct, X_train, y_train, 
                                 cv=5, scoring='roc_auc')

print(f"\n✅ Cross-validation scores (CORRECT - imblearn pipeline):")
print(f"  Scores: {scores_correct}")
print(f"  Mean ROC-AUC: {scores_correct.mean():.4f} (+/- {scores_correct.std():.4f})")

print("\n✅ What happened correctly:")
print("   - In each CV fold:")
print("     1. Split training data into train_fold and val_fold")
print("     2. Fit scaler on train_fold only")
print("     3. Transform both folds with scaler")
print("     4. FIT SMOTE on train_fold only (creates synthetics)")
print("     5. Train classifier on RESAMPLED train_fold")
print("     6. Evaluate on ORIGINAL val_fold (not resampled!)")
print("   - No data leakage!")
print("   - Validation folds never influenced by synthetic samples")

# =============================================================================
# COMPARISON: Wrong vs Correct
# =============================================================================
print("\n" + "="*80)
print("COMPARISON: Impact of Data Leakage")
print("="*80)

print(f"\n❌ Wrong approach (resample before CV):")
print(f"   Mean ROC-AUC: {scores_wrong.mean():.4f}")
print(f"   → Optimistically biased (too high)")

print(f"\n✅ Correct approach (imblearn pipeline):")
print(f"   Mean ROC-AUC: {scores_correct.mean():.4f}")
print(f"   → Unbiased estimate")

if scores_wrong.mean() > scores_correct.mean():
    difference = scores_wrong.mean() - scores_correct.mean()
    print(f"\n⚠️  Data leakage inflated performance by {difference:.4f} ROC-AUC points!")
    print(f"   This is why correct pipeline usage is CRITICAL.")

# =============================================================================
# TEST SET EVALUATION
# =============================================================================
print("\n" + "="*80)
print("TEST SET EVALUATION (The moment of truth)")
print("="*80)

# Train both approaches on full training set
print("\nTraining both models on full training set...")

# Wrong approach model
smote_wrong = SMOTE(random_state=42)
X_train_resampled_wrong, y_train_resampled_wrong = smote_wrong.fit_resample(X_train, y_train)
scaler_wrong = StandardScaler()
X_train_scaled_wrong = scaler_wrong.fit_transform(X_train_resampled_wrong)
model_wrong = LogisticRegression(max_iter=1000, random_state=42)
model_wrong.fit(X_train_scaled_wrong, y_train_resampled_wrong)

# Correct approach model
pipeline_correct.fit(X_train, y_train)

# Evaluate on test set (NEVER resampled!)
print("\nEvaluating on test set (original class distribution)...")

from sklearn.metrics import roc_auc_score

# Wrong approach: need to scale test set with same scaler
X_test_scaled_wrong = scaler_wrong.transform(X_test)
test_score_wrong = roc_auc_score(y_test, model_wrong.predict_proba(X_test_scaled_wrong)[:, 1])

# Correct approach: pipeline handles everything
test_score_correct = roc_auc_score(y_test, pipeline_correct.predict_proba(X_test)[:, 1])

print(f"\n❌ Wrong approach test ROC-AUC: {test_score_wrong:.4f}")
print(f"✅ Correct approach test ROC-AUC: {test_score_correct:.4f}")

print("\n" + "="*80)
print("KEY LESSONS")
print("="*80)
print("""
1. ALWAYS use imblearn.pipeline.Pipeline when using samplers
   - Never use sklearn.pipeline.Pipeline with samplers
   
2. NEVER resample before cross-validation
   - Resampling must happen INSIDE each CV fold
   
3. NEVER resample the test set
   - Test set should reflect real-world distribution
   
4. Sampling happens during training, not prediction
   - fit_resample() only called on training data
   - Prediction uses original samples
   
5. Data leakage from incorrect sampling can significantly
   overestimate model performance
   
Remember: The pipeline ensures that ALL preprocessing steps
(including sampling) happen correctly inside each CV fold!
""")

print("✅ Always use: from imblearn.pipeline import Pipeline")
print("❌ Never use:  from sklearn.pipeline import Pipeline (with samplers)")
