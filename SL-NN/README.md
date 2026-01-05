# Neural Network Implementation Plan
## CS7641 Supervised Learning Assignment - Fall 2025

This repository contains a complete, production-ready implementation for the Neural Network component of the Supervised Learning assignment.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Implementation Overview](#implementation-overview)
3. [Detailed Phase Plan](#detailed-phase-plan)
4. [Module Documentation](#module-documentation)
5. [Assignment Requirements Checklist](#assignment-requirements-checklist)
6. [Workflow Examples](#workflow-examples)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import pandas as pd
from nn_config import get_hotel_classification_config
from nn_experiment import NNExperiment

# 1. Load configuration
config = get_hotel_classification_config()

# 2. Initialize experiment
experiment = NNExperiment(config, results_dir='./results')

# 3. Load data
df = pd.read_csv('hotel_bookings.csv')
data = experiment.load_and_preprocess_data(df)

# 4. Run experiments
experiment.compare_architectures(data)
experiment.generate_learning_curves(data)
experiment.generate_complexity_curves(data)
experiment.final_evaluation(data)
experiment.save_results()
```

---

## 📊 Implementation Overview

### Architecture

```
nn_experiments/
├── config/
│   ├── nn_config.py           # All hyperparameter configurations
│   └── data_config.py         # Dataset-specific settings
│
├── data/
│   ├── preprocessing.py       # Feature engineering, encoding, scaling
│   ├── loaders.py            # Data loading utilities
│   └── leakage_control.py    # Dataset-specific leakage prevention
│
├── models/
│   ├── architectures.py      # FlexibleMLP, architecture validation
│   └── trainer.py            # Training loop, early stopping
│
├── evaluation/
│   ├── evaluator.py          # Metrics, all required plots
│   └── visualizations.py     # Additional visualization utilities
│
├── experiments/
│   ├── nn_experiment.py      # Main experiment runner
│   ├── hotel_nn.py           # Hotel-specific experiments
│   └── accidents_nn.py       # US Accidents experiments
│
└── results/
    ├── figures/              # All generated plots
    ├── metrics/              # JSON metric logs
    └── models/               # Saved model checkpoints
```

### Design Principles

1. **Modular**: Each component is self-contained and testable
2. **Reproducible**: All random seeds set, deterministic behavior
3. **Assignment-Compliant**: Follows all requirements strictly
4. **Production-Ready**: Proper error handling, logging, validation
5. **Extensible**: Easy to add new architectures, datasets, metrics

---

## 🗓️ Detailed Phase Plan

### Phase 1: Setup & Configuration (Days 1-2)

**Goal**: Environment ready, configurations validated

**Tasks**:
1. ✅ Install dependencies (`requirements.txt`)
2. ✅ Set up project structure
3. ✅ Create configuration files for both datasets
4. ✅ Validate parameter budgets (0.2M - 1.0M params)

**Deliverables**:
- Working environment
- Validated configs for Hotel (classification) and US Accidents (classification/regression)

**Code Example**:
```python
from nn_config import *

# Validate configurations
configs = [
    get_hotel_classification_config(),
    get_accidents_classification_config()
]

for config in configs:
    config.training.validate()
    print(f"✓ {config.dataset_name} config validated")
```

---

### Phase 2: Data Preprocessing (Days 3-4)

**Goal**: Clean, encoded, scaled data with proper leakage controls

**Tasks**:
1. ✅ Implement `DataPreprocessor`
2. ✅ Apply dataset-specific leakage controls
3. ✅ Implement target/frequency encoding for high-cardinality features
4. ✅ Create reproducible train/val/test splits
5. ✅ Validate preprocessing pipeline

**Key Leakage Controls**:

| Dataset | Target | Leakage Columns to Drop |
|---------|--------|------------------------|
| Hotel | `is_canceled` | `reservation_status`, `reservation_status_date` |
| US Accidents | `Severity` | `End_Time`, post-dated `Weather_Timestamp` |

**Code Example**:
```python
from preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(
    dataset_name='hotel',
    task_type='classification',
    target_column='is_canceled'
)

# Process data
data = preprocessor.fit_transform(
    df,
    test_size=0.2,
    val_size=0.2,
    stratify=True
)

print(f"Train: {data['X_train'].shape}")
print(f"Features: {data['n_features']}")
```

---

### Phase 3: Architecture Implementation (Days 5-6)

**Goal**: Two architectures implemented and validated

**Tasks**:
1. ✅ Implement `FlexibleMLP` with configurable depth/width
2. ✅ Implement parameter counting
3. ✅ Create shallow-wide (2 layers) architecture
4. ✅ Create deep-narrow (4 layers) architecture
5. ✅ Validate parameter budgets match

**Architecture Comparison**:

| Architecture | Layers | Example | Est. Params (100→1) |
|--------------|--------|---------|-------------------|
| Shallow-Wide | 2 | [512, 512] | ~260,000 |
| Deep-Narrow | 4 | [256, 256, 128, 128] | ~130,000 |

**Code Example**:
```python
from architectures import FlexibleMLP, ArchitectureValidator

# Build models
shallow = FlexibleMLP(
    input_dim=100,
    hidden_layers=[512, 512],
    output_dim=1,
    activation='relu'
)

deep = FlexibleMLP(
    input_dim=100,
    hidden_layers=[256, 256, 128, 128],
    output_dim=1,
    activation='relu'
)

# Compare
comparison = ArchitectureValidator.compare_architectures(shallow, deep)
print(comparison)
```

---

### Phase 4: Training Pipeline (Days 7-8)

**Goal**: SGD training with early stopping working

**Tasks**:
1. ✅ Implement `NNTrainer` with SGD (NO momentum)
2. ✅ Implement early stopping (patience 2-3)
3. ✅ Track training history
4. ✅ Implement metric computation
5. ✅ Validate training on sample data

**SGD Requirements** (CRITICAL):
- ✅ Optimizer: `torch.optim.SGD`
- ✅ Momentum: 0.0 (strictly enforced)
- ✅ L2 weight decay: 1e-4 to 1e-3
- ✅ Batch size: 512-2048
- ✅ Max epochs: ≤15

**Code Example**:
```python
from trainer import NNTrainer

trainer = NNTrainer(
    model=model,
    task_type='classification',
    learning_rate=0.01,
    batch_size=512,
    max_epochs=15,
    early_stopping_patience=3,
    l2_weight_decay=1e-4
)

history = trainer.fit(X_train, y_train, X_val, y_val)
print(f"Best epoch: {trainer.best_epoch}")
```

---

### Phase 5: Evaluation & Visualization (Days 9-10)

**Goal**: All required plots generated

**Tasks**:
1. ✅ Implement epoch curves (loss/metric vs. epoch)
2. ✅ Implement learning curves (metric vs. training size)
3. ✅ Implement model complexity curves
4. ✅ Implement classification plots (ROC, PR, confusion matrix, calibration)
5. ✅ Implement regression plots (residuals)

**Required Plots**:

| Plot Type | Purpose | When |
|-----------|---------|------|
| Epoch Curve | Show early stopping | All models |
| Learning Curve | Diagnose bias/variance | All models |
| Complexity Curve | Hyperparameter sensitivity | All models |
| ROC Curve | Classification performance | Classification |
| PR Curve | Imbalanced data | Classification (imbalanced) |
| Confusion Matrix | Error breakdown | Classification |
| Calibration Curve | Probability quality | Classification |
| Residual Plots | Regression diagnostics | Regression |

**Code Example**:
```python
from evaluator import NNEvaluator

evaluator = NNEvaluator(task_type='classification')

# Epoch curve
evaluator.plot_epoch_curve(history, best_epoch)

# Learning curve
evaluator.plot_learning_curve(train_sizes, train_scores, val_scores)

# ROC & PR curves
evaluator.plot_roc_curve(y_true, y_proba)
evaluator.plot_pr_curve(y_true, y_proba)
```

---

### Phase 6: Full Experiments (Days 11-13)

**Goal**: Complete experiments on both datasets

**Tasks**:
1. Run Hotel Booking experiments
   - Classification: `is_canceled`
   - Compare shallow-wide vs. deep-narrow
   - Generate all required plots
2. Run US Accidents experiments
   - Classification: `Severity`
   - Handle large dataset (subsample if needed with justification)
3. Document all hyperparameters, seeds, hardware
4. Save all results, figures, metrics

**Code Example**:
```python
from nn_experiment import NNExperiment

# Hotel experiment
config = get_hotel_classification_config()
experiment = NNExperiment(config)

df = pd.read_csv('hotel_bookings.csv')
data = experiment.load_and_preprocess_data(df)

experiment.compare_architectures(data)
experiment.generate_learning_curves(data)
experiment.generate_complexity_curves(data)
experiment.save_results()
```

---

### Phase 7: Analysis & Report Writing (Days 14-16)

**Goal**: Complete report with analysis

**Tasks**:
1. Interpret all figures
2. Compare architectures across datasets
3. Discuss trade-offs (accuracy vs. speed, depth vs. width)
4. Tie to course concepts (bias-variance, overfitting, regularization)
5. Write LaTeX report in Overleaf
6. Ensure all citations present

---

## 📚 Module Documentation

### `nn_config.py`

Centralized configuration management.

**Key Components**:
- `NNArchitectureConfig`: Architecture specifications
- `TrainingConfig`: Training hyperparameters
- `ExperimentConfig`: Complete experiment setup
- Predefined configs: `get_hotel_classification_config()`, etc.

**Example**:
```python
config = get_hotel_classification_config()
config.training.validate()  # Ensures compliance
```

---

### `preprocessing.py`

Data preprocessing with leakage prevention.

**Key Features**:
- Dataset-specific leakage controls
- Target/frequency encoding for high-cardinality features
- StandardScaler (fit on train only)
- Reproducible splits

**Example**:
```python
preprocessor = DataPreprocessor('hotel', 'classification', 'is_canceled')
data = preprocessor.fit_transform(df, test_size=0.2, val_size=0.2)
```

---

### `architectures.py`

Flexible MLP with validation.

**Key Classes**:
- `FlexibleMLP`: Main model class
- `ArchitectureValidator`: Checks parameter budgets

**Example**:
```python
model = FlexibleMLP(
    input_dim=100,
    hidden_layers=[512, 512],
    output_dim=1,
    activation='relu',
    task_type='classification'
)

params = model.count_parameters()
print(f"Parameters: {params['trainable']:,}")
```

---

### `trainer.py`

Training loop with early stopping.

**Key Features**:
- SGD only (enforced)
- Early stopping with patience
- Comprehensive metric tracking
- Train/val monitoring

**Example**:
```python
trainer = NNTrainer(model, task_type='classification')
history = trainer.fit(X_train, y_train, X_val, y_val)
summary = trainer.get_training_summary()
```

---

### `evaluator.py`

All evaluation metrics and plots.

**Key Methods**:
- `plot_epoch_curve()`: Loss/metric vs. epoch
- `plot_learning_curve()`: Metric vs. training size
- `plot_complexity_curve()`: Metric vs. hyperparameter
- `plot_roc_curve()`, `plot_pr_curve()`: Classification
- `plot_residuals()`: Regression
- `compute_metrics()`: All metrics

**Example**:
```python
evaluator = NNEvaluator(task_type='classification')
metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
evaluator.print_metrics(metrics)
```

---

## ✅ Assignment Requirements Checklist

### Data Requirements
- [ ] Leakage controls documented and applied
- [ ] Train/val/test splits with fixed seeds
- [ ] Stratified splits for classification
- [ ] Feature scaling (StandardScaler)
- [ ] Target/frequency encoding for high-cardinality

### Neural Network Requirements
- [ ] SGD only (no momentum, no Adam/etc.)
- [ ] Two architectures compared (shallow-wide vs. deep-narrow)
- [ ] Parameter count: 0.2M - 1.0M
- [ ] Batch size: 512-2048
- [ ] Max epochs: ≤15
- [ ] Early stopping: patience 2-3
- [ ] L2 regularization: 1e-4 to 1e-3
- [ ] float32 precision

### Required Plots (per algorithm, per dataset)
- [ ] Epoch curve (with early stopping marker)
- [ ] Learning curve (metric vs. training size)
- [ ] Model complexity curve (metric vs. hyperparameter)
- [ ] Classification: ROC curve
- [ ] Classification: PR curve (with prevalence baseline)
- [ ] Classification: Confusion matrix
- [ ] Classification: Calibration curve
- [ ] Regression: Residual plots

### Metrics
- [ ] Classification: Accuracy, F1, ROC-AUC, PR-AUC
- [ ] Regression: MAE, MSE, RMSE, Median AE
- [ ] Wall-clock time (fit + predict)
- [ ] Hardware documentation

### Analysis
- [ ] Architecture comparison with evidence
- [ ] Trade-offs discussed (metric vs. time, depth vs. width)
- [ ] Bias/variance diagnosis from learning curves
- [ ] Connection to course concepts
- [ ] Type I/II error discussion (classification)
- [ ] Next steps identified

### Reproducibility
- [ ] All random seeds set and documented
- [ ] Exact hyperparameters recorded
- [ ] Hardware specifications noted
- [ ] Data preprocessing steps documented

---

## 🔧 Workflow Examples

### Complete Workflow: Hotel Classification

```python
# 1. Setup
from nn_config import get_hotel_classification_config
from nn_experiment import NNExperiment
import pandas as pd

# 2. Load config
config = get_hotel_classification_config()

# 3. Initialize experiment
experiment = NNExperiment(
    config=config,
    results_dir='./results/hotel',
    random_seed=42
)

# 4. Load data
df = pd.read_csv('data/hotel_bookings.csv')

# 5. Preprocess with leakage controls
data = experiment.load_and_preprocess_data(df)

# 6. Compare architectures
experiment.compare_architectures(data)

# 7. Generate learning curves
experiment.generate_learning_curves(data)

# 8. Generate complexity curves
experiment.generate_complexity_curves(data)

# 9. Final evaluation
experiment.final_evaluation(data)

# 10. Save everything
experiment.save_results()
```

### Custom Architecture Experiment

```python
from architectures import FlexibleMLP
from trainer import NNTrainer

# Custom architecture
model = FlexibleMLP(
    input_dim=data['n_features'],
    hidden_layers=[384, 256, 128],  # Custom 3-layer
    output_dim=1,
    activation='relu'
)

# Train
trainer = NNTrainer(
    model=model,
    task_type='classification',
    learning_rate=0.005,
    batch_size=1024
)

history = trainer.fit(
    data['X_train'],
    data['y_train'],
    data['X_val'],
    data['y_val']
)

# Evaluate
from evaluator import NNEvaluator
evaluator = NNEvaluator(task_type='classification')
evaluator.plot_epoch_curve(history, trainer.best_epoch)
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Parameter count outside budget"
```python
# Solution: Adjust architecture
# Check actual count:
params = model.count_parameters()
print(f"Current: {params['trainable']:,}")

# Reduce width or depth to fit budget
```

**Issue**: "Early stopping too aggressive"
```python
# Solution: Increase patience
config.training.early_stopping_patience = 3  # Instead of 2
```

**Issue**: "Training unstable/diverging"
```python
# Solutions:
# 1. Lower learning rate
config.training.learning_rate = 0.001  # Instead of 0.01

# 2. Increase L2 regularization
config.training.l2_weight_decay = 1e-3  # Instead of 1e-4

# 3. Ensure features are scaled
# (preprocessing.py does this automatically)
```

**Issue**: "GPU out of memory"
```python
# Solutions:
# 1. Reduce batch size
config.training.batch_size = 512  # Instead of 2048

# 2. Use CPU
trainer = NNTrainer(..., device='cpu')

# 3. For US Accidents: subsample data
df_sample = df.sample(frac=0.5, random_state=42)
```

---

## 📖 References

1. Assignment PDF: `SL_Report_Fall_2025_v6-1.pdf`
2. PyTorch Documentation: https://pytorch.org/docs/
3. scikit-learn API: https://scikit-learn.org/stable/
4. Category Encoders: https://contrib.scikit-learn.org/category_encoders/

---

## 🎯 Success Criteria

Your implementation is ready for submission when:

1. ✅ All required plots generate without errors
2. ✅ Parameter budgets validated (0.2M - 1.0M)
3. ✅ SGD training (no momentum) works
4. ✅ Early stopping triggers appropriately
5. ✅ Leakage controls applied and documented
6. ✅ Results reproducible with fixed seeds
7. ✅ All metrics computed correctly
8. ✅ Hardware and runtime logged
9. ✅ Code is modular and well-documented
10. ✅ Ready to write analysis

---

## 📝 Next Steps

1. **Week 1**: Complete Phases 1-4 (setup through training)
2. **Week 2**: Complete Phases 5-6 (evaluation and experiments)
3. **Week 3**: Complete Phase 7 (report writing)

Good luck with your implementation! 🚀
