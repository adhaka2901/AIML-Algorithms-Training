"""
Neural Network Configuration
Defines all hyperparameters, architectures, and experimental settings for NN experiments.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch

@dataclass
class NNArchitectureConfig:
    """Configuration for a single NN architecture."""
    name: str
    hidden_layers: List[int]  # e.g., [512, 512] or [256, 256, 128, 128]
    activation: str = 'relu'  # relu, tanh, sigmoid
    dropout_rate: float = 0.0  # 0.0 to 0.2
    
    @property
    def depth(self) -> int:
        """Number of hidden layers."""
        return len(self.hidden_layers)
    
    def estimate_params(self, input_dim: int, output_dim: int) -> int:
        """
        Estimate total trainable parameters.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (1 for binary/regression, n_classes for multiclass)
        
        Returns:
            Approximate parameter count
        """
        params = 0
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_layers:
            params += prev_dim * hidden_dim + hidden_dim  # weights + bias
            prev_dim = hidden_dim
        
        # Output layer
        params += prev_dim * output_dim + output_dim
        
        return params


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimizer (SGD ONLY per assignment requirements)
    optimizer: str = 'sgd'
    learning_rate: float = 0.01
    momentum: float = 0.0  # Must be 0 per requirements
    
    # Training protocol
    batch_size: int = 512  # 512-2048 range
    max_epochs: int = 15
    early_stopping_patience: int = 3
    
    # Regularization
    l2_weight_decay: float = 1e-4  # 1e-4 to 1e-3
    
    # Precision
    dtype: str = 'float32'
    
    # Reproducibility
    random_seed: int = 42
    
    def validate(self):
        """Ensure configuration meets assignment requirements."""
        assert self.optimizer == 'sgd', "Only SGD allowed (no momentum variants)"
        assert self.momentum == 0.0, "No momentum allowed"
        assert 512 <= self.batch_size <= 2048, "Batch size must be in [512, 2048]"
        assert self.max_epochs <= 15, "Max 15 epochs allowed"
        assert 2 <= self.early_stopping_patience <= 3, "Patience should be 2-3"
        assert 1e-4 <= self.l2_weight_decay <= 1e-3, "L2 should be in [1e-4, 1e-3]"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    dataset_name: str  # 'hotel' or 'accidents'
    task_type: str  # 'classification' or 'regression'
    target_column: str
    
    # Architecture comparison (must compare 2)
    architectures: List[NNArchitectureConfig]
    training: TrainingConfig
    
    # Data splits
    test_size: float = 0.2
    val_size: float = 0.2  # From remaining train
    stratify: bool = True
    
    # Parameter budget
    min_params: int = 200_000  # 0.2M
    max_params: int = 1_000_000  # 1.0M


# ============================================================================
# PREDEFINED ARCHITECTURE PAIRS (with similar parameter counts)
# ============================================================================

def get_shallow_wide_arch(name: str = "shallow_wide") -> NNArchitectureConfig:
    """Shallow-wide architecture: 2 layers."""
    return NNArchitectureConfig(
        name=name,
        hidden_layers=[512, 512],
        activation='relu',
        dropout_rate=0.0
    )

def get_deep_narrow_arch(name: str = "deep_narrow") -> NNArchitectureConfig:
    """Deep-narrow architecture: 4 layers with similar param count to shallow-wide."""
    return NNArchitectureConfig(
        name=name,
        hidden_layers=[256, 256, 128, 128],
        activation='relu',
        dropout_rate=0.0
    )

def get_medium_arch(name: str = "medium") -> NNArchitectureConfig:
    """Medium architecture: 3 layers."""
    return NNArchitectureConfig(
        name=name,
        hidden_layers=[384, 256, 128],
        activation='relu',
        dropout_rate=0.0
    )


# ============================================================================
# DATASET-SPECIFIC CONFIGURATIONS
# ============================================================================

def get_hotel_classification_config() -> ExperimentConfig:
    """
    Configuration for Hotel Booking cancellation prediction.
    Target: is_canceled (binary classification)
    """
    return ExperimentConfig(
        dataset_name='hotel',
        task_type='classification',
        target_column='is_canceled',
        architectures=[
            get_shallow_wide_arch("hotel_shallow"),
            get_deep_narrow_arch("hotel_deep")
        ],
        training=TrainingConfig(
            learning_rate=0.01,
            batch_size=512,
            max_epochs=15,
            early_stopping_patience=3,
            l2_weight_decay=1e-4,
            random_seed=42
        ),
        test_size=0.2,
        val_size=0.2,
        stratify=True
    )

def get_hotel_regression_config() -> ExperimentConfig:
    """
    Configuration for Hotel Booking ADR prediction.
    Target: adr (regression)
    """
    return ExperimentConfig(
        dataset_name='hotel',
        task_type='regression',
        target_column='adr',
        architectures=[
            get_shallow_wide_arch("hotel_reg_shallow"),
            get_deep_narrow_arch("hotel_reg_deep")
        ],
        training=TrainingConfig(
            learning_rate=0.001,  # Lower LR for regression
            batch_size=1024,
            max_epochs=15,
            early_stopping_patience=3,
            l2_weight_decay=5e-4,
            random_seed=42
        ),
        test_size=0.2,
        val_size=0.2,
        stratify=False  # No stratification for regression
    )

def get_accidents_classification_config() -> ExperimentConfig:
    """
    Configuration for US Accidents severity prediction.
    Target: Severity (multiclass 1-4)
    """
    return ExperimentConfig(
        dataset_name='accidents',
        task_type='classification',
        target_column='Severity',
        architectures=[
            get_shallow_wide_arch("accidents_shallow"),
            get_deep_narrow_arch("accidents_deep")
        ],
        training=TrainingConfig(
            learning_rate=0.005,
            batch_size=2048,  # Larger batch for large dataset
            max_epochs=15,
            early_stopping_patience=2,
            l2_weight_decay=1e-3,  # Higher regularization for large data
            random_seed=42
        ),
        test_size=0.2,
        val_size=0.2,
        stratify=True
    )


# ============================================================================
# HYPERPARAMETER SEARCH SPACES (for model complexity curves)
# ============================================================================

LEARNING_RATE_SEARCH = [0.001, 0.005, 0.01, 0.05, 0.1]
L2_SEARCH = [1e-5, 1e-4, 5e-4, 1e-3]
BATCH_SIZE_SEARCH = [512, 1024, 2048]

# Width variations (keeping similar param count)
WIDTH_VARIATIONS = {
    'shallow_2_layers': [
        [256, 256],   # ~65k params (with 100 input, 1 output)
        [384, 384],   # ~150k params
        [512, 512],   # ~260k params
        [640, 640],   # ~410k params
    ],
    'deep_4_layers': [
        [128, 128, 64, 64],    # ~33k params
        [192, 192, 96, 96],    # ~75k params
        [256, 256, 128, 128],  # ~130k params
        [320, 320, 160, 160],  # ~205k params
    ]
}


if __name__ == "__main__":
    # Validate and display configurations
    print("=" * 80)
    print("NEURAL NETWORK CONFIGURATION VALIDATION")
    print("=" * 80)
    
    configs = [
        get_hotel_classification_config(),
        get_hotel_regression_config(),
        get_accidents_classification_config()
    ]
    
    for config in configs:
        print(f"\n{config.dataset_name.upper()} - {config.task_type}")
        print("-" * 80)
        
        # Validate training config
        config.training.validate()
        print("✓ Training config validated")
        
        # Show architecture details
        for arch in config.architectures:
            # Assume 100 input features and 1/4 output dims
            input_dim = 100
            output_dim = 1 if config.task_type == 'regression' else 4
            params = arch.estimate_params(input_dim, output_dim)
            
            print(f"\n  Architecture: {arch.name}")
            print(f"    Layers: {arch.hidden_layers}")
            print(f"    Depth: {arch.depth}")
            print(f"    Est. params (100→{output_dim}): {params:,}")
            print(f"    Within budget: {config.min_params <= params <= config.max_params}")
