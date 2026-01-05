"""
Neural Network Architectures
Defines flexible MLP architectures with parameter counting and validation.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np


class FlexibleMLP(nn.Module):
    """
    Flexible Multi-Layer Perceptron.
    
    Supports:
    - Arbitrary depth/width
    - Different activation functions
    - Dropout regularization
    - Classification and regression tasks
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        task_type: str = 'classification'
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes, e.g., [512, 512] or [256, 256, 128, 128]
            output_dim: Number of output units (1 for binary/regression, n_classes for multiclass)
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'gelu', 'silu')
            dropout_rate: Dropout probability (0.0 to 0.2)
            task_type: 'classification' or 'regression'
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.task_type = task_type
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_layers):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Activation
            layers.append(self._get_activation(activation))
            
            # Dropout (if specified)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Output activation (task-dependent)
        if task_type == 'classification':
            if output_dim == 1:
                # Binary classification
                layers.append(nn.Sigmoid())
            else:
                # Multiclass classification
                layers.append(nn.Softmax(dim=1))
        # Regression: no output activation (linear)
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights(activation)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),  # Also known as Swish
        }
        
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        
        return activations[name.lower()]
    
    def _initialize_weights(self, activation: str):
        """
        Initialize weights based on activation function.
        
        - He initialization for ReLU-family (ReLU, GELU, SiLU)
        - Xavier initialization for tanh, sigmoid
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if activation.lower() in ['relu', 'gelu', 'silu']:
                    # He initialization
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                else:
                    # Xavier initialization
                    nn.init.xavier_normal_(module.weight)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable and total parameters.
        
        Returns:
            Dictionary with counts
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        # Breakdown by layer type
        linear_params = sum(
            p.numel() for m in self.modules() 
            if isinstance(m, nn.Linear) 
            for p in m.parameters()
        )
        
        return {
            'trainable': trainable,
            'total': total,
            'linear_params': linear_params,
            'non_linear_params': total - linear_params
        }
    
    def get_architecture_summary(self) -> str:
        """Get human-readable architecture summary."""
        param_counts = self.count_parameters()
        
        summary = [
            f"Architecture: {self.input_dim} → {' → '.join(map(str, self.hidden_layers))} → {self.output_dim}",
            f"Depth: {len(self.hidden_layers)} hidden layers",
            f"Activation: {self.activation_name}",
            f"Dropout: {self.dropout_rate}",
            f"Task: {self.task_type}",
            f"Total parameters: {param_counts['trainable']:,}",
        ]
        
        return "\n".join(summary)


class ArchitectureValidator:
    """
    Validates NN architectures against assignment requirements.
    """
    
    MIN_PARAMS = 200_000  # 0.2M
    MAX_PARAMS = 1_000_000  # 1.0M
    
    @staticmethod
    def validate_architecture(
        model: FlexibleMLP,
        strict: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate architecture against assignment constraints.
        
        Args:
            model: FlexibleMLP instance
            strict: If True, enforce parameter budget strictly
        
        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []
        is_valid = True
        
        param_count = model.count_parameters()['trainable']
        
        # Check parameter budget
        if param_count < ArchitectureValidator.MIN_PARAMS:
            msg = f"Too few parameters: {param_count:,} < {ArchitectureValidator.MIN_PARAMS:,}"
            warnings.append(msg)
            if strict:
                is_valid = False
        
        if param_count > ArchitectureValidator.MAX_PARAMS:
            msg = f"Too many parameters: {param_count:,} > {ArchitectureValidator.MAX_PARAMS:,}"
            warnings.append(msg)
            if strict:
                is_valid = False
        
        # Check depth constraints
        if len(model.hidden_layers) < 2:
            warnings.append("Depth < 2 layers: may not show depth vs. width tradeoffs")
        
        if len(model.hidden_layers) > 6:
            warnings.append("Depth > 6 layers: may be harder to train with SGD")
        
        # Check dropout
        if model.dropout_rate > 0.2:
            warnings.append(f"Dropout {model.dropout_rate} > 0.2 (assignment max)")
            is_valid = False
        
        return is_valid, warnings
    
    @staticmethod
    def compare_architectures(
        arch1: FlexibleMLP,
        arch2: FlexibleMLP,
        tolerance: float = 0.1
    ) -> Dict:
        """
        Compare two architectures (e.g., shallow-wide vs. deep-narrow).
        
        Args:
            arch1: First architecture
            arch2: Second architecture
            tolerance: Acceptable parameter difference (as fraction)
        
        Returns:
            Comparison dictionary
        """
        params1 = arch1.count_parameters()['trainable']
        params2 = arch2.count_parameters()['trainable']
        
        param_diff_abs = abs(params1 - params2)
        param_diff_frac = param_diff_abs / max(params1, params2)
        
        return {
            'arch1_params': params1,
            'arch2_params': params2,
            'diff_absolute': param_diff_abs,
            'diff_fraction': param_diff_frac,
            'within_tolerance': param_diff_frac <= tolerance,
            'arch1_depth': len(arch1.hidden_layers),
            'arch2_depth': len(arch2.hidden_layers),
            'arch1_summary': arch1.get_architecture_summary(),
            'arch2_summary': arch2.get_architecture_summary()
        }


def build_model_from_config(
    config_dict: Dict,
    input_dim: int,
    output_dim: int
) -> FlexibleMLP:
    """
    Build model from configuration dictionary.
    
    Args:
        config_dict: Dictionary with 'hidden_layers', 'activation', etc.
        input_dim: Input feature dimension
        output_dim: Output dimension
    
    Returns:
        Configured FlexibleMLP model
    """
    return FlexibleMLP(
        input_dim=input_dim,
        hidden_layers=config_dict['hidden_layers'],
        output_dim=output_dim,
        activation=config_dict.get('activation', 'relu'),
        dropout_rate=config_dict.get('dropout_rate', 0.0),
        task_type=config_dict.get('task_type', 'classification')
    )


if __name__ == "__main__":
    print("=" * 80)
    print("ARCHITECTURE VALIDATION EXAMPLE")
    print("=" * 80)
    
    # Example: Compare shallow-wide vs. deep-narrow
    input_dim = 100
    output_dim = 1  # Binary classification
    
    # Shallow-wide: 2 layers
    shallow = FlexibleMLP(
        input_dim=input_dim,
        hidden_layers=[512, 512],
        output_dim=output_dim,
        activation='relu',
        task_type='classification'
    )
    
    # Deep-narrow: 4 layers
    deep = FlexibleMLP(
        input_dim=input_dim,
        hidden_layers=[256, 256, 128, 128],
        output_dim=output_dim,
        activation='relu',
        task_type='classification'
    )
    
    print("\nSHALLOW-WIDE:")
    print(shallow.get_architecture_summary())
    valid, warnings = ArchitectureValidator.validate_architecture(shallow)
    print(f"Valid: {valid}")
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    
    print("\n" + "-" * 80)
    print("\nDEEP-NARROW:")
    print(deep.get_architecture_summary())
    valid, warnings = ArchitectureValidator.validate_architecture(deep)
    print(f"Valid: {valid}")
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    comparison = ArchitectureValidator.compare_architectures(shallow, deep)
    print(f"Shallow params: {comparison['arch1_params']:,}")
    print(f"Deep params: {comparison['arch2_params']:,}")
    print(f"Difference: {comparison['diff_absolute']:,} ({comparison['diff_fraction']*100:.1f}%)")
    print(f"Within tolerance: {comparison['within_tolerance']}")
