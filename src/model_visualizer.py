from visualization_base import VisualizationBase
import torch
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import plotly.graph_objects as go
from pathlib import Path
import logging

class ModelVisualizer(VisualizationBase):
    """Concrete implementation of VisualizationBase for model visualization"""
    
    def __init__(self, save_dir: str = 'assets/model_viz'):
        super().__init__(save_dir)
        self.logger = logging.getLogger(__name__)
        
    def plot_model_architecture(self, model: nn.Module) -> None:
        """Plot model architecture using torchviz"""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = model(dummy_input)
            
            # Generate computational graph
            dot = make_dot(dummy_output, params=dict(model.named_parameters()))
            
            # Customize graph appearance
            dot.attr(rankdir='TB')
            dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
            
            # Save as PNG and PDF
            dot.render(str(self.save_dir / 'model_architecture'), format='png', cleanup=True)
            dot.render(str(self.save_dir / 'model_architecture'), format='pdf', cleanup=True)
            
            self.logger.info(f"Model architecture plots saved to {self.save_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to plot model architecture: {str(e)}")
    
    def plot_feature_maps(self, model: nn.Module,
                         sample_input: torch.Tensor,
                         layer_names: Optional[List[str]] = None) -> None:
        """Plot feature maps from specified layers"""
        try:
            # Register hooks to capture feature maps
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
            
            # If no layer names provided, use all conv and linear layers
            if layer_names is None:
                layer_names = []
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        layer_names.append(name)
            
            # Register hooks for each layer
            hooks = []
            for name in layer_names:
                layer = dict([*model.named_modules()])[name]
                hooks.append(layer.register_forward_hook(get_activation(name)))
            
            # Forward pass
            with torch.no_grad():
                _ = model(sample_input)
            
            # Plot feature maps
            for name, feat in activation.items():
                if len(feat.shape) == 4:  # Conv layer
                    self._plot_conv_features(feat, name)
                elif len(feat.shape) == 2:  # Linear layer
                    self._plot_linear_features(feat, name)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
                
            self.logger.info(f"Feature maps saved to {self.save_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to plot feature maps: {str(e)}")
    
    def _plot_conv_features(self, features: torch.Tensor, layer_name: str) -> None:
        """Plot convolutional layer feature maps"""
        features = features[0]  # Take first batch
        num_features = min(64, features.shape[0])  # Show max 64 features
        
        # Create grid plot
        grid_size = int(np.ceil(np.sqrt(num_features)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        
        for idx in range(num_features):
            i, j = idx // grid_size, idx % grid_size
            if grid_size > 1:
                ax = axes[i, j]
            else:
                ax = axes
            
            # Plot feature map
            feature = features[idx].cpu().numpy()
            im = ax.imshow(feature, cmap='viridis')
            ax.axis('off')
        
        # Remove empty subplots
        for idx in range(num_features, grid_size * grid_size):
            i, j = idx // grid_size, idx % grid_size
            if grid_size > 1:
                fig.delaxes(axes[i, j])
        
        plt.suptitle(f'Feature Maps: {layer_name}')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'feature_maps_{layer_name}.png')
        plt.close()
        
        # Create interactive version
        fig = go.Figure()
        
        for idx in range(num_features):
            feature = features[idx].cpu().numpy()
            fig.add_trace(
                go.Heatmap(z=feature,
                          showscale=False,
                          name=f'Channel {idx}')
            )
        
        fig.update_layout(
            title=f'Feature Maps: {layer_name}',
            updatemenus=[{
                'buttons': [
                    {
                        'args': [{'visible': [i == j for j in range(num_features)]}],
                        'label': f'Channel {i}',
                        'method': 'restyle'
                    } for i in range(num_features)
                ],
                'direction': 'down',
                'showactive': True,
            }]
        )
        
        fig.write_html(self.save_dir / f'feature_maps_{layer_name}.html')
    
    def _plot_linear_features(self, features: torch.Tensor, layer_name: str) -> None:
        """Plot linear layer activations"""
        features = features[0]  # Take first batch
        features = features.cpu().numpy()
        
        plt.figure(figsize=(10, 5))
        plt.plot(features)
        plt.title(f'Activation Values: {layer_name}')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'activations_{layer_name}.png')
        plt.close()
        
        # Create interactive version
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(y=features,
                      mode='lines',
                      name='Activation Values')
        )
        
        fig.update_layout(
            title=f'Activation Values: {layer_name}',
            xaxis_title='Neuron Index',
            yaxis_title='Activation'
        )
        
        fig.write_html(self.save_dir / f'activations_{layer_name}.html')
    
    def visualize_model_summary(self, model: nn.Module) -> None:
        """Create a comprehensive model summary visualization"""
        # Get model summary
        summary = self._get_model_summary(model)
        
        # Save as markdown
        with open(self.save_dir / 'model_summary.md', 'w') as f:
            f.write("# Model Summary\n\n")
            
            f.write("## Layer Information\n")
            f.write("| Layer | Type | Output Shape | Parameters |\n")
            f.write("|-------|------|--------------|------------|\n")
            
            total_params = 0
            for name, layer_info in summary.items():
                f.write(f"| {name} | {layer_info['type']} | {layer_info['output_shape']} | ")
                f.write(f"{layer_info['params']:,} |\n")
                total_params += layer_info['params']
            
            f.write(f"\nTotal Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {self._count_trainable_params(model):,}\n")
    
    def _get_model_summary(self, model: nn.Module) -> Dict:
        """Generate model summary information"""
        summary = {}
        
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)
                
                m_key = f"{class_name}-{module_idx+1}"
                summary[m_key] = {
                    "type": class_name,
                    "output_shape": list(output.size()),
                    "params": sum(p.numel() for p in module.parameters())
                }
            
            if not isinstance(module, nn.Sequential) and \
               not isinstance(module, nn.ModuleList) and \
               not (module == model):
                hooks.append(module.register_forward_hook(hook))
        
        # Register hooks
        hooks = []
        model.apply(register_hook)
        
        # Make a forward pass
        with torch.no_grad():
            model(torch.randn(1, 3, 224, 224))
            
        # Remove these hooks
        for hook in hooks:
            hook.remove()
            
        return summary
    
    def _count_trainable_params(self, model: nn.Module) -> int:
        """Count trainable parameters in the model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 