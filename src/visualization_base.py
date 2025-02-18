import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE
import json
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

class VisualizationBase(ABC):
    """Base class for visualization"""
    
    def __init__(self, save_dir: str = 'assets'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = self._get_class_names()
        
    def _get_class_names(self) -> List[str]:
        """Get class names in English and Persian"""
        return [
            "Healthy (سالم)",
            "White Root Worm (کرم سفید ریشه)",
            "Whitefly (سفیدبالک)",
            "Mealybug (شپشک آردالود)",
            "Water Thief (آبدزدک)",
            "Grain Aphid (شته غلات)",
            "Green Pest (آفت سبز)",
            "Oat Aphid (شته یولاف)",
            "Leafhopper (زنجره)",
            "Rust (زنگ زدگی)",
            "Rot (پوسیدگی)",
            "Wave Spot (لکه موجی)",
            "Mold (کپک)",
            "Wind Damage (بادزدگی)",
            "Powdery Mildew (سفیدک پودری)",
            "Others (سایر)"
        ]
    
    def plot_training_progress(self, history: Dict) -> None:
        """Plot training metrics"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'F1 Score', 'Learning Rate', 'Memory Usage')
        )
        
        # Loss curves
        fig.add_trace(
            go.Scatter(y=history['train_loss'], name="Train Loss"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history['val_loss'], name="Val Loss"),
            row=1, col=1
        )
        
        # F1 Score
        fig.add_trace(
            go.Scatter(y=history['val_f1'], name="F1 Score"),
            row=1, col=2
        )
        
        # Learning Rate
        fig.add_trace(
            go.Scatter(y=history['learning_rates'], name="Learning Rate"),
            row=2, col=1
        )
        
        # Memory Usage
        fig.add_trace(
            go.Scatter(y=history['memory_usage'], name="Memory (GB)"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Training Progress")
        fig.write_html(self.save_dir / 'training_progress.html')
        
        # Save static version
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.title('Loss')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(history['val_f1'])
        plt.title('F1 Score')
        
        plt.subplot(2, 2, 3)
        plt.plot(history['learning_rates'])
        plt.title('Learning Rate')
        
        plt.subplot(2, 2, 4)
        plt.plot(history['memory_usage'])
        plt.title('Memory Usage (GB)')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_progress.png')
        plt.close()
    
    def plot_confusion_matrix(self, y_true: np.ndarray,
                            y_pred: np.ndarray,
                            normalize: bool = True) -> None:
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close()
    
    def plot_class_performance(self, metrics: Dict) -> None:
        """Plot per-class performance metrics"""
        metrics_df = pd.DataFrame(metrics).round(3)
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        sns.barplot(data=metrics_df, x=metrics_df.index, y='precision')
        plt.title('Precision by Class')
        plt.xticks(rotation=45, ha='right')
        
        plt.subplot(1, 2, 2)
        sns.barplot(data=metrics_df, x=metrics_df.index, y='recall')
        plt.title('Recall by Class')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'class_performance.png')
        plt.close()
        
        # Save interactive version
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Precision', 'Recall'))
        
        fig.add_trace(
            go.Bar(x=metrics_df.index, y=metrics_df['precision'],
                  name='Precision'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=metrics_df.index, y=metrics_df['recall'],
                  name='Recall'),
            row=1, col=2
        )
        
        fig.update_layout(height=500, title_text="Class Performance Metrics")
        fig.write_html(self.save_dir / 'class_performance.html')
    
    def create_performance_report(self, history: Dict,
                                class_metrics: Dict,
                                timing_info: Dict) -> None:
        """Generate comprehensive performance report"""
        report = {
            'training': {
                'final_loss': history['val_loss'][-1],
                'best_f1': max(history['val_f1']),
                'epochs': len(history['val_loss']),
                'training_time': timing_info['training_time']
            },
            'inference': {
                'avg_inference_time': timing_info['avg_inference_time'],
                'memory_peak': timing_info['memory_peak']
            },
            'class_performance': class_metrics
        }
        
        # Save as JSON
        with open(self.save_dir / 'performance_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        # Save as Markdown
        with open(self.save_dir / 'performance_report.md', 'w') as f:
            f.write("# Performance Report\n\n")
            
            f.write("## Training Metrics\n")
            f.write(f"- Final Loss: {report['training']['final_loss']:.4f}\n")
            f.write(f"- Best F1 Score: {report['training']['best_f1']:.4f}\n")
            f.write(f"- Epochs: {report['training']['epochs']}\n")
            f.write(f"- Training Time: {report['training']['training_time']:.1f}s\n\n")
            
            f.write("## Inference Performance\n")
            f.write(f"- Average Inference Time: {report['inference']['avg_inference_time']:.3f}s\n")
            f.write(f"- Peak Memory Usage: {report['inference']['memory_peak']:.1f}GB\n\n")
            
            f.write("## Class Performance\n")
            f.write("| Class | Precision | Recall | F1 Score |\n")
            f.write("|-------|-----------|---------|----------|\n")
            for class_name, metrics in zip(self.class_names, class_metrics):
                f.write(f"| {class_name} | {metrics['precision']:.3f} | ")
                f.write(f"{metrics['recall']:.3f} | {metrics['f1']:.3f} |\n")
    
    @abstractmethod
    def plot_model_architecture(self, model: torch.nn.Module) -> None:
        """Plot model architecture"""
        pass
    
    @abstractmethod
    def plot_feature_maps(self, model: torch.nn.Module,
                         sample_input: torch.Tensor) -> None:
        """Plot feature maps from model layers"""
        pass 