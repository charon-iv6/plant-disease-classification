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

class AdvancedVisualizationManager:
    def __init__(self, save_dir='assets/advanced'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_interactive_training_progress(self, history):
        """Interactive training progress visualization"""
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
        fig.write_html(self.save_dir / 'interactive_training_progress.html')
    
    def plot_feature_space(self, features, labels, class_names):
        """t-SNE visualization of feature space"""
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=labels, cmap='tab20')
        plt.colorbar(scatter)
        plt.title('t-SNE Feature Space Visualization')
        plt.savefig(self.save_dir / 'feature_space.png')
        plt.close()
        
        # Interactive version
        fig = px.scatter(
            x=features_2d[:, 0], y=features_2d[:, 1],
            color=[class_names[l] for l in labels],
            title='Interactive Feature Space (t-SNE)'
        )
        fig.write_html(self.save_dir / 'interactive_feature_space.html')
    
    def plot_roc_curves(self, y_true, y_scores, class_names):
        """ROC curves for each class"""
        plt.figure(figsize=(12, 8))
        
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_scores[:, i])
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc_score:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'roc_curves.png')
        plt.close()
    
    def plot_prediction_confidence(self, confidences, predictions, true_labels):
        """Analyze prediction confidence distribution"""
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Confidence Distribution',
                                         'Confidence vs Accuracy'))
        
        # Confidence distribution
        fig.add_trace(
            go.Histogram(x=confidences, name="Confidence"),
            row=1, col=1
        )
        
        # Confidence vs Accuracy
        conf_bins = np.linspace(0, 1, 11)
        acc_by_conf = []
        for i in range(len(conf_bins)-1):
            mask = (confidences >= conf_bins[i]) & (confidences < conf_bins[i+1])
            if mask.any():
                acc = (predictions[mask] == true_labels[mask]).mean()
                acc_by_conf.append(acc)
            else:
                acc_by_conf.append(0)
        
        fig.add_trace(
            go.Scatter(x=conf_bins[:-1], y=acc_by_conf, name="Accuracy"),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text="Prediction Confidence Analysis")
        fig.write_html(self.save_dir / 'prediction_confidence.html')
    
    def plot_error_analysis(self, predictions, true_labels, images, class_names):
        """Analyze and visualize prediction errors"""
        errors = predictions != true_labels
        error_indices = np.where(errors)[0]
        
        if len(error_indices) > 0:
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            axes = axes.ravel()
            
            for i, idx in enumerate(error_indices[:9]):
                axes[i].imshow(images[idx])
                axes[i].set_title(f'True: {class_names[true_labels[idx]]}\n'
                                f'Pred: {class_names[predictions[idx]]}')
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.save_dir / 'error_examples.png')
            plt.close()
    
    def create_detailed_report(self, metrics, class_names):
        """Generate detailed performance report with visualizations"""
        report = {
            'overall_metrics': {
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1'],
                'weighted_f1': metrics['weighted_f1']
            },
            'class_metrics': {}
        }
        
        for i, name in enumerate(class_names):
            report['class_metrics'][name] = {
                'precision': metrics['precision'][i],
                'recall': metrics['recall'][i],
                'f1': metrics['f1'][i],
                'support': metrics['support'][i]
            }
        
        # Save as JSON
        import json
        with open(self.save_dir / 'detailed_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        return report 