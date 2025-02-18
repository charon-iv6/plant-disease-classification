import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import torch

class TrainingVisualizer:
    def __init__(self, save_dir='training_results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'per_class_f1': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
    def update(self, epoch_metrics):
        for key, value in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def plot_training_progress(self):
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0,0].plot(self.history['train_loss'], label='Train Loss')
        axes[0,0].plot(self.history['val_loss'], label='Val Loss')
        axes[0,0].set_title('Loss Progress')
        axes[0,0].legend()
        
        # F1 Score plot
        axes[0,1].plot(self.history['val_f1'], label='Validation F1')
        axes[0,1].set_title('F1 Score Progress')
        axes[0,1].legend()
        
        # Per-class F1 heatmap
        if self.history['per_class_f1']:
            df_f1 = pd.DataFrame(self.history['per_class_f1'])
            sns.heatmap(df_f1.T, ax=axes[1,0], cmap='YlOrRd')
            axes[1,0].set_title('Per-class F1 Scores')
        
        # Learning rate plot
        axes[1,1].plot(self.history['learning_rates'])
        axes[1,1].set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_progress.png')
        plt.close()
    
    def generate_report(self, class_names):
        report = {
            "Training Summary": {
                "Best F1 Score": max(self.history['val_f1']),
                "Best Epoch": np.argmax(self.history['val_f1']) + 1,
                "Final Loss": self.history['val_loss'][-1],
                "Training Time": sum(self.history['epoch_times']),
            },
            "Per-Class Performance": {}
        }
        
        # Add per-class metrics
        last_f1_scores = self.history['per_class_f1'][-1]
        for class_idx, (class_name, f1_score) in enumerate(zip(class_names, last_f1_scores)):
            report["Per-Class Performance"][class_name] = {
                "F1 Score": f1_score,
                "Performance Category": "Good" if f1_score > 0.8 else "Needs Improvement"
            }
        
        # Save report
        with open(self.save_dir / 'training_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        return report

class VisualizationManager:
    def __init__(self, save_dir='assets'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_training_progress(self, history):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0,0].plot(history['train_loss'], label='Train')
        axes[0,0].plot(history['val_loss'], label='Validation')
        axes[0,0].set_title('Loss Progress')
        axes[0,0].legend()
        
        # F1 Score plot
        axes[0,1].plot(history['val_f1'], label='Validation F1')
        axes[0,1].set_title('F1 Score Progress')
        axes[0,1].legend()
        
        # Learning rate plot
        axes[1,0].plot(history['learning_rates'])
        axes[1,0].set_title('Learning Rate Schedule')
        
        # Memory usage plot
        axes[1,1].plot(history['memory_usage'])
        axes[1,1].set_title('Memory Usage (GB)')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_progress.png')
        plt.close()

    def plot_confusion_matrix(self, true_labels, predictions, class_names):
        """Generate confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close()

    def plot_class_performance(self, class_metrics, class_names):
        """Plot per-class performance metrics"""
        metrics_df = pd.DataFrame(class_metrics)
        
        plt.figure(figsize=(15, 6))
        sns.barplot(data=metrics_df.melt(), x='variable', y='value', hue='Class')
        plt.title('Per-class Performance Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'class_performance.png')
        plt.close()

    def plot_lr_schedule(self, scheduler, num_epochs):
        """Visualize learning rate schedule"""
        lrs = []
        for epoch in range(num_epochs):
            lrs.extend(scheduler.get_last_lr())
            scheduler.step()
        
        plt.figure(figsize=(10, 5))
        plt.plot(lrs)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.savefig(self.save_dir / 'lr_schedule.png')
        plt.close()

    def plot_model_architecture(self, model):
        """Visualize model architecture"""
        def count_parameters(m):
            return sum(p.numel() for p in m.parameters())
        
        layers = []
        for name, module in model.named_children():
            layers.append({
                'name': name,
                'params': count_parameters(module),
                'type': module.__class__.__name__
            })
        
        df = pd.DataFrame(layers)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='name', y='params')
        plt.title('Model Architecture - Parameters per Layer')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'model_architecture.png')
        plt.close()

    def create_performance_report(self, history, class_metrics, timing_info):
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