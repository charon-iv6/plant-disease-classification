import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import mlflow
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import time
from datetime import datetime
import psutil
import gc
from typing import Dict

from models import ResNet9, ModelEMA

class TrainingPipeline:
    def __init__(self, train_dataset, val_dataset, config):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = self._setup_device()
        self.logger = self._setup_logging()
        
        # Initialize model
        self.model = ResNet9(
            num_classes=16,
            dropout=self.config.get('dropout', 0.1)
        ).to(self.device)
        
        # Initialize EMA model
        self.ema_model = ModelEMA(self.model)
        
        self.memory_stats = []
        
    def _setup_device(self):
        """Setup compute device with proper error handling"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            self.logger.info("Using Apple Silicon GPU")
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU")
        return device
    
    def _setup_logging(self):
        """Setup logging with proper format and handlers"""
        logger = logging.getLogger('TrainingPipeline')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _get_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        labels = [sample[1] for sample in self.train_dataset]
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        
        weights = total_samples / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights).to(self.device)
    
    def _create_dataloaders(self):
        """Create memory-efficient dataloaders"""
        # Calculate sample weights for balanced sampling
        labels = [sample[1] for sample in self.train_dataset]
        class_counts = np.bincount(labels)
        sample_weights = [1/class_counts[label] for label in labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Memory-efficient data loading
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            prefetch_factor=2,  # Reduce prefetch to save memory
            persistent_workers=True,  # Keep workers alive between epochs
            drop_last=True  # Avoid small last batch
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        return train_loader, val_loader
    
    def monitor_memory(self) -> Dict[str, float]:
        """Monitor and log memory usage"""
        stats = {}
        
        # CPU Memory
        process = psutil.Process()
        stats['cpu_memory_gb'] = process.memory_info().rss / 1024 ** 3
        
        # GPU Memory
        if torch.cuda.is_available():
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated(0) / 1024 ** 3
            stats['gpu_cached_gb'] = torch.cuda.memory_cached(0) / 1024 ** 3
            stats['gpu_max_gb'] = torch.cuda.max_memory_allocated(0) / 1024 ** 3
        
        self.memory_stats.append(stats)
        self.logger.info(f"Memory Usage - CPU: {stats['cpu_memory_gb']:.2f}GB, " + 
                        (f"GPU: {stats['gpu_allocated_gb']:.2f}GB allocated" 
                         if torch.cuda.is_available() else "GPU: N/A"))
        
        return stats
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self.logger.info("Cleared GPU memory cache")
    
    def _optimize_memory(self):
        """Optimize memory usage"""
        if torch.cuda.is_available():
            # Set memory allocator settings
            torch.cuda.set_per_process_memory_fraction(0.8)  # Reserve 20% for system
            torch.cuda.empty_cache()
            
            # Use memory-efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                torch.backends.cuda.enable_mem_efficient_sdp()
            
            # Enable memory-efficient methods
            torch.backends.cudnn.benchmark = True
            
        # Clear unused memory
        gc.collect()
    
    def train(self):
        """Main training loop with memory management"""
        try:
            # Initial memory check
            self.monitor_memory()
            
            self.model = self.model.to(self.device)
            train_loader, val_loader = self._create_dataloaders()
            
            # Setup training components
            criterion = nn.CrossEntropyLoss(weight=self._get_class_weights())
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config['learning_rate'],
                epochs=self.config['epochs'],
                steps_per_epoch=len(train_loader)
            )
            scaler = GradScaler()
            
            # Training loop
            best_val_f1 = 0
            for epoch in range(self.config['epochs']):
                # Training phase
                epoch_stats = self._train_epoch(train_loader, criterion, optimizer, scheduler, scaler)
                
                # Memory monitoring
                mem_stats = self.monitor_memory()
                epoch_stats.update({f"memory_{k}": v for k, v in mem_stats.items()})
                
                # Clear memory if needed
                if mem_stats.get('gpu_allocated_gb', 0) > self.config.get('memory_threshold_gb', 10):
                    self.clear_gpu_memory()
                
                # Calculate metrics
                train_f1 = self._calculate_f1(epoch_stats['train_targets'], epoch_stats['train_predictions'])
                val_f1 = self._calculate_f1(epoch_stats['val_targets'], epoch_stats['val_predictions'])
                
                # Log epoch metrics
                metrics = {
                    'train_loss': epoch_stats['train_loss'] / len(train_loader),
                    'val_loss': epoch_stats['val_loss'] / len(val_loader),
                    'train_f1': train_f1,
                    'val_f1': val_f1,
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                metrics.update(epoch_stats)
                
                self._log_epoch(epoch, metrics)
                
                # Save best model
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    self._save_model('best_model.pth')
                    
                # Generate and save confusion matrix
                if epoch % 5 == 0:
                    self._plot_confusion_matrix(epoch_stats['val_targets'], epoch_stats['val_predictions'], epoch)
            
            # Final evaluation and report
            self._generate_final_report()
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            self.clear_gpu_memory()
            raise
        
        finally:
            # Final memory cleanup
            self.clear_gpu_memory()
    
    def _train_epoch(self, train_loader, criterion, optimizer, scheduler, scaler):
        """Single epoch training with memory monitoring"""
        self.model.train()
        stats = {'train_loss': 0, 'samples_processed': 0}
        
        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                # Process batch
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                with autocast():
                    output = self.model(data)
                    loss = criterion(output, target)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                stats['train_loss'] += loss.item()
                stats['samples_processed'] += len(data)
                
                # Monitor memory periodically
                if batch_idx % 10 == 0:
                    self.monitor_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.error(f"OOM in batch {batch_idx}. Clearing memory and skipping batch.")
                    self.clear_gpu_memory()
                    continue
                raise
        
        return stats
    
    def _validate(self, val_loader, criterion):
        """Validation step with proper error handling"""
        self.model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                val_predictions.extend(output.argmax(dim=1).cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        return val_loss, val_predictions, val_targets
    
    def _calculate_f1(self, targets, predictions):
        """Calculate macro-averaged F1 score"""
        from sklearn.metrics import f1_score
        return f1_score(targets, predictions, average='macro')
    
    def _log_epoch(self, epoch, metrics):
        """Log epoch metrics to file and MLflow"""
        # Log to file
        self.logger.info(f"Epoch {epoch} metrics: {metrics}")
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"epoch_{epoch}"):
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
    
    def _save_model(self, filename):
        """Save model with proper error handling"""
        try:
            torch.save(self.model.state_dict(), filename)
            self.logger.info(f"Model saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
    
    def _plot_confusion_matrix(self, targets, predictions, epoch):
        """Generate and save confusion matrix plot"""
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_epoch_{epoch}.png')
        plt.close()
    
    def _generate_final_report(self):
        """Generate comprehensive training report"""
        report_path = Path('training_report.md')
        with report_path.open('w') as f:
            f.write("# Training Report\n\n")
            f.write(f"## Training Configuration\n")
            for key, value in self.config.items():
                f.write(f"- {key}: {value}\n")
            
            f.write("\n## Model Architecture\n")
            f.write(f"```\n{str(self.model)}\n```\n")
            
        self.logger.info(f"Training report generated at {report_path}") 