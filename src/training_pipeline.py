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
        
        # Initialize logger first
        self.logger = self._setup_logging()
        
        # Then setup device
        self.device = self._setup_device()
        
        # Initialize model with memory-efficient settings
        self.model = ResNet9(
            num_classes=16,
            dropout=self.config.get('dropout', 0.1)
        ).to(self.device)
        
        if self.config.get('memory_efficient', False):
            # Use memory-efficient settings for Mac
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends, 'mps'):
                # MPS-specific memory optimizations
                if hasattr(torch.mps, 'set_cache_limit'):
                    # Limit MPS cache to 80% of available memory
                    total_memory = psutil.virtual_memory().total
                    torch.mps.set_cache_limit(int(total_memory * 0.8))
                
                self.logger.info("Enabled memory-efficient settings for MPS")
        
        # Initialize EMA model only if not in memory-efficient mode
        if not self.config.get('memory_efficient', False):
            self.ema_model = ModelEMA(self.model)
        else:
            self.ema_model = None
        
        # Initialize optimizer with memory-efficient settings
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Initialize other attributes
        self.memory_stats = []
        self.current_epoch = 0
        self.best_val_f1 = 0
        self.scheduler = None
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler(enabled=self.config.get('mixed_precision', True))
        
        # Set gradient accumulation steps
        self.grad_accum_steps = self.config.get('gradient_accumulation_steps', 1)
        
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
        
        # Memory-efficient data loading settings
        dataloader_kwargs = {
            'batch_size': self.config['batch_size'],
            'num_workers': self.config['num_workers'],
            'pin_memory': self.config.get('pin_memory', False),
            'persistent_workers': False if self.config.get('memory_efficient', False) else True,
            'prefetch_factor': 2 if not self.config.get('memory_efficient', False) else None
        }
        
        train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            **dataloader_kwargs,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            **dataloader_kwargs
        )
        
        return train_loader, val_loader
    
    def monitor_memory(self):
        """Monitor memory usage with less aggressive checks"""
        stats = {}
        
        # CPU Memory
        process = psutil.Process()
        stats['cpu_memory_gb'] = process.memory_info().rss / (1024 * 1024 * 1024)
        
        # GPU Memory (if available)
        if self.device.type == 'cuda':
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
            stats['gpu_cached_gb'] = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
        elif self.device.type == 'mps':
            # MPS (Apple Silicon) doesn't provide memory stats
            stats['gpu_allocated_gb'] = 0
            stats['gpu_cached_gb'] = 0
        
        # Only store if significant change
        if not self.memory_stats or \
           abs(stats['cpu_memory_gb'] - self.memory_stats[-1]['cpu_memory_gb']) > 0.1:
            self.memory_stats.append(stats)
        
        return stats
    
    def clear_gpu_memory(self):
        """Clear GPU memory only when necessary"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
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
        elif torch.backends.mps.is_available():
            # M1/M2 specific optimizations
            torch.mps.empty_cache()
            # Set garbage collection threshold for better memory management
            gc.set_threshold(100, 5, 5)
            
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
        """Single epoch training with memory-efficient processing"""
        self.model.train()
        stats = {
            'train_loss': 0,
            'samples_processed': 0,
            'train_predictions': [],
            'train_targets': []
        }
        
        optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                # Process batch with gradient accumulation
                is_accumulation_step = (batch_idx + 1) % self.grad_accum_steps != 0
                
                with autocast(enabled=self.config.get('mixed_precision', True)):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target) / self.grad_accum_steps
                
                # Scale loss and backward pass
                scaler.scale(loss).backward()
                
                if not is_accumulation_step:
                    # Update weights and zero gradients
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    
                    if scheduler is not None:
                        scheduler.step()
                
                # Update statistics
                stats['train_loss'] += loss.item() * self.grad_accum_steps
                stats['samples_processed'] += len(data)
                
                # Collect predictions and targets
                with torch.no_grad():
                    stats['train_predictions'].extend(output.argmax(dim=1).cpu().numpy())
                    stats['train_targets'].extend(target.cpu().numpy())
                
                # Clear memory for Mac
                if self.config.get('memory_efficient', False):
                    del data, target, output, loss
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    elif hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                
                # Monitor memory periodically
                if batch_idx % 10 == 0:
                    self.monitor_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.error(f"OOM in batch {batch_idx}. Clearing memory and skipping batch.")
                    self.clear_gpu_memory()
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise
        
        return stats
    
    def _validate(self, val_loader, criterion):
        """Memory-efficient validation step"""
        self.model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                with autocast(enabled=self.config.get('mixed_precision', True)):
                    output = self.model(data)
                    loss = criterion(output, target)
                
                val_loss += loss.item()
                val_predictions.extend(output.argmax(dim=1).cpu().numpy())
                val_targets.extend(target.cpu().numpy())
                
                # Clear memory for Mac
                if self.config.get('memory_efficient', False):
                    del data, target, output, loss
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    elif hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
        
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
        """Memory-efficient model saving"""
        try:
            save_path = Path(self.config['save_dir']) / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model state with reduced memory footprint
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'epoch': self.current_epoch,
                'best_val_f1': self.best_val_f1,
            }
            
            # Only save EMA if enabled
            if self.ema_model is not None:
                save_dict['ema_state_dict'] = self.ema_model.shadow
            
            # Save optimizer and scheduler if not in memory-efficient mode
            if not self.config.get('memory_efficient', False):
                save_dict.update({
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
                })
            
            torch.save(save_dict, save_path)
            self.logger.info(f"Model saved to {save_path}")
            
            # Log to MLflow
            mlflow.log_artifact(str(save_path))
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            try:
                # Create emergency backup with minimal state
                emergency_path = Path('models') / f'emergency_{filename}'
                torch.save(self.model.state_dict(), emergency_path)
                self.logger.info(f"Emergency model backup saved to {emergency_path}")
            except Exception as backup_error:
                self.logger.error(f"Emergency backup also failed: {str(backup_error)}")
                raise
    
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