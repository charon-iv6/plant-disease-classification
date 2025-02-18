import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import time
from tqdm import tqdm
import albumentations as A
import cv2
import copy
import mlflow
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
from functools import wraps
from datetime import datetime
import psutil
import platform
from pathlib import Path
import warnings
import json
from dataset_utils import verify_dataset, prepare_test_dir, verify_submission_format
import sys
from visualization import VisualizationManager

# Disable albumentations warning
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

class ResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=16, dropout=0.1):
        super().__init__()
        
        # Initial layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # First block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        
        # Second block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Third block
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.res2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.maxpool = nn.AdaptiveAvgPool2d((1, 1))  # This ensures fixed output size
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Initial conv
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        
        # First block
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = out + self.res1(out)
        
        # Second block
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.pool(out)
        
        # Third block
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.pool(out)
        out = out + self.res2(out)
        
        # Global pooling and classifier
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.dropout(out)
        out = self.classifier(out)
        
        return out

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.class_names = {
            '0': 'Healthy', '1': 'White_Root_Worm', '2': 'Whitefly',
            '3': 'Mealybug', '4': 'Water_Thief', '5': 'Grain_Aphid',
            '6': 'Green_Pest', '7': 'Oat_Aphid', '8': 'Leafhopper',
            '9': 'Rust', '10': 'Rot', '11': 'Wave_Spot',
            '12': 'Mold', '13': 'Wind_Damage', '14': 'Powdery_Mildew',
            '15': 'Others'
        }
        
        # Update train augmentation
        self.train_aug = A.Compose([
            A.Resize(224, 224),  # Fixed size
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.2),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Update validation augmentation
        self.val_aug = A.Compose([
            A.Resize(224, 224),  # Fixed size
            A.CenterCrop(224, 224),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def analyze_and_prepare_data(self, augment_threshold=100):
        """Analyze dataset and prepare it for training"""
        print("Analyzing dataset...")
        class_counts = self._get_class_counts()
        self._visualize_distribution(class_counts)
        
        # Augment small classes
        if augment_threshold > 0:
            self._augment_small_classes(class_counts, augment_threshold)
        
        return self._create_datasets()
    
    def _get_class_counts(self):
        """Count images in each class"""
        class_counts = {}
        for class_id in self.class_names.keys():
            class_path = os.path.join(self.data_path, class_id)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                class_counts[class_id] = len(images)
                print(f"Class {self.class_names[class_id]}: {len(images)} images")
        return class_counts
    
    def _visualize_distribution(self, class_counts):
        """Visualize class distribution"""
        plt.figure(figsize=(15, 6))
        classes = [self.class_names[str(i)] for i in range(len(self.class_names))]
        counts = [class_counts[str(i)] for i in range(len(self.class_names))]
        
        plt.bar(classes, counts)
        plt.xticks(rotation=45, ha='right')
        plt.title('Class Distribution')
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.tight_layout()
        plt.savefig('class_distribution.png')
        plt.close()
    
    def _augment_small_classes(self, class_counts, threshold):
        """Augment classes with fewer samples than threshold"""
        print("\nAugmenting small classes...")
        for class_id, count in class_counts.items():
            if count < threshold:
                class_path = os.path.join(self.data_path, class_id)
                num_augment = threshold - count
                print(f"Generating {num_augment} augmented images for {self.class_names[class_id]}")
                
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for i in tqdm(range(num_augment)):
                    # Random image selection
                    img_name = np.random.choice(images)
                    img_path = os.path.join(class_path, img_name)
                    
                    # Augment and save
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    augmented = self.train_aug(image=image)['image']
                    
                    aug_name = f"aug_{i}_{img_name}"
                    aug_path = os.path.join(class_path, aug_name)
                    cv2.imwrite(aug_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

    def _create_datasets(self):
        """Create train and validation datasets"""
        print("\nCreating datasets...")
        
        # Get all image paths and labels
        image_paths = []
        labels = []
        
        for class_id in self.class_names.keys():
            class_path = os.path.join(self.data_path, class_id)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(class_path, img_name))
                        labels.append(int(class_id))
        
        # Split into train and validation
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=labels
        )
        
        # Create datasets
        train_dataset = PlantDataset(
            image_paths=train_paths,
            labels=train_labels,
            transform=self.train_aug
        )
        
        val_dataset = PlantDataset(
            image_paths=val_paths,
            labels=val_labels,
            transform=self.val_aug
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        return train_dataset, val_dataset

class PlantDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image, self.labels[idx]

class MixupTransform:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, labels = batch
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        label_a, label_b = labels, labels[index]
        return mixed_images, label_a, label_b, lam

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay
    
    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        # Create log directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('PlantDiseaseClassification')
        self.logger.setLevel(logging.INFO)
        
        # Create timestamp for unique log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'training_{timestamp}.log'
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize metrics storage
        self.metrics = {
            'epoch_times': [],
            'batch_times': [],
            'gpu_usage': [],
            'memory_usage': []
        }
    
    def log_system_info(self):
        """Log system information at start of training"""
        gpu_info = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else "No GPU"
        system_info = {
            'CPU': psutil.cpu_count(),
            'RAM': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            'GPU': str(gpu_info),
            'Python': platform.python_version(),
            'PyTorch': torch.__version__,
            'CUDA': torch.version.cuda if torch.cuda.is_available() else "N/A"
        }
        self.logger.info("System Information:")
        for k, v in system_info.items():
            self.logger.info(f"{k}: {v}")
    
    def log_epoch(self, epoch, metrics):
        """Log epoch metrics"""
        self.logger.info(f"\nEpoch {epoch} Results:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
    
    def log_batch(self, epoch, batch, loss, lr):
        """Log batch metrics"""
        self.logger.debug(
            f"Epoch {epoch}, Batch {batch}, Loss: {loss:.4f}, LR: {lr:.6f}"
        )
    
    def log_memory(self):
        """Log memory usage"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**2)
            self.metrics['gpu_usage'].append(gpu_memory)
            self.logger.debug(f"GPU Memory: {gpu_memory:.2f} MB")
        
        ram_usage = psutil.Process().memory_info().rss / (1024**2)
        self.metrics['memory_usage'].append(ram_usage)
        self.logger.debug(f"RAM Usage: {ram_usage:.2f} MB")

def benchmark(logger):
    """Decorator for benchmarking functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.log_memory()
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            duration = end_time - start_time
            logger.logger.info(f"{func.__name__} took {duration:.2f} seconds")
            logger.log_memory()
            
            return result
        return wrapper
    return decorator

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate per-class F1 scores
    f1_scores = {}
    for i in range(16):
        class_f1 = f1_score(
            [1 if t == i else 0 for t in all_targets],
            [1 if p == i else 0 for p in all_preds],
            average='binary'
        )
        f1_scores[f'class_{i}'] = class_f1
    
    # Calculate macro F1
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    
    return val_loss / len(val_loader), macro_f1, f1_scores

def plot_f1_scores(f1_scores):
    plt.figure(figsize=(15, 6))
    classes = list(f1_scores.keys())
    scores = list(f1_scores.values())
    
    plt.bar(classes, scores)
    plt.axhline(y=0.8, color='r', linestyle='--', label='Good Performance Threshold')
    plt.xticks(rotation=45, ha='right')
    plt.title('F1 Scores by Class')
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.tight_layout()
    plt.savefig('f1_scores.png')
    plt.close()

class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class SWAModel(torch.optim.swa_utils.AveragedModel):
    def forward(self, x):
        return super(SWAModel, self).forward(x)

def train_with_swa(train_loader, val_loader, device, class_weights, num_epochs=30):
    model = ResNet9(num_classes=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    swa_model = SWAModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(
        optimizer, swa_lr=0.05
    )
    
    # Start SWA from 75% of training
    swa_start = int(0.75 * num_epochs)
    
    for epoch in range(num_epochs):
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            # Regular training
            train_epoch(model, train_loader, optimizer, device)
    
    # Update batch norm statistics
    torch.optim.swa_utils.update_bn(train_loader, swa_model)
    
    return swa_model

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

def train_model(train_loader, val_loader, device, class_weights, num_epochs=30):
    logger = TrainingLogger()
    logger.log_system_info()
    
    # Use Focal Loss instead of CrossEntropyLoss
    criterion = BalancedFocalLoss()
    
    # Increase model capacity slightly
    model = ResNet9(num_classes=16, dropout=0.2).to(device)
    
    # Use stronger augmentations for rare classes
    train_transform = A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.HueSaturationValue(p=1)
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
            A.MotionBlur(p=1)
        ], p=0.2),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Implement learning rate finder
    def find_lr(model, train_loader, optimizer, criterion, device):
        num_batches = len(train_loader)
        log_lrs = torch.linspace(-8, 1, num_batches)
        losses = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.param_groups[0]['lr'] = torch.exp(log_lrs[batch_idx])
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
        return log_lrs, losses
    
    # Use learning rate finder before training
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    log_lrs, losses = find_lr(model, train_loader, optimizer, criterion, device)
    optimal_lr = torch.exp(log_lrs[np.argmin(losses)])
    
    # Update optimizer with found learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimal_lr, weight_decay=0.01)
    
    ema = ModelEMA(model)
    early_stopping = EarlyStopping(patience=5)
    mixup = MixupTransform(alpha=0.2)
    
    mlflow.set_experiment("plant_disease_classification")
    
    visualizer = TrainingVisualizer()
    
    best_f1 = 0.0
    accumulation_steps = 4  # Gradient accumulation steps
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
                
                # Apply Mixup
                if np.random.random() > 0.5:
                    mixed_data, target_a, target_b, lam = mixup((data, target))
                    data = mixed_data
                
                optimizer.zero_grad()
                
                output = model(data)
                
                loss = criterion(output, target)
                
                loss = loss / accumulation_steps
                loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Update EMA model
                ema.update(model)
                
                total_loss += loss.item() * accumulation_steps
                
                pbar.set_postfix({'loss': loss.item() * accumulation_steps})
        
        # Validation
        val_loss, macro_f1, per_class_f1 = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - time.time()
        
        # Update visualizer
        epoch_metrics = {
            'train_loss': total_loss / len(train_loader),
            'val_loss': val_loss,
            'val_f1': macro_f1,
            'per_class_f1': per_class_f1,
            'learning_rates': optimizer.param_groups[0]['lr'],
            'epoch_times': epoch_time
        }
        visualizer.update(epoch_metrics)
        visualizer.plot_training_progress()
        
        # MLflow logging
        mlflow.log_metrics(epoch_metrics, step=epoch)
        
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {total_loss / len(train_loader):.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val F1: {macro_f1:.4f}')
        
        # Save best model
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict': ema.model.state_dict(),
                'val_f1': macro_f1,
            }, 'best_model.pth')
        
        # Early stopping
        early_stopping(val_loss / len(val_loader))
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Generate final report
    report = visualizer.generate_report(class_names=preprocessor.class_names)
    print("\nTraining Report:")
    print(json.dumps(report, indent=2))
    
    return model

def predict_with_tta(model, image, num_augments=5):
    augmentations = [
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomRotate90(p=1.0),
        A.ColorJitter(brightness=0.1, contrast=0.1, p=1.0),
        A.GaussNoise(p=1.0)
    ]
    
    predictions = []
    with torch.no_grad():
        pred = model(image).softmax(dim=1)
        predictions.append(pred)
    
    for aug in augmentations[:num_augments]:
        aug_image = aug(image=image.cpu().numpy())['image']
        aug_tensor = torch.from_numpy(aug_image).to(image.device)
        with torch.no_grad():
            aug_pred = model(aug_tensor).softmax(dim=1)
            predictions.append(aug_pred)
    
    weights = [1.0] + [0.8] * num_augments
    final_pred = torch.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        final_pred += pred * weight
    
    return final_pred / sum(weights)

@benchmark(TrainingLogger())
def create_submissions(model, test_dir, device):
    submissions = []
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    for sub_id in range(1, 4):
        predictions = []
        confidences = []
        
        for img_name in sorted(os.listdir(test_dir)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_id = int(img_name.split('.')[0].lstrip('0'))
                image = Image.open(os.path.join(test_dir, img_name)).convert('RGB')
                tensor = transform(image).unsqueeze(0).to(device)
                
                if sub_id == 1:
                    with torch.no_grad():
                        output = model(tensor)
                        pred = output.argmax(dim=1).item()
                        conf = torch.softmax(output, dim=1).max().item()
                    
                    if conf < 0.8:
                        pred = predict_with_tta(model, tensor).argmax(dim=1).item()
                
                elif sub_id == 2:
                    pred = predict_with_tta(model, tensor, num_augments=7).argmax(dim=1).item()
                    conf = 1.0  # TTA confidence
                
                else:
                    with torch.no_grad():
                        output = model(tensor)
                        conf = torch.softmax(output, dim=1).max().item()
                        pred = output.argmax(dim=1).item() if conf > 0.6 else 15
                
                predictions.append(pred)
                confidences.append(conf)
        
        df = pd.DataFrame({
            'ID': range(len(predictions)),
            'class': predictions,
            'confidence': confidences
        })
        df.to_csv(f'd_jabbari_7727_{sub_id}.csv', index=False)
        submissions.append(df)
    
    return submissions

class FastInferenceModel:
    def __init__(self, model_path, device):
        self.device = device
        self.model = ResNet9(num_classes=16).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Optimize for M1 Mac
        if device == 'mps':
            self.model = self.model.to('mps')
        
        # Cache the transforms
        self.transform_cache = {}
    
    @torch.no_grad()
    def predict_batch(self, image_paths, batch_size=32):
        """Batch prediction for faster inference"""
        predictions = []
        confidences = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for img_path in batch_paths:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                transformed = self.val_transform(image=image)
                batch_images.append(transformed['image'])
            
            batch_tensor = torch.stack([torch.from_numpy(img).permute(2, 0, 1) 
                                      for img in batch_images]).float().to(self.device)
            
            output = self.model(batch_tensor)
            confidence, pred = torch.max(torch.softmax(output, dim=1), dim=1)
            
            predictions.extend(pred.cpu().numpy())
            confidences.extend(confidence.cpu().numpy())
        
        return predictions, confidences

    @torch.no_grad()
    def predict_single(self, image_path):
        # Read and transform image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.val_transform(image=image)
        image_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Basic prediction
        output = self.model(image_tensor)
        confidence = torch.softmax(output, dim=1).max().item()
        
        # If confidence is low, use TTA
        if confidence < 0.8:
            pred = self.predict_with_tta(image)
        else:
            pred = output.argmax(dim=1).item()
        
        return pred, confidence
    
    @torch.no_grad()
    def predict_with_tta(self, image):
        transforms = [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
            A.Transpose(p=1.0)
        ]
        
        predictions = []
        # Original prediction
        transformed = self.val_transform(image=image)
        image_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        pred = self.model(image_tensor)
        predictions.append(pred)
        
        # TTA predictions
        for transform in transforms:
            aug = A.Compose([transform, self.val_transform])
            transformed = aug(image=image)
            image_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float()
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            pred = self.model(image_tensor)
            predictions.append(pred)
        
        # Average predictions
        final_pred = torch.stack(predictions).mean(0)
        return final_pred.argmax(dim=1).item()

def create_challenge_submission(model_path, test_dir, output_path, submission_number):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    inference_model = FastInferenceModel(model_path, device)
    
    # Get all image paths first
    image_paths = []
    image_ids = []
    for img_name in sorted(os.listdir(test_dir)):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_id = int(img_name.split('.')[0].lstrip('0'))
            img_path = os.path.join(test_dir, img_name)
            image_paths.append(img_path)
            image_ids.append(img_id)
    
    # Batch process images
    batch_size = 32
    predictions = []
    
    if submission_number == 1:
        # Conservative strategy
        preds, confs = inference_model.predict_batch(image_paths, batch_size)
        for pred, conf in zip(preds, confs):
            if conf < 0.8:
                pred = inference_model.predict_with_tta(img_path)
            predictions.append(pred)
    
    elif submission_number == 2:
        # Full TTA for all images
        for img_path in tqdm(image_paths):
            pred = inference_model.predict_with_tta(img_path)
            predictions.append(pred)
    
    else:
        # Balanced approach
        preds, confs = inference_model.predict_batch(image_paths, batch_size)
        predictions = [pred if conf > 0.6 else 15 for pred, conf in zip(preds, confs)]
    
    # Create submission DataFrame
    df = pd.DataFrame({
        'ID': image_ids,
        'class': predictions
    })
    
    output_file = f'd_jabbari_7727_{submission_number}.csv'
    df.to_csv(output_file, index=False)
    
    # Verify submission format
    if verify_submission_format(output_file):
        print(f"Submission {submission_number} created successfully")
    else:
        print(f"Warning: Submission {submission_number} may have format issues")

def challenge_mode(test_dir, model_path='best_model.pth'):
    """
    Function to run during the challenge evaluation
    """
    try:
        # Create three different submissions
        for submission_number in range(1, 4):
            output_file = f'd_jabbari_7727_{submission_number}.csv'
            create_challenge_submission(
                model_path=model_path,
                test_dir=test_dir,
                output_path=output_file,
                submission_number=submission_number
            )
        print("All submissions created successfully!")
        
    except Exception as e:
        print(f"Error during challenge mode: {str(e)}")
        # Create emergency submission with class 0 for all images
        emergency_submission = []
        for img_name in os.listdir(test_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_id = int(img_name.split('.')[0].lstrip('0'))
                emergency_submission.append({'ID': img_id, 'class': 0})
        
        df = pd.DataFrame(emergency_submission)
        df.to_csv('d_jabbari_7727_emergency.csv', index=False)
        print("Emergency submission created")

def create_training_submission(model, train_dir, device, output_prefix='train'):
    """Create submission files for training data"""
    inference_model = FastInferenceModel(model, device)
    
    for strategy in range(1, 4):
        predictions = []
        print(f"Creating training submission {strategy}/3...")
        
        for class_id in range(16):  # For each class directory
            class_dir = os.path.join(train_dir, str(class_id))
            if not os.path.exists(class_dir):
                continue
                
            for img_name in tqdm(sorted(os.listdir(class_dir))):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_id = int(img_name.split('.')[0].lstrip('0'))
                    img_path = os.path.join(class_dir, img_name)
                    
                    # Different strategies for different submissions
                    if strategy == 1:
                        # Conservative prediction
                        pred, conf = inference_model.predict_single(img_path)
                        if conf < 0.8:
                            pred = inference_model.predict_with_tta(img_path)
                    elif strategy == 2:
                        # Aggressive TTA
                        pred = inference_model.predict_with_tta(img_path)
                    else:
                        # Balanced approach
                        pred, conf = inference_model.predict_single(img_path)
                        if conf < 0.6:
                            pred = 15
                    
                    predictions.append({
                        'ID': img_id,
                        'class': pred,
                        'true_class': class_id
                    })
        
        # Create and save submission
        df = pd.DataFrame(predictions)
        
        # Calculate accuracy and F1 score
        accuracy = (df['class'] == df['true_class']).mean()
        f1 = f1_score(df['true_class'], df['class'], average='macro')
        
        print(f"Strategy {strategy} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Save submission without true_class column
        submission_df = df[['ID', 'class']]
        output_file = f'd_jabbari_7727_{output_prefix}_{strategy}.csv'
        submission_df.to_csv(output_file, index=False)
        
        # Save detailed results for analysis
        analysis_file = f'd_jabbari_7727_{output_prefix}_{strategy}_analysis.csv'
        df.to_csv(analysis_file, index=False)

@benchmark(TrainingLogger())
def main():
    logger = TrainingLogger()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    data_path = "/Users/dani/Documents/AI/vision_plants"
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(data_path)
    
    # Analyze and prepare data
    train_dataset, val_dataset = preprocessor.analyze_and_prepare_data(augment_threshold=100)
    
    # Calculate class weights
    class_counts = np.bincount(train_dataset.labels)
    total_samples = len(train_dataset.labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights)
    
    # Create weighted sampler
    train_sampler = WeightedRandomSampler(
        weights=[class_weights[label] for label in train_dataset.labels],
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,  # New: Prefetch factor
        persistent_workers=True  # New: Persistent workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print("Dataset sizes:")
    print(f"Training: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    
    # Train model
    model = train_model(train_loader, val_loader, device, class_weights)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_weights': class_weights,
    }, 'final_model.pth')
    
    print("\nCreating training data submissions...")
    create_training_submission(
        model=model,
        train_dir="/Users/dani/Documents/AI/vision_plants",
        device=device,
        output_prefix='train'
    )
    
    print("\nCreating test data submissions...")
    create_submissions(model, "/Users/dani/Documents/AI/vision_plants/test_images", device)

    # Initialize visualizer
    visualizer = VisualizationManager()

    # During training
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_f1': f1_scores,
        'learning_rates': lr_history,
        'memory_usage': memory_usage
    }
    visualizer.plot_training_progress(history)

    # After training
    visualizer.plot_confusion_matrix(y_true, y_pred, class_names)
    visualizer.plot_class_performance(class_metrics, class_names)
    visualizer.plot_lr_schedule(scheduler, num_epochs)
    visualizer.plot_model_architecture(model)

    # Generate final report
    timing_info = {
        'training_time': total_training_time,
        'avg_inference_time': avg_inference_time,
        'memory_peak': max_memory_usage
    }
    visualizer.create_performance_report(history, class_metrics, timing_info)

if __name__ == "__main__":
    try:
        data_path = "/Users/dani/Documents/AI/vision_plants"
        
        if os.getenv('CHALLENGE_MODE'):
            if not os.path.exists('best_model.pth'):
                print("Error: best_model.pth not found. Please run training first.")
                sys.exit(1)
            
            test_dir = prepare_test_dir(data_path)
            challenge_mode(
                test_dir=test_dir,
                model_path="best_model.pth"
            )
        else:
            # Use flexible verification (allows augmented images)
            if verify_dataset(data_path, strict=False):
                main()
            else:
                print("Dataset verification failed! Some classes have fewer images than required.")
                sys.exit(1)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        if os.getenv('CHALLENGE_MODE'):
            print("Creating emergency submission...")
            emergency_df = pd.DataFrame({
                'ID': range(500),
                'class': [0] * 500
            })
            emergency_df.to_csv('d_jabbari_7727_emergency.csv', index=False)