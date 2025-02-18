import torch
import torch.nn as nn
import albumentations as A
import numpy as np
from pathlib import Path
import time
import logging
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import queue
import cv2
from abc import ABC, abstractmethod

class InferenceBase(ABC):
    """Base class for inference optimization"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._setup_device()
        self.logger = self._setup_logger()
        self.image_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue()
        
    def _setup_device(self) -> torch.device:
        """Setup compute device with proper error handling"""
        if torch.cuda.is_available() and self.config['hardware']['use_cuda']:
            return torch.device('cuda')
        elif torch.backends.mps.is_available() and self.config['hardware']['use_mps']:
            return torch.device('mps')
        return torch.device('cpu')
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f'{self.__class__.__name__.lower()}.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        return logger
    
    @abstractmethod
    def _get_transforms(self) -> A.Compose:
        """Get image transforms"""
        pass
    
    @abstractmethod
    def _get_tta_transforms(self) -> List[A.Compose]:
        """Get test-time augmentation transforms"""
        pass
    
    def _preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Preprocess single image"""
        try:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = self._get_transforms()(image=image)['image']
            return torch.from_numpy(transformed.transpose(2, 0, 1))
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return None
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load and optimize model"""
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        
        if self.device.type == 'cuda':
            model = model.half()  # FP16 for faster inference
            
        # JIT compile for faster inference
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        if self.device.type == 'cuda':
            dummy_input = dummy_input.half()
        model = torch.jit.trace(model, dummy_input)
        
        return model
    
    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Basic batch prediction"""
        if self.device.type == 'cuda':
            images = images.half()
        return model(images.to(self.device))
    
    @torch.no_grad()
    def predict_with_tta(self, image: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Prediction with test-time augmentation"""
        predictions = []
        
        # Original prediction
        predictions.append(self.predict_batch(image.unsqueeze(0), model))
        
        # TTA predictions
        for transform in self._get_tta_transforms():
            aug_image = transform(image=image.cpu().numpy())['image']
            aug_tensor = torch.from_numpy(aug_image).unsqueeze(0)
            predictions.append(self.predict_batch(aug_tensor, model))
        
        return torch.stack(predictions).mean(0)
    
    def process_dataset(self, model: nn.Module, dataset_path: Path,
                       batch_size: int = 32, time_limit: int = 1800,
                       use_tta: bool = False) -> Dict[int, int]:
        """Process entire dataset with time management"""
        start_time = time.time()
        results = {}
        
        # Get all image paths
        image_paths = sorted(list(dataset_path.glob('*.jpg')) +
                           list(dataset_path.glob('*.jpeg')) +
                           list(dataset_path.glob('*.png')))
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            # Check time limit
            if time.time() - start_time > time_limit:
                self.logger.warning("Time limit reached!")
                break
                
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_indices = []
            
            # Preprocess batch
            for path in batch_paths:
                img = self._preprocess_image(str(path))
                if img is not None:
                    batch_images.append(img)
                    batch_indices.append(int(path.stem))
            
            if not batch_images:
                continue
                
            # Make predictions
            batch_tensor = torch.stack(batch_images)
            if use_tta:
                predictions = [self.predict_with_tta(img, model) for img in batch_tensor]
                predictions = torch.stack(predictions)
            else:
                predictions = self.predict_batch(batch_tensor, model)
            
            # Store results
            for idx, pred in zip(batch_indices, predictions.argmax(dim=1).cpu().numpy()):
                results[idx] = int(pred)
        
        return results
    
    def validate_timing(self, model: nn.Module, batch_size: int,
                       use_tta: bool = False) -> Tuple[float, bool]:
        """Validate inference timing"""
        dummy_batch = torch.randn(batch_size, 3, 224, 224)
        
        # Run multiple trials
        times = []
        for _ in range(5):
            start_time = time.time()
            if use_tta:
                _ = [self.predict_with_tta(img, model) for img in dummy_batch]
            else:
                _ = self.predict_batch(dummy_batch, model)
            times.append(time.time() - start_time)
        
        avg_time = np.mean(times)
        estimated_total = (500 / batch_size) * avg_time
        
        return estimated_total, estimated_total < self.config['time_limit'] 