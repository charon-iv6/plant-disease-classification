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

class AdvancedInference:
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._setup_device()
        self.logger = self._setup_logger()
        
        # Cache transforms for efficiency
        self.transform_cache = {}
        self.val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.error_count = 0
        self.max_errors = 5  # Maximum number of errors before falling back to safe mode
        self.is_safe_mode = False
        self.time_buffer = 0.9  # Use 90% of available time, leave 10% buffer
        
    def _setup_device(self):
        if torch.cuda.is_available() and self.config['hardware']['use_cuda']:
            return torch.device('cuda')
        elif torch.backends.mps.is_available() and self.config['hardware']['use_mps']:
            return torch.device('mps')
        return torch.device('cpu')
    
    def _setup_logger(self):
        logger = logging.getLogger('AdvancedInference')
        logger.setLevel(logging.INFO)
        return logger

    def _get_tta_transforms(self):
        """Get test-time augmentation transforms"""
        if 'tta_transforms' not in self.transform_cache:
            self.transform_cache['tta_transforms'] = [
                A.Compose([A.HorizontalFlip(p=1)]),
                A.Compose([A.VerticalFlip(p=1)]),
                A.Compose([A.RandomRotate90(p=1)]),
                A.Compose([A.HorizontalFlip(p=1), A.VerticalFlip(p=1)]),
                A.Compose([A.ColorJitter(brightness=0.1, contrast=0.1, p=1.0)]),
                A.Compose([A.GaussNoise(p=1.0)])
            ]
        return self.transform_cache['tta_transforms']

    @torch.no_grad()
    def predict_single(self, model: nn.Module, image_path: str) -> Tuple[int, float]:
        """Predict single image with confidence-based TTA"""
        # Read and transform image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.val_transform(image=image)
        image_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Basic prediction
        output = model(image_tensor)
        confidence = torch.softmax(output, dim=1).max().item()
        
        # If confidence is low, use TTA
        if confidence < self.config.get('confidence_threshold', 0.8):
            pred = self.apply_tta(model, image_tensor).argmax(dim=1).item()
        else:
            pred = output.argmax(dim=1).item()
        
        return pred, confidence

    @torch.no_grad()
    def apply_tta(self, model: nn.Module, image: torch.Tensor) -> torch.Tensor:
        """Apply test-time augmentation with weighted averaging"""
        predictions = []
        weights = [1.0] + [0.8] * len(self._get_tta_transforms())
        
        # Original prediction
        pred = model(image)
        predictions.append(pred)
        
        # TTA predictions
        for transform in self._get_tta_transforms():
            aug_image = transform(image=image.cpu().numpy())['image']
            aug_image = torch.from_numpy(aug_image).to(self.device)
            pred = model(aug_image)
            predictions.append(pred)
        
        # Weighted average of predictions
        final_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            final_pred += pred * weight
        
        return final_pred / sum(weights)

    def ensemble_predictions(self, models: List[nn.Module], image: torch.Tensor, 
                           weights: Optional[List[float]] = None) -> torch.Tensor:
        """Get weighted ensemble predictions"""
        if weights is None:
            weights = [1/len(models)] * len(models)
        
        predictions = []
        for model, weight in zip(models, weights):
            with torch.no_grad():
                pred = model(image) * weight
                predictions.append(pred)
        
        return torch.stack(predictions).sum(0)

    def _load_model_checkpoints(self, checkpoint_paths: List[str]) -> List[nn.Module]:
        """Load multiple model checkpoints"""
        models = []
        for path in checkpoint_paths:
            try:
                model = torch.load(path, map_location=self.device)
                model.eval()
                models.append(model)
                self.logger.info(f"Loaded model checkpoint: {path}")
            except Exception as e:
                self.logger.error(f"Error loading model {path}: {str(e)}")
        return models

    def _time_execution(self, func, *args, **kwargs):
        """Time the execution of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time

    def validate_time_constraint(self, model: nn.Module, sample_batch: torch.Tensor, 
                               strategy: Dict) -> bool:
        """Validate if inference strategy meets time constraint"""
        batch_size = strategy['batch_size']
        n_samples = 10  # Number of test runs
        total_time = 0
        
        for _ in range(n_samples):
            if strategy['strategy'] == 'base':
                _, t = self._time_execution(model, sample_batch)
            elif strategy['strategy'] == 'ensemble':
                _, t = self._time_execution(self.ensemble_predictions, 
                                          [model]*3, sample_batch,
                                          strategy.get('ensemble_weights'))
            else:  # tta
                _, t = self._time_execution(self.apply_tta, model, sample_batch)
            total_time += t
        
        avg_time = total_time / n_samples
        estimated_total = (500 / batch_size) * avg_time  # Estimate for full dataset
        
        is_valid = estimated_total < self.config['time_limit']
        self.logger.info(f"Strategy '{strategy['strategy']}' estimated time: {estimated_total:.2f}s "
                        f"({'Valid' if is_valid else 'Invalid'} for {self.config['time_limit']}s limit)")
        return is_valid

    def adjust_batch_size(self, current_batch_size: int, elapsed_time: float, 
                         processed_images: int, total_images: int = 500) -> int:
        """Dynamically adjust batch size based on time constraints"""
        remaining_time = self.config['time_limit'] * self.time_buffer - elapsed_time
        remaining_images = total_images - processed_images
        
        if remaining_images == 0:
            return current_batch_size
            
        time_per_image = elapsed_time / processed_images if processed_images > 0 else 0.1
        suggested_batch_size = int(remaining_images * time_per_image / remaining_time)
        
        # Bound the batch size
        return max(1, min(suggested_batch_size, current_batch_size))
    
    def should_stop_processing(self, elapsed_time: float) -> bool:
        """Check if processing should stop to ensure submission time"""
        return elapsed_time > self.config['time_limit'] * self.time_buffer

    def process_dataset(self, model: nn.Module, dataset_path: Path, strategy: Dict) -> Dict[int, int]:
        """Process dataset with dynamic batch sizing and time management"""
        start_time = time.time()
        results = {}
        processed_images = 0
        current_batch_size = strategy['batch_size']
        
        image_paths = sorted(dataset_path.glob('*.jpg'))
        total_images = len(image_paths)
        
        for i in range(0, total_images, current_batch_size):
            elapsed_time = time.time() - start_time
            
            if self.should_stop_processing(elapsed_time):
                self.logger.warning("Time limit approaching, stopping processing")
                break
            
            # Adjust batch size dynamically
            current_batch_size = self.adjust_batch_size(
                current_batch_size, elapsed_time, processed_images
            )
            
            batch_paths = image_paths[i:i + current_batch_size]
            batch_results = self._process_batch(model, batch_paths, strategy)
            results.update(batch_results)
            
            processed_images += len(batch_paths)
            
            # Log progress
            self.logger.info(f"Processed {processed_images}/{total_images} images. "
                           f"Elapsed time: {elapsed_time:.2f}s, "
                           f"Batch size: {current_batch_size}")
        
        return results

    def safe_predict(self, model: nn.Module, image: torch.Tensor) -> torch.Tensor:
        """Safe prediction with error handling and fallback"""
        try:
            if self.is_safe_mode:
                return self._safe_fallback_predict(model, image)
            
            prediction = model(image)
            self.error_count = 0  # Reset error count on successful prediction
            return prediction
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Prediction failed: {str(e)}")
            
            if self.error_count >= self.max_errors:
                self.logger.warning("Too many errors, switching to safe mode")
                self.is_safe_mode = True
            
            return self._safe_fallback_predict(model, image)
    
    def _safe_fallback_predict(self, model: nn.Module, image: torch.Tensor) -> torch.Tensor:
        """Fallback prediction method with minimal operations"""
        try:
            # Try simplified prediction
            with torch.no_grad():
                image = image.to(self.device)
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                return model(image)
        except Exception as e:
            self.logger.error(f"Fallback prediction failed: {str(e)}")
            # Return zero predictions as last resort
            return torch.zeros((image.size(0), 16), device=self.device)
    
    @torch.no_grad()
    def predict_batch(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """Enhanced batch prediction with error handling"""
        predictions = []
        for i in range(images.size(0)):
            pred = self.safe_predict(model, images[i:i+1])
            predictions.append(pred)
        
        try:
            return torch.cat(predictions, dim=0)
        except Exception as e:
            self.logger.error(f"Failed to concatenate predictions: {str(e)}")
            return torch.zeros((images.size(0), 16), device=self.device)

class BatchInferenceQueue:
    """Handle batch processing with queues for efficient inference"""
    def __init__(self, batch_size: int, num_workers: int):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue()
        
    def process_dataset(self, model: nn.Module, dataset_path: Path, 
                       inference_strategy: AdvancedInference,
                       strategy_config: Dict) -> Dict:
        """Process entire dataset with specified strategy"""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Start loading workers
            load_future = executor.submit(self._load_images, dataset_path)
            # Start inference worker
            infer_future = executor.submit(self._inference_worker, model, 
                                         inference_strategy, strategy_config)
            
            # Wait for completion
            load_future.result()
            infer_future.result()
        
        # Collect results
        results = {}
        while not self.result_queue.empty():
            idx, pred = self.result_queue.get()
            results[idx] = pred
            
        return results
    
    def _load_images(self, dataset_path: Path):
        """Load and preprocess images"""
        image_paths = sorted(dataset_path.glob('*.jpg'))
        for path in image_paths:
            # Implement image loading logic
            pass
    
    def _inference_worker(self, model: nn.Module, 
                         inference_strategy: AdvancedInference,
                         strategy_config: Dict):
        """Worker for performing inference"""
        batch_images = []
        batch_indices = []
        
        while True:
            try:
                idx, image = self.image_queue.get(timeout=5)
                batch_images.append(image)
                batch_indices.append(idx)
                
                if len(batch_images) == self.batch_size:
                    self._process_batch(model, batch_images, batch_indices,
                                     inference_strategy, strategy_config)
                    batch_images = []
                    batch_indices = []
                    
            except queue.Empty:
                if batch_images:  # Process remaining images
                    self._process_batch(model, batch_images, batch_indices,
                                     inference_strategy, strategy_config)
                break
    
    def _process_batch(self, model: nn.Module, images: List[torch.Tensor],
                      indices: List[int], inference_strategy: AdvancedInference,
                      strategy_config: Dict):
        """Process a batch of images"""
        batch = torch.stack(images).to(inference_strategy.device)
        
        if strategy_config['strategy'] == 'base':
            predictions = model(batch)
        elif strategy_config['strategy'] == 'ensemble':
            predictions = inference_strategy.ensemble_predictions(
                [model]*3, batch, strategy_config.get('ensemble_weights')
            )
        else:  # tta
            predictions = inference_strategy.apply_tta(model, batch)
        
        # Store results
        for idx, pred in zip(indices, predictions):
            self.result_queue.put((idx, pred.cpu())) 