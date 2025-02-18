import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Tuple, Optional, List, Dict
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantDiseaseDataset(Dataset):
    """Dataset class for plant disease classification
    
    Attributes:
        root_dir: Root directory of the dataset
        transform: Albumentations transform pipeline
        is_train: Whether this is training set
        image_paths: List of paths to images
        labels: List of corresponding labels
        class_to_idx: Dictionary mapping class names to indices
        idx_to_class: Dictionary mapping indices to class names
    """
    
    def __init__(
        self, 
        root_dir: str, 
        transform: Optional[A.Compose] = None, 
        is_train: bool = True,
        cache_size: int = 1000
    ):
        """Initialize the dataset
        
        Args:
            root_dir: Root directory containing train/test splits
            transform: Optional transform pipeline
            is_train: Whether this is training set
            cache_size: Number of images to cache in memory
        
        Raises:
            FileNotFoundError: If root directory doesn't exist
            RuntimeError: If no valid images found
        """
        self.root_dir = Path(root_dir) / ('train' if is_train else 'test')
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory {self.root_dir} does not exist")
            
        self.transform = transform or self._get_default_transform(is_train)
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.cache_size = cache_size
        
        # Load all image paths and labels
        self._load_dataset()
        
        if not self.image_paths:
            raise RuntimeError(f"No valid images found in {self.root_dir}")
            
        logger.info(f"Loaded {len(self)} images from {self.root_dir}")
        
    def _load_dataset(self) -> None:
        """Load dataset paths and create class mappings"""
        class_dirs = sorted(d for d in self.root_dir.iterdir() if d.is_dir())
        
        if not class_dirs:
            raise RuntimeError(f"No class directories found in {self.root_dir}")
        
        self.class_to_idx = {str(d.name): idx for idx, d in enumerate(class_dirs)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[str(class_dir.name)]
            for img_path in class_dir.glob('*.[jJ][pP][gG]'):
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
    
    def _get_default_transform(self, is_train: bool) -> A.Compose:
        """Get default transforms based on train/test mode
        
        Args:
            is_train: Whether this is training set
            
        Returns:
            Albumentations transform pipeline
        """
        if is_train:
            return A.Compose([
                A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                    A.HueSaturationValue(p=1)
                ], p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    @lru_cache(maxsize=1000)
    def _load_image(self, path: str) -> np.ndarray:
        """Load and cache image
        
        Args:
            path: Path to image file
            
        Returns:
            Loaded image array
            
        Raises:
            RuntimeError: If image loading fails
        """
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Failed to load image {path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise RuntimeError(f"Error loading image {path}: {str(e)}")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, label)
            
        Raises:
            RuntimeError: If image loading or transform fails
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load and transform image
            image = self._load_image(str(img_path))
            
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {str(e)}")
            raise RuntimeError(f"Failed to load/transform image {img_path}")
    
    def get_class_name(self, idx: int) -> str:
        """Get class name from index
        
        Args:
            idx: Class index
            
        Returns:
            Class name
            
        Raises:
            KeyError: If invalid class index
        """
        if idx not in self.idx_to_class:
            raise KeyError(f"Invalid class index: {idx}")
        return self.idx_to_class[idx] 