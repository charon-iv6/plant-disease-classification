import os
import shutil
from pathlib import Path
import logging
from typing import Tuple, Dict, List
from tqdm import tqdm
import random
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """Manages data preparation, validation, and statistics for the plant disease dataset"""
    
    def __init__(self, config: dict):
        """Initialize the data manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_path = Path(config['data_path'])
        self.train_path = self.data_path / 'train'
        self.test_path = self.data_path / 'test'
        self.image_size = (
            config['image_size']['height'],
            config['image_size']['width']
        )
        
    def prepare_dataset(self, source_dir: str, train_ratio: float = 0.8) -> None:
        """Prepare and validate the dataset
        
        Args:
            source_dir: Source directory containing class folders
            train_ratio: Ratio of training data
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory {source_dir} does not exist")
        
        # Create directories
        self.train_path.mkdir(parents=True, exist_ok=True)
        self.test_path.mkdir(parents=True, exist_ok=True)
        
        # Process each class
        class_stats = self._process_classes(source_path, train_ratio)
        
        # Save dataset statistics
        self._save_dataset_stats(class_stats)
        
        logger.info("Dataset preparation completed successfully")
        
    def _process_classes(self, source_path: Path, train_ratio: float) -> Dict:
        """Process all classes and split into train/test
        
        Args:
            source_path: Path to source directory
            train_ratio: Ratio of training data
            
        Returns:
            Dictionary containing dataset statistics
        """
        class_stats = {}
        
        for class_dir in sorted(d for d in source_path.iterdir() if d.is_dir()):
            class_idx = class_dir.name
            logger.info(f"Processing class {class_idx}")
            
            # Create class directories
            (self.train_path / class_idx).mkdir(exist_ok=True)
            (self.test_path / class_idx).mkdir(exist_ok=True)
            
            # Get valid images
            valid_images = self._get_valid_images(class_dir)
            if not valid_images:
                logger.warning(f"No valid images found in class {class_idx}")
                continue
                
            # Split and copy images
            train_imgs, test_imgs = self._split_and_copy_images(
                valid_images, 
                class_idx, 
                train_ratio
            )
            
            # Record statistics
            class_stats[class_idx] = {
                'total': len(valid_images),
                'train': len(train_imgs),
                'test': len(test_imgs)
            }
            
        return class_stats
    
    def _get_valid_images(self, class_dir: Path) -> List[Path]:
        """Get list of valid images from directory
        
        Args:
            class_dir: Directory to scan
            
        Returns:
            List of valid image paths
        """
        valid_images = []
        for img_path in class_dir.glob('*.[jJ][pP][gG]'):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    # Verify image can be resized
                    resized = cv2.resize(img, self.image_size)
                    valid_images.append(img_path)
            except Exception as e:
                logger.warning(f"Invalid image {img_path}: {str(e)}")
                
        return valid_images
    
    def _split_and_copy_images(
        self, 
        images: List[Path], 
        class_idx: str,
        train_ratio: float
    ) -> Tuple[List[Path], List[Path]]:
        """Split images into train/test and copy to respective directories
        
        Args:
            images: List of image paths
            class_idx: Class index/name
            train_ratio: Ratio of training data
            
        Returns:
            Tuple of (train_images, test_images)
        """
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]
        
        # Copy files with parallel processing
        with ThreadPoolExecutor() as executor:
            # Copy train images
            train_futures = [
                executor.submit(
                    self._copy_image, 
                    img, 
                    self.train_path / class_idx / img.name
                )
                for img in train_imgs
            ]
            
            # Copy test images
            test_futures = [
                executor.submit(
                    self._copy_image,
                    img,
                    self.test_path / class_idx / img.name
                )
                for img in test_imgs
            ]
            
            # Wait for completion
            for future in train_futures + test_futures:
                future.result()
        
        return train_imgs, test_imgs
    
    def _copy_image(self, src: Path, dst: Path) -> None:
        """Copy image with error handling
        
        Args:
            src: Source path
            dst: Destination path
        """
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            logger.error(f"Failed to copy {src} to {dst}: {str(e)}")
            raise
    
    def _save_dataset_stats(self, stats: Dict) -> None:
        """Save dataset statistics to JSON file
        
        Args:
            stats: Dictionary containing dataset statistics
        """
        stats_file = self.data_path / 'dataset_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
        
        # Log summary
        total_train = sum(s['train'] for s in stats.values())
        total_test = sum(s['test'] for s in stats.values())
        logger.info(f"Dataset split complete: {total_train} train, {total_test} test images")
    
    def validate_dataset(self) -> bool:
        """Validate the prepared dataset
        
        Returns:
            True if dataset is valid, False otherwise
        """
        try:
            # Check directory structure
            if not (self.train_path.exists() and self.test_path.exists()):
                logger.error("Train or test directory missing")
                return False
            
            # Check class directories
            train_classes = set(d.name for d in self.train_path.iterdir() if d.is_dir())
            test_classes = set(d.name for d in self.test_path.iterdir() if d.is_dir())
            
            if train_classes != test_classes:
                logger.error("Mismatch in train/test class directories")
                return False
            
            # Validate images in each class
            for class_idx in train_classes:
                train_imgs = list((self.train_path / class_idx).glob('*.[jJ][pP][gG]'))
                test_imgs = list((self.test_path / class_idx).glob('*.[jJ][pP][gG]'))
                
                if not (train_imgs and test_imgs):
                    logger.error(f"Missing images in class {class_idx}")
                    return False
                
                # Validate random sample of images
                sample_size = min(10, len(train_imgs))
                for img_path in random.sample(train_imgs + test_imgs, sample_size):
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            raise ValueError("Failed to load image")
                        # Verify image can be resized
                        resized = cv2.resize(img, self.image_size)
                    except Exception as e:
                        logger.error(f"Invalid image {img_path}: {str(e)}")
                        return False
            
            logger.info("Dataset validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            return False
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats_file = self.data_path / 'dataset_stats.json'
        if stats_file.exists():
            with open(stats_file) as f:
                return json.load(f)
        return {} 