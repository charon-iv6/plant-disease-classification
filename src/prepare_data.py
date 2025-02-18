import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm
import argparse
import logging
from typing import Union, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_image(image_path: Path) -> bool:
    """
    Validate if the file is a valid image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        bool: True if valid image, False otherwise
    """
    valid_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
    return image_path.suffix in valid_extensions

def prepare_dataset(
    source_dir: Union[str, Path], 
    target_dir: Union[str, Path], 
    train_ratio: float = 0.8, 
    seed: int = 42,
    num_classes: Optional[int] = 16
) -> None:
    """
    Prepare dataset by splitting into train and test sets.
    
    Args:
        source_dir: Source directory containing class folders (0-15)
        target_dir: Target directory for organized dataset
        train_ratio: Ratio of training data (default: 0.8)
        seed: Random seed for reproducibility
        num_classes: Number of classes expected (default: 16)
        
    Raises:
        FileNotFoundError: If source directory doesn't exist
        ValueError: If train_ratio is invalid or required class folders are missing
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Validate inputs
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist")
    
    if not 0 < train_ratio < 1:
        raise ValueError(f"Train ratio must be between 0 and 1, got {train_ratio}")
    
    random.seed(seed)
    
    # Create target directories
    train_dir = target_dir / 'train'
    test_dir = target_dir / 'test'
    
    try:
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create target directories: {e}")
        raise
    
    # Process each class
    total_files = 0
    valid_classes = []
    
    # First pass to validate and count files
    for class_idx in range(num_classes):
        class_dir = source_dir / str(class_idx)
        if not class_dir.exists():
            logger.warning(f"Class directory {class_idx} not found")
            continue
            
        valid_images = [f for f in class_dir.glob('*') if validate_image(f)]
        if not valid_images:
            logger.warning(f"No valid images found in class {class_idx}")
            continue
            
        total_files += len(valid_images)
        valid_classes.append(class_idx)
    
    if not valid_classes:
        raise ValueError("No valid classes found in the source directory")
    
    logger.info(f"Found {total_files} valid images across {len(valid_classes)} classes")
    
    # Process files
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for class_idx in valid_classes:
            class_dir = source_dir / str(class_idx)
            
            # Create class directories in train and test
            (train_dir / str(class_idx)).mkdir(exist_ok=True)
            (test_dir / str(class_idx)).mkdir(exist_ok=True)
            
            # Get all valid image files
            image_files = [f for f in class_dir.glob('*') if validate_image(f)]
            random.shuffle(image_files)
            
            # Split into train and test
            split_idx = int(len(image_files) * train_ratio)
            train_files = image_files[:split_idx]
            test_files = image_files[split_idx:]
            
            # Copy files with error handling
            for files, dest_dir in [(train_files, train_dir), (test_files, test_dir)]:
                for src in files:
                    try:
                        dst = dest_dir / str(class_idx) / src.name
                        shutil.copy2(src, dst)
                    except Exception as e:
                        logger.error(f"Failed to copy {src} to {dst}: {e}")
                    pbar.update(1)
            
            logger.info(f"Class {class_idx}: {len(train_files)} train, {len(test_files)} test")

def main():
    parser = argparse.ArgumentParser(description='Prepare plant disease dataset')
    parser.add_argument('--source', type=str, required=True,
                      help='Source directory containing class folders')
    parser.add_argument('--target', type=str, default='data',
                      help='Target directory for organized dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                      help='Ratio of training data (between 0 and 1)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--num-classes', type=int, default=16,
                      help='Number of classes expected')
    
    args = parser.parse_args()
    
    try:
        prepare_dataset(
            args.source, 
            args.target, 
            args.train_ratio, 
            args.seed,
            args.num_classes
        )
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise

if __name__ == '__main__':
    main() 