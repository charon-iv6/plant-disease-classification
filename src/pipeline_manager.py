import argparse
import logging
import sys
from pathlib import Path
import json
import time
import gc
from typing import Dict, Optional, List, Tuple
import torch
from datetime import datetime
import mlflow
import pandas as pd
import concurrent.futures
import torch.nn as nn
from torch.cuda.amp import GradScaler
import psutil
import numpy as np
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.logging import RichHandler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from training_pipeline import TrainingPipeline
from advanced_inference import AdvancedInference
from model_visualizer import ModelVisualizer
from submission_visualizer import SubmissionVisualizer
from dataset import PlantDiseaseDataset

class UnifiedPipelineManager:
    """Unified pipeline manager for training, inference, visualization, and submission"""
    
    TIME_SAFETY_MARGIN = 60  # seconds
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._setup_directories()
        self.logger = self._setup_logger()
        self.device = self._setup_device()
        
        # Initialize MLflow with more configuration
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        # Initialize performance monitoring
        self.start_time = None
        self.memory_monitor = self._setup_memory_monitor()
        
    def _setup_directories(self):
        """Setup all required directories"""
        dirs = ['output_dir', 'log_dir', 'model_dir', 'visualization_dir', 'submission_dir']
        for dir_name in dirs:
            if dir_name in self.config:
                Path(self.config[dir_name]).mkdir(parents=True, exist_ok=True)
    
    def _setup_memory_monitor(self):
        """Setup memory monitoring"""
        if torch.cuda.is_available():
            import pynvml
            pynvml.nvmlInit()
            return pynvml.nvmlDeviceGetHandleByIndex(0)
        return None
    
    def _get_memory_stats(self) -> Dict:
        """Get current memory statistics"""
        stats = {
            'cpu_percent': psutil.cpu_percent(),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / (1024**3)
        }
        
        if self.memory_monitor:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.memory_monitor)
            stats.update({
                'gpu_used_gb': info.used / (1024**3),
                'gpu_percent': (info.used / info.total) * 100
            })
        elif torch.backends.mps.is_available():
            # For Apple Silicon
            stats.update({
                'gpu_used_gb': torch.mps.current_allocated_memory() / (1024**3),
                'gpu_percent': None  # Not available for MPS
            })
            
        return stats
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        with open(config_path) as f:
            return json.load(f)
        
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging configuration"""
        logger = logging.getLogger('UnifiedPipeline')
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # File handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            log_dir / f'challenge_{timestamp}.log'
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters and add it to the handlers
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Log system information
        self._log_system_info(logger)
        
        return logger
    
    def _log_system_info(self, logger: logging.Logger):
        """Log system and environment information"""
        import platform
        import torch
        import psutil
        
        logger.info("=== System Information ===")
        logger.info(f"OS: {platform.system()} {platform.release()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        logger.info(f"CPU Count: {psutil.cpu_count()}")
        logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
        logger.info("========================")
    
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        if torch.cuda.is_available() and self.config['hardware']['use_cuda']:
            device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available() and self.config['hardware']['use_mps']:
            device = torch.device('mps')
            self.logger.info("Using Apple Silicon GPU")
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU")
        return device
    
    def train_model(self) -> str:
        """Train model with enhanced monitoring and safety features"""
        self.logger.info("Starting model training")
        
        # Create transforms from config
        train_transform = A.Compose([
            A.Resize(*self.config['augmentation']['train']['resize']),
            A.HorizontalFlip(p=self.config['augmentation']['train']['horizontal_flip']),
            A.VerticalFlip(p=self.config['augmentation']['train']['vertical_flip']),
            A.RandomRotate90(p=self.config['augmentation']['train']['rotate90']),
            A.RandomBrightnessContrast(p=self.config['augmentation']['train']['brightness_contrast']),
            A.Normalize(
                mean=self.config['augmentation']['train']['normalize']['mean'],
                std=self.config['augmentation']['train']['normalize']['std']
            ),
            ToTensorV2()
        ])
        
        val_transform = A.Compose([
            A.Resize(*self.config['augmentation']['val']['resize']),
            A.Normalize(
                mean=self.config['augmentation']['val']['normalize']['mean'],
                std=self.config['augmentation']['val']['normalize']['std']
            ),
            ToTensorV2()
        ])
        
        # Create datasets with verification
        train_dataset = PlantDiseaseDataset(
            root_dir=self.config['data_path'],
            is_train=True,
            transform=train_transform
        )
        val_dataset = PlantDiseaseDataset(
            root_dir=self.config['data_path'],
            is_train=False,
            transform=val_transform
        )
        
        # Verify datasets
        self._verify_datasets(train_dataset, val_dataset)
        
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            # Log detailed configuration
            mlflow.log_params(self._flatten_dict(self.config))
            
            # Initialize training pipeline with enhanced config
            pipeline = TrainingPipeline(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=self.config
            )
            
            try:
                # Train with monitoring
                best_model_path = pipeline.train()
                
                # Log final artifacts
                self._log_training_artifacts(pipeline, best_model_path)
                
                return best_model_path
                
            except Exception as e:
                self.logger.error(f"Training failed: {str(e)}")
                self._handle_training_failure(pipeline)
                raise
    
    def _verify_datasets(self, train_dataset, val_dataset):
        """Verify dataset integrity"""
        self.logger.info("Verifying datasets...")
        
        # Check class balance
        train_labels = [sample[1] for sample in train_dataset]
        class_counts = np.bincount(train_labels)
        imbalance_ratio = np.max(class_counts) / np.min(class_counts)
        
        if imbalance_ratio > 10:
            self.logger.warning(f"High class imbalance detected: {imbalance_ratio:.2f}x")
        
        # Log class distribution
        for class_idx, count in enumerate(class_counts):
            self.logger.info(f"Class {class_idx}: {count} samples")
    
    def _log_training_artifacts(self, pipeline, model_path):
        """Log comprehensive training artifacts"""
        # Log model
        mlflow.pytorch.log_model(pipeline.model, "model")
        
        # Log metrics history
        metrics_path = Path(self.config['log_dir']) / "metrics_history.json"
        with open(metrics_path, 'w') as f:
            json.dump(pipeline.metrics_history, f, indent=4)
        mlflow.log_artifact(str(metrics_path))
        
        # Log visualizations
        viz_dir = Path(self.config['visualization_dir'])
        for viz_file in viz_dir.glob('*.png'):
            mlflow.log_artifact(str(viz_file))
    
    def _handle_training_failure(self, pipeline):
        """Handle training failures gracefully"""
        # Save emergency checkpoint
        emergency_path = Path(self.config['model_dir']) / "emergency_checkpoint.pth"
        pipeline.save_checkpoint(emergency_path)
        self.logger.info(f"Emergency checkpoint saved to {emergency_path}")
        
        # Log failure information
        memory_stats = self._get_memory_stats()
        self.logger.error("Training failed with memory stats:")
        for key, value in memory_stats.items():
            self.logger.error(f"- {key}: {value}")
    
    def generate_submissions(self, model_path: str) -> List[str]:
        """Generate submissions with enhanced error handling and parallelization"""
        self.logger.info("Starting submission generation")
        self.start_time = time.time()
        
        submission_paths = []
        inference = AdvancedInference(self.config)
        
        # Load and verify model
        try:
            model = self._load_and_verify_model(model_path)
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            return [self._create_emergency_submission(i) for i in range(1, 4)]
        
        # Enhanced strategy validation
        valid_strategies = self._validate_strategies(model, inference)
        
        # Parallel processing with better error handling
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(valid_strategies), 3)
        ) as executor:
            future_to_strategy = {
                executor.submit(
                    self._process_single_submission_with_retry,
                    model, inference, strategy
                ): strategy for strategy in valid_strategies
            }
            
            for future in concurrent.futures.as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    submission_path = future.result()
                    if submission_path:
                        submission_paths.append(submission_path)
                except Exception as e:
                    self.logger.error(f"Strategy {strategy['number']} failed: {str(e)}")
                    submission_paths.append(
                        self._create_emergency_submission(strategy['number'])
                    )
        
        # Ensure we have exactly 3 submissions
        while len(submission_paths) < 3:
            submission_paths.append(
                self._create_emergency_submission(len(submission_paths) + 1)
            )
        
        return submission_paths[:3]
    
    def _load_and_verify_model(self, model_path: str) -> nn.Module:
        """Load and verify model integrity"""
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        
        # Verify with dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        try:
            with torch.no_grad():
                _ = model(dummy_input)
        except Exception as e:
            raise ValueError(f"Model verification failed: {str(e)}")
        
        return model
    
    def _validate_strategies(self, model: nn.Module,
                           inference: AdvancedInference) -> List[Dict]:
        """Validate submission strategies with timing"""
        valid_strategies = []
        
        for strategy in self.config['submissions']:
            remaining_time, is_critical = self.check_time_remaining()
            
            if is_critical:
                self.logger.warning(f"Time critical! Only {remaining_time:.1f}s remaining")
                break
            
            # Validate with retry
            for attempt in range(3):
                try:
                    sample_batch = torch.randn(
                        strategy['batch_size'], 3, 224, 224
                    ).to(self.device)
                    
                    if inference.validate_time_constraint(model, sample_batch, strategy):
                        valid_strategies.append(strategy)
                        break
                except Exception as e:
                    if attempt == 2:
                        self.logger.warning(
                            f"Strategy {strategy['number']} validation failed after 3 attempts: {str(e)}"
                        )
                    time.sleep(1)
        
        return valid_strategies
    
    def _process_single_submission_with_retry(self, model: nn.Module,
                                            inference: AdvancedInference,
                                            strategy: Dict,
                                            max_retries: int = 3) -> Optional[str]:
        """Process submission with retry mechanism"""
        for attempt in range(max_retries):
            try:
                return self._process_single_submission(model, inference, strategy)
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(
                        f"Strategy {strategy['number']} failed after {max_retries} attempts: {str(e)}"
                    )
                    return None
                time.sleep(1)
    
    def _create_emergency_submission(self, submission_number: int) -> str:
        """Create sophisticated emergency submission"""
        self.logger.warning(f"Creating emergency submission {submission_number}")
        
        # Use class distribution from training set if available
        try:
            train_dataset = PlantDiseaseDataset(
                root_dir=self.config['data_path'],
                is_train=True
            )
            labels = [sample[1] for sample in train_dataset]
            class_dist = np.bincount(labels) / len(labels)
        except:
            class_dist = np.ones(16) / 16  # Uniform distribution as fallback
        
        # Generate predictions based on class distribution
        predictions = {}
        for i in range(500):
            predictions[i] = np.random.choice(16, p=class_dist)
        
        return self._save_submission(predictions, submission_number, 0.0)
    
    def verify_submission(self, submission_path: str) -> bool:
        """Enhanced submission verification"""
        try:
            df = pd.read_csv(submission_path)
            
            # Comprehensive format checks
            checks = {
                'columns': set(df.columns) == {'ID', 'class'},
                'row_count': len(df) == 500,
                'class_range': df['class'].between(0, 15).all(),
                'duplicates': df.duplicated('ID').sum() == 0,
                'missing_ids': set(range(1, 501)) == set(df['ID']),
                'filename_format': self._verify_filename_format(submission_path)
            }
            
            # Log specific issues
            for check_name, passed in checks.items():
                if not passed:
                    self.logger.error(f"Submission verification failed: {check_name}")
                    return False
            
            # Additional statistical checks
            class_dist = df['class'].value_counts(normalize=True)
            if class_dist.max() > 0.5:  # More than 50% predictions are same class
                self.logger.warning("Suspicious class distribution detected")
            
            self.logger.info(f"Submission verification passed: {submission_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Submission verification failed: {str(e)}")
            return False
    
    def _verify_filename_format(self, submission_path: str) -> bool:
        """Verify submission filename format"""
        filename = Path(submission_path).name
        manager_info = self.config['manager_info']
        expected_pattern = (
            f"{manager_info['first_name'][0].lower()}_"
            f"{manager_info['last_name'].lower()}_"
            f"{manager_info['phone'][-4:]}_"
        )
        return filename.startswith(expected_pattern)
    
    def generate_visualizations(self, model_path: str,
                              submission_paths: List[str]) -> None:
        """Generate comprehensive visualizations"""
        self.logger.info("Starting visualization generation")
        
        # Model visualization with enhanced features
        model_viz = ModelVisualizer(
            save_dir=str(Path(self.config['visualization_dir']) / 'model_viz')
        )
        model = torch.load(model_path, map_location=self.device)
        
        try:
            # Basic visualizations
            model_viz.plot_model_architecture(model)
            model_viz.visualize_model_summary(model)
            
            # Advanced visualizations
            sample_input = torch.randn(1, 3, 224, 224).to(self.device)
            model_viz.plot_feature_maps(model, sample_input)
            model_viz.plot_attention_maps(model, sample_input)
            model_viz.generate_grad_cam(model, sample_input)
            
        except Exception as e:
            self.logger.error(f"Model visualization failed: {str(e)}")
        
        # Submission visualization
        submission_viz = SubmissionVisualizer(
            save_dir=str(Path(self.config['visualization_dir']) / 'submission_viz')
        )
        
        try:
            metadata_files = [
                p.with_suffix('.json') for p in map(Path, submission_paths)
            ]
            submission_viz.compare_submissions(submission_paths, metadata_files)
            
        except Exception as e:
            self.logger.error(f"Submission visualization failed: {str(e)}")
    
    def run_pipeline(self) -> None:
        """Run complete pipeline with enhanced monitoring"""
        try:
            # Phase 1: Training
            self.logger.info("Phase 1: Training")
            model_path = self.train_model()
            
            # Phase 2: Generate Submissions
            self.logger.info("Phase 2: Submission Generation")
            submission_paths = self.generate_submissions(model_path)
            
            # Phase 3: Generate Visualizations
            self.logger.info("Phase 3: Visualization")
            self.generate_visualizations(model_path, submission_paths)
            
            # Log final statistics
            self._log_final_statistics()
            
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self._handle_pipeline_failure()
            raise
    
    def _log_final_statistics(self):
        """Log comprehensive final statistics"""
        stats = {
            'total_time': time.time() - self.start_time,
            'memory_stats': self._get_memory_stats(),
            'system_stats': {
                'cpu_count': psutil.cpu_count(),
                'total_memory': psutil.virtual_memory().total / (1024**3)
            }
        }
        
        with open(Path(self.config['log_dir']) / 'final_stats.json', 'w') as f:
            json.dump(stats, f, indent=4)
    
    def _handle_pipeline_failure(self):
        """Handle pipeline failures gracefully"""
        # Save current state
        state_path = Path(self.config['log_dir']) / 'pipeline_state.json'
        with open(state_path, 'w') as f:
            json.dump({
                'config': self.config,
                'timestamp': datetime.now().isoformat(),
                'memory_stats': self._get_memory_stats()
            }, f, indent=4)
        
        self.logger.info(f"Pipeline state saved to {state_path}")
    
    def check_time_remaining(self) -> Tuple[float, bool]:
        """Check remaining time and return (remaining_time, is_critical)"""
        if self.start_time is None:
            return float('inf'), False
        
        elapsed = time.time() - self.start_time
        remaining = self.config['time_limit'] - elapsed
        return remaining, remaining < self.TIME_SAFETY_MARGIN
    
    def _save_submission(self, predictions: Dict[int, int],
                        submission_number: int,
                        elapsed_time: float) -> str:
        """Save submission file with verification"""
        # Format submission name
        manager_info = self.config['manager_info']
        filename = f"{manager_info['first_name'][0].lower()}_{manager_info['last_name'].lower()}_{manager_info['phone'][-4:]}_{submission_number}"
        
        # Save predictions
        submission_path = self.config['submission_dir'] / f"{filename}.csv"
        submission_df = pd.DataFrame(
            [(idx, pred) for idx, pred in predictions.items()],
            columns=['ID', 'class']
        ).sort_values('ID')
        
        # Verify before saving
        submission_df.to_csv(submission_path, index=False)
        if not self.verify_submission(str(submission_path)):
            self.logger.warning(f"Submission {submission_number} failed verification, creating emergency submission")
            return self._create_emergency_submission(submission_number)
        
        # Save metadata
        metadata_path = self.config['submission_dir'] / f"{filename}_metadata.json"
        metadata = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_time': elapsed_time,
            'num_predictions': len(predictions),
            'submission_number': submission_number,
            'manager_info': manager_info,
            'verification_status': 'passed'
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return str(submission_path)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for MLflow logging"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

def main():
    parser = argparse.ArgumentParser(
        description='Unified pipeline for plant disease classification'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test1', 'test2'],
        required=True,
        help='Pipeline mode: train, test1, or test2'
    )
    args = parser.parse_args()
    
    try:
        pipeline = UnifiedPipelineManager(args.config)
        if args.mode == 'train':
            pipeline.train_model()
        elif args.mode in ['test1', 'test2']:
            pipeline.generate_submissions(pipeline.config['model_path'])
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 