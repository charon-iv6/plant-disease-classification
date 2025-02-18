import argparse
import logging
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

from training_pipeline import TrainingPipeline
from advanced_inference import AdvancedInference
from model_visualizer import ModelVisualizer
from submission_visualizer import SubmissionVisualizer

class UnifiedPipelineManager:
    """Unified pipeline manager for training, inference, visualization, and submission"""
    
    TIME_SAFETY_MARGIN = 60  # seconds
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.device = self._setup_device()
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        self.start_time = None
        
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
        """Train model and return path to best checkpoint"""
        self.logger.info("Starting model training")
        
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Log configuration
            mlflow.log_params(self.config)
            
            # Initialize training pipeline
            pipeline = TrainingPipeline(
                train_dataset=self.config['data_path'],
                val_dataset=self.config['data_path'],
                config=self.config['training']
            )
            
            # Train model
            best_model_path = pipeline.train()
            
            # Log metrics and artifacts
            mlflow.log_artifacts(self.config['training']['log_dir'])
            mlflow.log_artifact(best_model_path)
        
        return best_model_path
    
    def check_time_remaining(self) -> Tuple[float, bool]:
        """Check remaining time and return (remaining_time, is_critical)"""
        if self.start_time is None:
            return float('inf'), False
        
        elapsed = time.time() - self.start_time
        remaining = self.config['time_limit'] - elapsed
        return remaining, remaining < self.TIME_SAFETY_MARGIN
    
    def generate_submissions(self, model_path: str) -> List[str]:
        """Generate submissions in parallel with optimized handling"""
        self.logger.info("Starting submission generation")
        self.start_time = time.time()
        
        submission_paths = []
        inference = AdvancedInference(self.config)
        
        # Load model once
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        
        # Pre-validate all strategies
        valid_strategies = []
        for strategy in self.config['submissions']:
            remaining_time, is_critical = self.check_time_remaining()
            
            if is_critical:
                self.logger.warning(f"Time critical! Only {remaining_time:.1f}s remaining")
                break
                
            # Validate strategy timing
            sample_batch = torch.randn(strategy['batch_size'], 3, 224, 224).to(self.device)
            if inference.validate_time_constraint(model, sample_batch, strategy):
                valid_strategies.append(strategy)
            else:
                self.logger.warning(f"Strategy {strategy['number']} validation failed, skipping")
        
        # Process valid strategies
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(valid_strategies), 3)) as executor:
            future_to_strategy = {
                executor.submit(self._process_single_submission, model, inference, strategy): strategy
                for strategy in valid_strategies
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
        
        # Fill remaining submissions with emergency if needed
        while len(submission_paths) < 3:
            submission_paths.append(
                self._create_emergency_submission(len(submission_paths) + 1)
            )
        
        return submission_paths
    
    def _process_single_submission(self, model: nn.Module, 
                                 inference: AdvancedInference,
                                 strategy: Dict) -> Optional[str]:
        """Process a single submission strategy"""
        try:
            self.logger.info(f"Processing strategy {strategy['number']}: {strategy['strategy']}")
            
            # Generate predictions
            start_time = time.time()
            predictions = inference.process_dataset(
                model=model,
                dataset_path=Path(self.config['test_dir']),
                strategy=strategy
            )
            
            # Save submission
            submission_path = self._save_submission(
                predictions,
                strategy['number'],
                time.time() - start_time
            )
            
            # Log performance
            self.logger.info(f"Strategy {strategy['number']} completed:")
            self.logger.info(f"- Time taken: {time.time() - start_time:.2f}s")
            self.logger.info(f"- Images processed: {len(predictions)}")
            
            return submission_path
            
        except Exception as e:
            self.logger.error(f"Strategy {strategy['number']} failed: {str(e)}")
            return None
    
    def _create_emergency_submission(self, submission_number: int) -> str:
        """Create emergency submission with class 0 predictions"""
        self.logger.warning(f"Creating emergency submission {submission_number}")
        predictions = {i: 0 for i in range(500)}  # Default to class 0
        return self._save_submission(predictions, submission_number, 0.0)
    
    def verify_submission(self, submission_path: str) -> bool:
        """Verify submission format and content"""
        try:
            df = pd.read_csv(submission_path)
            
            # Check basic format
            required_columns = {'ID', 'class'}
            if set(df.columns) != required_columns:
                self.logger.error(f"Invalid columns. Expected {required_columns}, got {set(df.columns)}")
                return False
            
            # Check number of predictions
            if len(df) != 500:
                self.logger.error(f"Invalid number of predictions. Expected 500, got {len(df)}")
                return False
            
            # Check value ranges
            if not df['class'].between(0, 15).all():
                invalid_classes = df[~df['class'].between(0, 15)]['class'].unique()
                self.logger.error(f"Invalid class values found: {invalid_classes}")
                return False
            
            # Check for duplicates
            duplicates = df[df.duplicated('ID')]
            if not duplicates.empty:
                self.logger.error(f"Duplicate IDs found: {duplicates['ID'].tolist()}")
                return False
            
            # Check ID format and completeness
            expected_ids = set(range(1, 501))
            actual_ids = set(df['ID'])
            missing_ids = expected_ids - actual_ids
            extra_ids = actual_ids - expected_ids
            
            if missing_ids:
                self.logger.error(f"Missing IDs: {missing_ids}")
                return False
            if extra_ids:
                self.logger.error(f"Extra IDs: {extra_ids}")
                return False
            
            # Verify file naming convention
            filename = Path(submission_path).name
            manager_info = self.config['manager_info']
            expected_pattern = f"{manager_info['first_name'][0].lower()}_{manager_info['last_name'].lower()}_{manager_info['phone'][-4:]}_"
            
            if not filename.startswith(expected_pattern):
                self.logger.error(f"Invalid filename format. Should start with {expected_pattern}")
                return False
            
            self.logger.info(f"Submission verification passed: {submission_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Submission verification failed: {str(e)}")
            return False
    
    def _save_submission(self, predictions: Dict[int, int],
                        submission_number: int,
                        elapsed_time: float) -> str:
        """Save submission file with verification"""
        # Format submission name
        manager_info = self.config['manager_info']
        filename = f"{manager_info['first_name'][0].lower()}_{manager_info['last_name'].lower()}_{manager_info['phone'][-4:]}_{submission_number}"
        
        # Save predictions
        submission_path = self.output_dir / f"{filename}.csv"
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
        metadata_path = self.output_dir / f"{filename}_metadata.json"
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
    
    def generate_visualizations(self, model_path: str,
                              submission_paths: List[str]) -> None:
        """Generate all visualizations"""
        self.logger.info("Starting visualization generation")
        
        # Model visualization
        model_viz = ModelVisualizer(save_dir=str(self.output_dir / 'model_viz'))
        model = torch.load(model_path, map_location=self.device)
        
        model_viz.plot_model_architecture(model)
        model_viz.visualize_model_summary(model)
        
        # Create sample input for feature maps
        sample_input = torch.randn(1, 3, 224, 224).to(self.device)
        model_viz.plot_feature_maps(model, sample_input)
        
        # Submission visualization
        submission_viz = SubmissionVisualizer(save_dir=str(self.output_dir / 'submission_viz'))
        metadata_files = [p.with_suffix('.json') for p in map(Path, submission_paths)]
        submission_viz.compare_submissions(submission_paths, metadata_files)
    
    def run_pipeline(self) -> None:
        """Run complete pipeline"""
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
            
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def log_performance_metrics(self, phase: str, metrics: Dict):
        """Log comprehensive performance metrics"""
        self.logger.info(f"\n=== {phase.upper()} Performance Metrics ===")
        
        # Time metrics
        if 'time' in metrics:
            self.logger.info("\nTiming:")
            self.logger.info(f"Total Time: {metrics['time']['total']:.2f}s")
            self.logger.info(f"Average Time per Image: {metrics['time']['per_image']:.3f}s")
            self.logger.info(f"Batch Processing Time: {metrics['time']['batch_avg']:.3f}s")
        
        # Memory metrics
        if 'memory' in metrics:
            self.logger.info("\nMemory Usage:")
            self.logger.info(f"Peak GPU Memory: {metrics['memory']['gpu_peak']:.2f}GB")
            self.logger.info(f"Peak CPU Memory: {metrics['memory']['cpu_peak']:.2f}GB")
        
        # Performance metrics
        if 'performance' in metrics:
            self.logger.info("\nModel Performance:")
            self.logger.info(f"Macro F1 Score: {metrics['performance']['macro_f1']:.4f}")
            self.logger.info(f"Per-class F1 Scores:")
            for class_id, f1 in enumerate(metrics['performance']['class_f1']):
                self.logger.info(f"Class {class_id}: {f1:.4f}")
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"{phase}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            flattened_metrics = self._flatten_dict(metrics)
            for key, value in flattened_metrics.items():
                mlflow.log_metric(key, value)
    
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
    args = parser.parse_args()
    
    pipeline = UnifiedPipelineManager(args.config)
    pipeline.run_pipeline()

if __name__ == '__main__':
    main() 