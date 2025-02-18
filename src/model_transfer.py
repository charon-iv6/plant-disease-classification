import torch
import logging
from pathlib import Path
import shutil
import json
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_model_for_transfer(model_path: str, output_path: str = "model_transfer"):
    """Save model in a format suitable for transfer between PC and Mac
    
    Args:
        model_path: Path to the trained model
        output_path: Directory to save the transfer files
    """
    try:
        # Load model
        model = torch.load(model_path, map_location='cpu')
        
        # Create transfer directory
        transfer_dir = Path(output_path)
        transfer_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        state_dict_path = transfer_dir / "model_state_dict.pth"
        torch.save(model.state_dict(), state_dict_path)
        
        # Save model architecture info
        arch_info = {
            'num_classes': model.classifier[-1].out_features,
            'in_channels': model.conv1.in_channels,
            'dropout': model.dropout.p
        }
        
        with open(transfer_dir / "model_info.json", 'w') as f:
            json.dump(arch_info, f, indent=4)
            
        # Create README
        readme_content = """Model Transfer Package
        
This package contains:
1. model_state_dict.pth - Model weights
2. model_info.json - Model architecture information
3. README.md - This file

To load this model:
1. Initialize ResNet9 with parameters from model_info.json
2. Load state_dict from model_state_dict.pth
"""
        
        with open(transfer_dir / "README.md", 'w') as f:
            f.write(readme_content)
            
        # Create zip archive
        shutil.make_archive(output_path, 'zip', transfer_dir)
        
        logger.info(f"Model transfer package created at {output_path}.zip")
        
    except Exception as e:
        logger.error(f"Failed to create transfer package: {str(e)}")
        raise

def load_transferred_model(transfer_dir: str) -> torch.nn.Module:
    """Load model from transfer package
    
    Args:
        transfer_dir: Directory containing the transfer files
        
    Returns:
        Loaded PyTorch model
    """
    try:
        transfer_dir = Path(transfer_dir)
        
        # Load model architecture info
        with open(transfer_dir / "model_info.json") as f:
            arch_info = json.load(f)
            
        # Initialize model
        from models import ResNet9
        model = ResNet9(
            in_channels=arch_info['in_channels'],
            num_classes=arch_info['num_classes'],
            dropout=arch_info['dropout']
        )
        
        # Load state dict
        state_dict = torch.load(
            transfer_dir / "model_state_dict.pth",
            map_location='cpu'
        )
        model.load_state_dict(state_dict)
        
        logger.info("Model loaded successfully from transfer package")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model from transfer package: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Model transfer utility')
    parser.add_argument('--mode', choices=['save', 'load'], required=True,
                      help='Operation mode: save or load')
    parser.add_argument('--model-path', type=str,
                      help='Path to model file (for save mode)')
    parser.add_argument('--transfer-dir', type=str,
                      help='Path to transfer directory (for load mode)')
    parser.add_argument('--output', type=str, default='model_transfer',
                      help='Output path (for save mode)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'save':
            if not args.model_path:
                raise ValueError("--model-path required for save mode")
            save_model_for_transfer(args.model_path, args.output)
        else:
            if not args.transfer_dir:
                raise ValueError("--transfer-dir required for load mode")
            model = load_transferred_model(args.transfer_dir)
            
    except Exception as e:
        logger.error(f"Model transfer failed: {str(e)}")
        raise

if __name__ == '__main__':
    main() 