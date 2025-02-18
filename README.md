# Plant Disease Classification Challenge 🌿

## About the Project
A high-performance deep learning solution for classifying plant diseases into 16 classes, developed for a time-constrained classification challenge by [Danial Jabbari](https://maxion.ir), founder of Maxion AI. This implementation features a unified pipeline optimized for both accuracy and inference speed.

## Latest Updates
- Added comprehensive error handling and validation
- Implemented image caching for improved performance
- Added type hints and improved documentation
- Consolidated requirements with platform-specific markers
- Improved model weight initialization
- Enhanced dataset handling with validation

## About Maxion AI
Maxion is a leading AI solutions company specializing in cutting-edge artificial intelligence applications:
- [KeyTex](https://keytex.ir) - AI-powered financial signal provider and market analysis platform
- [Legal AI](https://chat.keytex.ir) - Advanced legal assistant and documentation platform
- Custom Enterprise AI Solutions

## Challenge Requirements
- **Task**: Classify plant disease images into 16 classes
- **Dataset**: 6,884 training images
- **Test Set**: 500 images
- **Time Limit**: 30 minutes for test set predictions
- **Evaluation**: Macro-averaged F1 score (30% Phase 1, 70% Phase 2)
- **Submission Format**: CSV with ID and class predictions

## Key Features

### 1. Unified Pipeline
- Single pipeline managing all aspects of the workflow
- Integrated MLflow experiment tracking
- Comprehensive logging and visualization
- Automated submission generation for multiple phases

### 2. Training Optimization
- Mixed precision training (FP16)
- Test-time augmentation (TTA)
- Weighted sampling for class imbalance
- Exponential Moving Average (EMA)
- Learning rate optimization
- Automatic checkpointing
- Kaiming weight initialization

### 3. Inference Optimization
- Batch processing with queues
- Hardware-specific acceleration (CUDA/Metal)
- Confidence-based TTA
- Transform caching
- Time constraint validation
- Dynamic batch size adjustment
- Image caching with LRU policy

### 4. Robust Error Handling
- Comprehensive input validation
- Graceful failure recovery
- Detailed error logging
- Memory optimization
- Performance monitoring
- Automatic error reporting

### 5. Visualization Tools
- Interactive training progress plots
- Model architecture visualization
- Feature map analysis
- Performance metrics visualization
- Submission comparison tools
- Comprehensive reporting

## Project Structure
```
├── src/
│   ├── models.py              # Model architecture (ResNet9)
│   ├── dataset.py            # Dataset handling with caching
│   ├── prepare_data.py       # Data preparation with validation
│   ├── training_pipeline.py   # Training implementation
│   ├── advanced_inference.py  # Optimized inference
│   ├── pipeline_manager.py    # Unified pipeline
│   ├── model_visualizer.py    # Model visualization
│   ├── submission_visualizer.py # Submission analysis
│   └── visualization_base.py  # Base visualization
├── docs/                      # Documentation
├── config.json               # Configuration
└── requirements.txt          # Platform-specific dependencies
```

## Class Distribution
| Class ID | Persian Name | English Name | Count |
|----------|--------------|--------------|-------|
| 0 | سالم | Healthy | 93 |
| 1 | کرم سفید ریشه | White Root Worm | 684 |
| 2 | سفیدبالک | Whitefly | 551 |
| 3 | شپشک آردالود | Mealybug | 183 |
| 4 | آبدزدک | Water Thief | 1523 |
| 5 | شته غلات | Grain Aphid | 568 |
| 6 | آفت سبز | Green Pest | 304 |
| 7 | شته یولاف | Oat Aphid | 590 |
| 8 | زنجره | Leafhopper | 913 |
| 9 | زنگ زدگی | Rust | 184 |
| 10 | پوسیدگی | Rot | 56 |
| 11 | لکه موجی | Wave Spot | 108 |
| 12 | کپک | Mold | 85 |
| 13 | بادزدگی | Wind Damage | 197 |
| 14 | سفیدک پودری | Powdery Mildew | 124 |
| 15 | سایر | Others | 721 |

## Installation

### Prerequisites
- Python 3.8+
- CUDA toolkit (optional, for GPU support)
- Metal support (optional, for Apple Silicon)

### Platform-Specific Setup

#### Windows
```bash
# Clone repository
git clone https://github.com/charon-iv6/plant-disease-classification.git
cd plant-disease-classification

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### macOS (Intel/M1/M2)
```bash
# Clone repository
git clone https://github.com/charon-iv6/plant-disease-classification.git
cd plant-disease-classification

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation
```bash
# Prepare dataset with validation
python src/prepare_data.py --source /path/to/source --target data
```

### Training Phase
```bash
# Train with improved initialization
python src/pipeline_manager.py --config config.json --mode train
```

### Test Phase
```bash
# Run inference with optimizations
python src/pipeline_manager.py --config config.json --mode test
```

## Performance Improvements

### Training
- Improved convergence with Kaiming initialization
- Reduced memory usage with image caching
- Better error handling and validation
- Type-safe implementation

### Inference
- Optimized image loading with LRU cache
- Robust error recovery
- Platform-specific optimizations
- Comprehensive logging

## Documentation
Detailed documentation is available in the `docs/` directory:
- API Reference
- Architecture Details
- Performance Optimization
- Visualization Guide
- Error Handling Guide
- Platform-Specific Notes

## Contact Information
- **Developer**: Danial Jabbari
- **Company**: Maxion AI
- **Email**: danijabbari@protonmail.com
- **Website**: [maxion.ir](https://maxion.ir)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- PyTorch team
- MLflow team
- Open source community

---
<p align="center">
  <img src="assets/maxion_footer.png" alt="Maxion AI" width="200"/>
  <br>
  <i>Empowering Agriculture with AI</i>
</p>
