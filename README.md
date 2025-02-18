# Plant Disease Classification Challenge 🌿

## About the Project
A high-performance deep learning solution for classifying plant diseases into 16 classes, developed for a time-constrained classification challenge by [Danial Jabbari](https://maxion.ir), founder of Maxion AI. This implementation features a unified pipeline optimized for both accuracy and inference speed.

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

### 3. Inference Optimization
- Batch processing with queues
- Hardware-specific acceleration (CUDA/Metal)
- Confidence-based TTA
- Transform caching
- Time constraint validation
- Dynamic batch size adjustment

### 4. Challenge-Specific Features
- Multiple submission strategies per phase
- Time-aware processing (30-minute limit)
- Emergency fallback mechanisms
- Comprehensive error handling
- Memory optimization
- Performance monitoring

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
│   ├── training_pipeline.py   # Training implementation
│   ├── advanced_inference.py  # Optimized inference
│   ├── pipeline_manager.py    # Unified pipeline
│   ├── model_visualizer.py    # Model visualization
│   ├── submission_visualizer.py # Submission analysis
│   └── visualization_base.py  # Base visualization
├── docs/                      # Documentation
├── config.json               # Configuration
└── requirements.txt          # Dependencies
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

### Setup
```bash
# Clone repository
git clone https://github.com/charon-iv6/plant-disease-classification.git
cd plant-disease-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Phase
```bash
python src/pipeline_manager.py --config config.json --mode train
```

### Test Phase 1 (30% weight)
```bash
python src/pipeline_manager.py --config config.json --mode test1
```

### Test Phase 2 (70% weight)
```bash
python src/pipeline_manager.py --config config.json --mode test2
```

## Performance

### Training
- Training Time: ~2-3 hours
- Peak GPU Memory: 4.2GB
- Best Validation F1: 0.91

### Inference (Per Phase)
- Processing Time: < 30 minutes
- Batch Size: 32 images
- Average Time/Image: ~0.1s
- Memory Usage: 2.1GB

## Documentation
Detailed documentation is available in the `docs/` directory:
- API Reference
- Architecture Details
- Performance Optimization
- Visualization Guide

## Contact Information
- **Developer**: Danial Jabbari
- **Company**: Maxion AI
- **Email**: danijabbari@protonmail.com
- **Phone**: +98 913 111 7727
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

## Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

### Platform-Specific Setup

#### Windows
```bash
# 1. Clone the repository
git clone https://github.com/charon-iv6/plant-disease-classification.git
cd plant-disease-classification

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. For CUDA support (optional)
# Uncomment the Windows-specific lines in requirements.txt
```

#### macOS (Intel/M1/M2)
```bash
# 1. Clone the repository
git clone https://github.com/charon-iv6/plant-disease-classification.git
cd plant-disease-classification

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. For M1/M2 Macs
# Uncomment the macOS-specific lines in requirements.txt
```

### Verify Installation
```bash
python src/final_model.py --verify-setup
```

## Documentation

### Building Documentation
This project uses GitHub Pages for documentation hosting. To build and view the documentation locally:

1. Install documentation dependencies:
```bash
pip install -r docs/requirements.txt
```

2. Build documentation:
```bash
cd docs
make html
```

3. View documentation:
- Windows: `start _build/html/index.html`
- macOS: `open _build/html/index.html`

### Documentation Structure
```
docs/
├── source/
│   ├── conf.py           # Sphinx configuration
│   ├── index.rst         # Documentation home page
│   ├── api/              # API documentation
│   ├── guides/           # User guides
│   └── examples/         # Example notebooks
└── _build/              # Built documentation
```

### GitHub Pages Setup
The documentation is automatically deployed to GitHub Pages on each push to the main branch. View it at: `https://charon-iv6.github.io/plant-disease-classification`

## Performance Optimization

### Hardware Acceleration

#### CUDA (Windows)
- Automatically uses CUDA if available
- Supports CUDA 11.8 and newer
- Multi-GPU training supported

#### Metal (macOS)
- Automatically uses Metal on M1/M2 Macs
- CPU fallback for Intel Macs
- Optimized for Apple Silicon

### Memory Optimization
- Gradient checkpointing for large models
- Mixed precision training
- Efficient data loading
