# Plant Disease Classification Challenge 🌿

## Project Overview
A deep learning solution for classifying plant diseases into 16 classes, developed for a time-constrained classification challenge by [Danial Jabbari](https://maxion.ir).

### Challenge Requirements
- **Task**: Classify plant disease images into 16 classes
- **Dataset**: 6,884 training images
- **Test Set**: 500 images
- **Time Limit**: 30 minutes for test set predictions
- **Evaluation**: Macro-averaged F1 score
- **Submission Format**: CSV with ID and class predictions

### Class Distribution (Persian - English)
```
0: سالم (Healthy) - 93 images
1: کرم سفید ریشه (White Root Worm) - 684 images
2: سفیدبالک (Whitefly) - 551 images
3: شپشک آردالود (Mealybug) - 183 images
4: آبدزدک (Water Thief) - 1523 images
5: شته غلات (Grain Aphid) - 568 images
6: آفت سبز (Green Pest) - 304 images
7: شته یولاف (Oat Aphid) - 590 images
8: زنجره (Leafhopper) - 913 images
9: زنگ زدگی (Rust) - 184 images
10: پوسیدگی (Rot) - 56 images
11: لکه موجی (Wave Spot) - 108 images
12: کپک (Mold) - 85 images
13: بادزدگی (Wind Damage) - 197 images
14: سفیدک پودری (Powdery Mildew) - 124 images
15: سایر (Others) - 721 images
```

## Built with 🛠️
<code><img height="30" src="https://raw.githubusercontent.com/pytorch/pytorch/39fa0b5d0a3b966a50dcd90b26e6c36942705d6d/docs/source/_static/img/pytorch-logo-dark.svg"></code>
<code><img height="30" src="https://raw.githubusercontent.com/numpy/numpy/7e7f4adab814b223f7f917369a72757cd28b10cb/branding/icons/numpylogo.svg"></code>
<code><img height="30" src="https://raw.githubusercontent.com/pandas-dev/pandas/761bceb77d44aa63b71dda43ca46e8fd4b9d7422/web/pandas/static/img/pandas.svg"></code>
<code><img height="30" src="https://matplotlib.org/_static/logo2.svg"></code>
<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png"></code>

## Project Structure
```
├── final_model.py          # Main model implementation
├── dataset_utils.py        # Dataset handling utilities
├── visualization.py        # Training visualization tools
├── logs/                   # Training logs
└── submissions/           # Challenge submissions
```

## Technical Implementation Details

### Model Architecture Deep Dive
```python
class ResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=16):
        super().__init__()
        # Network structure visualization
        """
        Input (3, 224, 224)
            ↓
        Conv1 + BN + ReLU (64 channels)
            ↓
        MaxPool
            ↓
        ResBlock1 (128 channels)
            ↓
        ResBlock2 (256 channels)
            ↓
        AdaptiveAvgPool
            ↓
        Classifier (16 classes)
        """
```

### Training Optimizations
1. **Mixed Precision Training**
   ```python
   scaler = GradScaler()
   with autocast():
       outputs = model(images)
       loss = criterion(outputs, targets)
   ```

2. **Learning Rate Schedule**
   ![Learning Rate Schedule](assets/lr_schedule.png)
   - One Cycle Policy
   - Max LR: 1e-3
   - Min LR: 1e-5

3. **Data Augmentation Pipeline**
   ```python
   train_transform = A.Compose([
       A.RandomResizedCrop(224, 224),
       A.HorizontalFlip(p=0.5),
       A.VerticalFlip(p=0.5),
       A.RandomRotate90(p=0.5),
       A.OneOf([
           A.RandomBrightnessContrast(p=1),
           A.RandomGamma(p=1),
           A.HueSaturationValue(p=1)
       ], p=0.3),
   ])
   ```

## Performance Analysis

### Training Progress
![Training Progress](assets/training_progress.png)

### Class-wise Performance
![Class Performance](assets/class_performance.png)

### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

### Resource Utilization
```
Memory Usage:
- Training Peak: 4.2GB
- Inference Peak: 2.1GB

GPU Utilization:
- Training: 85-95%
- Inference: 60-70%

CPU Usage:
- Data Loading: 4 workers
- Preprocessing: ~30% utilization
```

## Implementation Challenges & Solutions

### 1. Class Imbalance
- **Challenge**: Extreme imbalance (56 vs 1523 images)
- **Solution**: 
  - Weighted sampling
  - Focal Loss
  - Heavy augmentation for minority classes

### 2. Time Constraint
- **Challenge**: 30-minute limit for 500 images
- **Solution**:
  - Batch processing
  - Confidence-based TTA
  - M1 Mac optimization

### 3. Persian Text Handling
- **Challenge**: Bilingual dataset labeling
- **Solution**:
  - UTF-8 encoding
  - Bidirectional text support

## Deployment Strategy

### M1 Mac Optimization
```python
# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)

# Batch processing optimization
@torch.no_grad()
def predict_batch(images, batch_size=32):
    all_preds = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        preds = model(batch)
        all_preds.extend(preds.cpu())
    return all_preds
```

## About the Author

### Danial Jabbari
- Founder & CEO of [Maxion](https://maxion.ir)
- AI/ML Specialist with focus on Computer Vision
- Experience in agricultural technology and automation

![Maxion Logo](assets/maxion_logo.png)

### Recent Projects
- Plant Disease Detection Systems
- Agricultural Automation Solutions
- Computer Vision Applications

## Contact Information
- 🌐 Website: [maxion.ir](https://maxion.ir)
- 📧 Email: danial.jabbari@maxion.ir
- 💼 LinkedIn: [Danial Jabbari](https://linkedin.com/in/danial-jabbari)
- 🐦 Twitter: [@danial_jabbari](https://twitter.com/danial_jabbari)

## Awards and Recognition
- 🏆 Top Performer in Plant Disease Classification Challenge
- 🌟 Featured in Agricultural AI Solutions
- 💡 Patent Pending: Automated Disease Detection System

## Future Improvements
1. **Model Optimization**
   - Knowledge distillation
   - Model quantization
   - Mobile deployment

2. **Dataset Enhancement**
   - Additional augmentation techniques
   - Cross-validation strategies
   - Active learning implementation

3. **Inference Speed**
   - ONNX runtime integration
   - TensorRT optimization
   - Batch size optimization

## Contributing
Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
Special thanks to:
- The challenge organizers
- PyTorch team
- Agricultural research partners
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
git clone https://github.com/yourusername/plant-disease-classification.git
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
git clone https://github.com/yourusername/plant-disease-classification.git
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
The documentation is automatically deployed to GitHub Pages on each push to the main branch. View it at: `https://yourusername.github.io/plant-disease-classification`

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
