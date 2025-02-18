Model Architecture
================

ResNet9 Architecture
------------------

The project uses a modified ResNet9 architecture optimized for plant disease classification:

Architecture Overview
^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Input (3, 224, 224)
       ↓
   Conv1 + BN + ReLU (64)
       ↓
   MaxPool
       ↓
   ResBlock1 (128)
       ↓
   ResBlock2 (256)
       ↓
   ResBlock3 (512)
       ↓
   AdaptiveAvgPool
       ↓
   Classifier (16)

Key Components
^^^^^^^^^^^^

1. **Initial Layer**:
   - Convolutional layer (64 channels)
   - Batch normalization
   - ReLU activation

2. **Residual Blocks**:
   - Three residual blocks
   - Channel expansion (64 → 128 → 256 → 512)
   - Skip connections for better gradient flow

3. **Pooling Layers**:
   - MaxPool2d for spatial reduction
   - AdaptiveAvgPool2d for fixed output size

4. **Classifier**:
   - Dropout for regularization
   - Two fully connected layers
   - Final softmax for 16 classes

Training Optimizations
-------------------

Mixed Precision Training
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   scaler = GradScaler()
   with autocast():
       outputs = model(images)
       loss = criterion(outputs, targets)

Learning Rate Schedule
^^^^^^^^^^^^^^^^^^

- One Cycle Policy
- Max LR: 1e-3
- Min LR: 1e-5
- Cosine annealing

Data Augmentation
^^^^^^^^^^^^^^^

.. code-block:: python

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

Model Components
-------------

Exponential Moving Average
^^^^^^^^^^^^^^^^^^^^^^^

The ``ModelEMA`` class maintains an exponential moving average of model parameters:

.. code-block:: python

   ema_model = ModelEMA(model, decay=0.999)
   ema_model.update(model)

Test Time Augmentation
^^^^^^^^^^^^^^^^^^^

Multiple augmentation strategies during inference:

.. code-block:: python

   def predict_with_tta(model, image):
       transforms = [
           A.HorizontalFlip(p=1.0),
           A.VerticalFlip(p=1.0),
           A.RandomRotate90(p=1.0)
       ]
       predictions = []
       # Original prediction
       predictions.append(model(image))
       # Augmented predictions
       for transform in transforms:
           aug_image = transform(image=image)
           predictions.append(model(aug_image))
       return torch.stack(predictions).mean(0)

Hardware Acceleration
------------------

CUDA Support (Windows)
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)

Metal Support (macOS)
^^^^^^^^^^^^^^^^^

.. code-block:: python

   device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
   model = model.to(device)

Performance Metrics
----------------

Training Performance
^^^^^^^^^^^^^^^^^
- Training time: ~2-3 hours
- Peak GPU memory: ~4GB
- Best validation F1: ~0.91

Inference Performance
^^^^^^^^^^^^^^^^^
- Batch size: 32 images
- Average inference time: ~0.1s per image
- Memory usage: ~2.1GB

Implementation Details
-------------------

1. **Initialization**:
   - Xavier initialization for convolutional layers
   - Kaiming initialization for linear layers
   - Bias initialization to zero

2. **Regularization**:
   - Dropout (p=0.1)
   - Weight decay (1e-4)
   - Label smoothing (0.1)

3. **Optimization**:
   - AdamW optimizer
   - Gradient clipping
   - Learning rate warmup

4. **Loss Function**:
   - Cross-entropy with class weights
   - Label smoothing
   - Focal loss for imbalanced classes

Class Imbalance
^^^^^^^^^^^^^

* Challenge: Extreme imbalance (56 vs 1523 images)
* Solutions:
    * Weighted sampling
    * Focal Loss
    * Heavy augmentation for minority classes

Time Constraint
^^^^^^^^^^^^^

* Challenge: 30-minute limit for 500 images
* Solutions:
    * Batch processing
    * Confidence-based TTA
    * Hardware acceleration optimization

Persian Text Support
^^^^^^^^^^^^^^^^^

* Challenge: Bilingual dataset labeling
* Solutions:
    * UTF-8 encoding
    * Bidirectional text support

Batch Processing
^^^^^^^^^^^^^

.. code-block:: python

   @torch.no_grad()
   def predict_batch(images, batch_size=32):
       all_preds = []
       for i in range(0, len(images), batch_size):
           batch = images[i:i + batch_size].to(device)
           preds = model(batch)
           all_preds.extend(preds.cpu())
       return all_preds

Training Pipeline
---------------
.. autofunction:: src.final_model.train_model

Inference Pipeline
----------------
.. autofunction:: src.final_model.create_challenge_submission

Class Distribution
----------------
.. code-block:: text

   0: سالم (Healthy) - 93 images
   1: کرم سفید ریشه (White Root Worm) - 684 images
   ...
   15: سایر (Others) - 721 images 