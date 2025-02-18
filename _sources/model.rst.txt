Model Architecture
================

ResNet9 Architecture
------------------

The project uses a modified ResNet9 architecture optimized for plant disease classification:

.. code-block:: text

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

* One Cycle Policy
* Max LR: 1e-3
* Min LR: 1e-5

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

Performance Analysis
-----------------

Resource Utilization
^^^^^^^^^^^^^^^^^

* Memory Usage:
    * Training Peak: 4.2GB
    * Inference Peak: 2.1GB

* GPU Utilization:
    * Training: 85-95%
    * Inference: 60-70%

* CPU Usage:
    * Data Loading: 4 workers
    * Preprocessing: ~30% utilization

Implementation Challenges
----------------------

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

Performance Metrics
-----------------
* Training time: ~2-3 hours on M1 Mac
* Inference time: ~0.1s per image
* Best validation F1: ~0.91
* Memory usage: ~4GB

Class Distribution
----------------
.. code-block:: text

   0: سالم (Healthy) - 93 images
   1: کرم سفید ریشه (White Root Worm) - 684 images
   ...
   15: سایر (Others) - 721 images 