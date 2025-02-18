Plant Disease Classification Challenge
==================================

Project Overview
---------------
A deep learning solution for classifying plant diseases into 16 classes, developed for a time-constrained classification challenge by `Danial Jabbari <https://maxion.ir>`_, founder of Maxion AI. This project demonstrates Maxion's expertise in developing efficient, high-performance AI solutions across various domains.

About Maxion AI
-------------
Maxion is a leading AI solutions company that specializes in developing cutting-edge artificial intelligence applications across multiple industries. Notable solutions include:

* `KeyTex <https://keytex.ir>`_ - AI-powered financial signal provider and market analysis platform
* `Legal AI <https://chat.keytex.ir>`_ - Advanced legal assistant and documentation platform
* Custom Enterprise AI Solutions for various industries

Challenge Requirements
--------------------
* **Task**: Classify plant disease images into 16 classes
* **Dataset**: 6,884 training images
* **Test Set**: 500 images per phase
* **Time Limit**: 30 minutes for test set predictions
* **Evaluation**: Macro-averaged F1 score
* **Phases**:
    * Training Phase: Initial model submission
    * Test Phase 1: 30% of final score
    * Test Phase 2: 70% of final score
* **Submission Format**: CSV with ID and class predictions
* **File Naming**: d_jabbari_7727_N.csv (N = submission number)

Challenge Workflow
----------------
1. **Training Phase**
    * Train on 6,884 images
    * Submit trained model
    * Validate model performance

2. **Test Phase 1 (30%)**
    * Process 500 test images
    * Generate predictions within 30 minutes
    * Submit up to 3 different prediction files
    * Evaluated using macro-averaged F1 score

3. **Test Phase 2 (70%)**
    * Process new set of 500 test images
    * Generate predictions within 30 minutes
    * Submit up to 3 different prediction files
    * Evaluated using macro-averaged F1 score

Class Distribution
----------------
The dataset includes the following classes (Persian - English):

.. list-table::
   :header-rows: 1

   * - Class ID
     - Persian Name
     - English Name
     - Image Count
   * - 0
     - سالم
     - Healthy
     - 93
   * - 1
     - کرم سفید ریشه
     - White Root Worm
     - 684
   * - 2
     - سفیدبالک
     - Whitefly
     - 551
   * - 3
     - شپشک آردالود
     - Mealybug
     - 183
   * - 4
     - آبدزدک
     - Water Thief
     - 1523
   * - 5
     - شته غلات
     - Grain Aphid
     - 568
   * - 6
     - آفت سبز
     - Green Pest
     - 304
   * - 7
     - شته یولاف
     - Oat Aphid
     - 590
   * - 8
     - زنجره
     - Leafhopper
     - 913
   * - 9
     - زنگ زدگی
     - Rust
     - 184
   * - 10
     - پوسیدگی
     - Rot
     - 56
   * - 11
     - لکه موجی
     - Wave Spot
     - 108
   * - 12
     - کپک
     - Mold
     - 85
   * - 13
     - بادزدگی
     - Wind Damage
     - 197
   * - 14
     - سفیدک پودری
     - Powdery Mildew
     - 124
   * - 15
     - سایر
     - Others
     - 721

Submission Strategies
-------------------
The system implements three different submission strategies:

1. **Base Strategy**
    * Standard inference
    * Batch size: 32
    * No augmentation
    * Fastest processing

2. **Ensemble Strategy**
    * Combines top 3 checkpoints
    * Batch size: 24
    * Weighted averaging
    * Higher accuracy

3. **TTA Strategy**
    * Test-time augmentation
    * Batch size: 16
    * Multiple transforms
    * Highest accuracy

Time Management
-------------
* 30-minute time limit per phase
* 10% time buffer for safety
* Dynamic batch size adjustment
* Emergency fallback mechanisms
* Progress monitoring and logging

Error Handling
------------
* Hardware failure recovery
* Memory overflow protection
* Network error handling
* Safe mode fallback
* Automatic error logging

Performance Metrics
----------------
* Training:
    * Time: ~2-3 hours
    * Memory: 4.2GB peak
    * F1 Score: 0.91 (validation)

* Inference (per phase):
    * Time: < 30 minutes
    * Memory: 2.1GB peak
    * Images/second: ~10

Contact Information
-----------------
* **Developer**: Danial Jabbari
* **Company**: Maxion AI
* **Email**: danijabbari@protonmail.com
* **Phone**: +98 913 111 7727
* **Website**: https://maxion.ir

Built with
----------
The project is built using state-of-the-art deep learning and data science tools:

* PyTorch - Deep Learning Framework
* NumPy - Numerical Computing
* Pandas - Data Manipulation
* MLflow - Experiment Tracking
* Matplotlib - Visualization
* Python - Programming Language 