Plant Disease Classification Challenge
==================================

Project Overview
---------------
A deep learning solution for classifying plant diseases into 16 classes, developed for a time-constrained classification challenge by `Danial Jabbari <https://maxion.ir>`_.

Challenge Requirements
--------------------
* **Task**: Classify plant disease images into 16 classes
* **Dataset**: 6,884 training images
* **Test Set**: 500 images
* **Time Limit**: 30 minutes for test set predictions
* **Evaluation**: Macro-averaged F1 score
* **Submission Format**: CSV with ID and class predictions

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

Built with
----------
The project is built using state-of-the-art deep learning and data science tools:

* PyTorch - Deep Learning Framework
* NumPy - Numerical Computing
* Pandas - Data Manipulation
* Matplotlib - Visualization
* Python - Programming Language 