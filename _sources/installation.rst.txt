Installation Guide
=================

Prerequisites
------------

Before installing the project, ensure you have:

* Python 3.8 or higher
* pip (Python package installer)
* Git
* CUDA toolkit (optional, for GPU support on Windows)

Platform-Specific Installation
----------------------------

Windows Installation
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # 1. Clone the repository
   git clone https://github.com/charon-iv6/plant-disease-classification.git
   cd plant-disease-classification

   # 2. Create and activate virtual environment
   python -m venv venv
   .\venv\Scripts\activate

   # 3. Install dependencies
   pip install -r requirements.txt

   # 4. For CUDA support (optional)
   # Uncomment Windows-specific lines in requirements.txt:
   # torch>=2.0.0+cu118; platform_system == "Windows"
   # torchvision>=0.15.0+cu118; platform_system == "Windows"

macOS Installation
^^^^^^^^^^^^^^^

.. code-block:: bash

   # 1. Clone the repository
   git clone https://github.com/charon-iv6/plant-disease-classification.git
   cd plant-disease-classification

   # 2. Create and activate virtual environment
   python3 -m venv venv
   source venv/bin/activate

   # 3. Install dependencies
   pip install -r requirements.txt

   # 4. For M1/M2 Macs (optional)
   # Uncomment macOS-specific lines in requirements.txt:
   # torch>=2.0.0; platform_system == "Darwin"
   # torchvision>=0.15.0; platform_system == "Darwin"

Verifying Installation
--------------------

To verify your installation:

.. code-block:: bash

   # Activate virtual environment (if not already activated)
   # Windows:
   .\venv\Scripts\activate
   # macOS:
   source venv/bin/activate

   # Run verification script
   python src/final_model.py --verify-setup

You should see output confirming:
* All dependencies are correctly installed
* PyTorch can access your GPU (if applicable)
* Test image processing works correctly

Troubleshooting
-------------

Common Issues
^^^^^^^^^^^^

1. CUDA not found (Windows):
   
   * Ensure CUDA toolkit is installed
   * Verify CUDA version matches PyTorch requirements
   * Check system PATH includes CUDA directories

2. Metal acceleration not working (M1/M2 Macs):
   
   * Ensure you're using PyTorch 2.0 or higher
   * Verify macOS-specific PyTorch installation

3. Package conflicts:
   
   * Try creating a fresh virtual environment
   * Install packages in the order listed in requirements.txt

Getting Help
^^^^^^^^^^

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/charon-iv6/plant-disease-classification/issues>`_
2. Create a new issue with:
   * Your operating system and version
   * Python version
   * Full error message
   * Steps to reproduce the problem 