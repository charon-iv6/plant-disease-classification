API Reference
============

This section provides detailed API documentation for all modules in the project.

Pipeline Management
-----------------

UnifiedPipelineManager
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: src.pipeline_manager
   :members:
   :undoc-members:
   :show-inheritance:

Model Architecture
----------------

ResNet9
^^^^^^^
.. automodule:: src.models
   :members:
   :undoc-members:
   :show-inheritance:

Training Pipeline
---------------

TrainingPipeline
^^^^^^^^^^^^^^^
.. automodule:: src.training_pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Inference
--------

AdvancedInference
^^^^^^^^^^^^^^^
.. automodule:: src.advanced_inference
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
-----------

Base Visualization
^^^^^^^^^^^^^^^^
.. automodule:: src.visualization_base
   :members:
   :undoc-members:
   :show-inheritance:

Model Visualization
^^^^^^^^^^^^^^^^^
.. automodule:: src.model_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

Submission Visualization
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: src.submission_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-----------

The project uses a JSON configuration file to control all aspects of the pipeline:

.. code-block:: json

    {
        "data_path": "data/",
        "augment_threshold": 100,
        "dropout": 0.3,
        "training": {
            "batch_size": 32,
            "num_workers": 4,
            "epochs": 50,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "early_stopping_patience": 5,
            "mixed_precision": true
        },
        "hardware": {
            "use_cuda": true,
            "use_mps": true,
            "fallback_to_cpu": true
        }
    }

Module Index
----------

* :ref:`modindex`
* :ref:`genindex` 