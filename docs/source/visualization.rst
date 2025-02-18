Visualization Guide
=================

The project provides comprehensive visualization tools for analyzing model performance, training progress, and submission results.

Base Visualization
----------------

The ``VisualizationBase`` class provides common functionality for all visualization types:

.. code-block:: python

    from visualization_base import VisualizationBase
    
    visualizer = VisualizationBase(save_dir='assets')
    visualizer.plot_training_progress(history)
    visualizer.plot_confusion_matrix(y_true, y_pred)

Model Visualization
-----------------

The ``ModelVisualizer`` class provides tools for analyzing model architecture and behavior:

Architecture Visualization
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from model_visualizer import ModelVisualizer
    
    visualizer = ModelVisualizer(save_dir='assets/model_viz')
    visualizer.plot_model_architecture(model)
    visualizer.visualize_model_summary(model)

Feature Maps
^^^^^^^^^^

.. code-block:: python

    # Visualize feature maps from specific layers
    sample_input = torch.randn(1, 3, 224, 224)
    visualizer.plot_feature_maps(model, sample_input)

Submission Visualization
---------------------

The ``SubmissionVisualizer`` class provides tools for analyzing and comparing submissions:

Submission Comparison
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from submission_visualizer import SubmissionVisualizer
    
    visualizer = SubmissionVisualizer(save_dir='assets/submission_viz')
    visualizer.compare_submissions(
        submission_files=['sub1.csv', 'sub2.csv'],
        metadata_files=['meta1.json', 'meta2.json']
    )

Performance Analysis
^^^^^^^^^^^^^^^^^

The visualizer generates comprehensive performance reports including:

- Prediction distributions
- Submission agreement matrices
- Performance comparisons
- Detailed analysis reports

Interactive Visualizations
-----------------------

All visualizations are generated in both static (PNG) and interactive (HTML) formats using Plotly:

Training Progress
^^^^^^^^^^^^^^^

.. code-block:: python

    # Interactive training progress visualization
    visualizer.plot_training_progress(history)

Feature Maps
^^^^^^^^^^

.. code-block:: python

    # Interactive feature map exploration
    visualizer.plot_feature_maps(model, sample_input, interactive=True)

Customization
-----------

Style Configuration
^^^^^^^^^^^^^^^^

.. code-block:: python

    # Configure visualization style
    visualizer.set_style(
        style='seaborn',
        context='paper',
        font_scale=1.2
    )

Export Options
^^^^^^^^^^^^

Visualizations can be exported in multiple formats:

.. code-block:: python

    # Save as PNG
    visualizer.save('visualization.png', dpi=300)
    
    # Save as PDF
    visualizer.save('visualization.pdf')
    
    # Save as interactive HTML
    visualizer.save('visualization.html', interactive=True)

Performance Reports
----------------

Comprehensive Reports
^^^^^^^^^^^^^^^^^^

The visualizer can generate detailed performance reports:

.. code-block:: python

    # Generate comprehensive report
    visualizer.create_performance_report(
        history=training_history,
        class_metrics=metrics,
        timing_info=timing_data
    )

Report Contents
^^^^^^^^^^^^^

- Training metrics
- Model performance
- Resource utilization
- Class-wise analysis
- Recommendations

Best Practices
------------

1. **Directory Structure**:
   - Keep visualizations organized by type
   - Use consistent naming conventions
   - Maintain separate directories for different analysis types

2. **Memory Management**:
   - Clear figures after saving
   - Use batch processing for large datasets
   - Monitor memory usage during visualization

3. **Interactive vs Static**:
   - Use interactive plots for exploration
   - Use static plots for documentation
   - Consider file size when choosing format 