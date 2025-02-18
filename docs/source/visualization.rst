Visualization Guide
=================

This guide covers the visualization tools available in the project for analyzing model performance and data distribution.

Training Visualizations
--------------------

Learning Curves
^^^^^^^^^^^^^

.. code-block:: python

   from src.visualization import plot_learning_curves
   
   plot_learning_curves(history)

This will generate plots for:
* Training and validation loss
* Training and validation accuracy
* Learning rate schedule

Class Distribution
^^^^^^^^^^^^^^^

.. code-block:: python

   from src.visualization import plot_class_distribution
   
   plot_class_distribution(dataset)

Features:
* Bar chart of samples per class
* Percentage distribution
* Imbalance ratio calculation

Model Performance
---------------

Confusion Matrix
^^^^^^^^^^^^^

.. code-block:: python

   from src.visualization import plot_confusion_matrix
   
   plot_confusion_matrix(y_true, y_pred, class_names)

Features:
* Normalized confusion matrix
* Per-class accuracy
* Misclassification patterns

Performance Metrics
^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.visualization import plot_performance_metrics
   
   plot_performance_metrics(y_true, y_pred)

Displays:
* Precision
* Recall
* F1-score
* Support

Advanced Visualizations
--------------------

Grad-CAM Visualization
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.advanced_visualizations import visualize_gradcam
   
   visualize_gradcam(model, image, target_class)

Features:
* Class activation mapping
* Attention visualization
* Feature importance

t-SNE Embedding
^^^^^^^^^^^^^

.. code-block:: python

   from src.advanced_visualizations import plot_tsne
   
   plot_tsne(features, labels)

Shows:
* Feature space clustering
* Class separation
* Dimensionality reduction

Customization
-----------

Style Configuration
^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.visualization import set_visualization_style
   
   set_visualization_style(
       style='seaborn',
       context='paper',
       font_scale=1.2
   )

Available styles:
* seaborn
* matplotlib
* plotly

Export Options
^^^^^^^^^^^^

.. code-block:: python

   # Save as PNG
   plt.savefig('visualization.png', dpi=300, bbox_inches='tight')
   
   # Save as PDF
   plt.savefig('visualization.pdf', format='pdf', bbox_inches='tight')
   
   # Save as SVG
   plt.savefig('visualization.svg', format='svg', bbox_inches='tight') 