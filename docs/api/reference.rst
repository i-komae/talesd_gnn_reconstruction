Python Reference
================

This page exposes the main Python entry points. The usage-oriented pages in
``ja/`` and ``en/`` should be read first.

Training
--------

.. automodule:: talesd_gnn_reconstruction.train
   :members: train_model

Dataset
-------

.. automodule:: talesd_gnn_reconstruction.dataset
   :members: H5GraphDataset

Prediction
----------

.. automodule:: talesd_gnn_reconstruction.predict
   :members: predict_graphs

Feature analysis
----------------

.. automodule:: talesd_gnn_reconstruction.feature_analysis
   :members: save_input_distributions, save_feature_group_importance

Graph construction and I/O
--------------------------

.. automodule:: talesd_gnn_reconstruction.event_graph
   :members: build_graph_event

.. automodule:: talesd_gnn_reconstruction.graph_io
   :members: create_graph_file, write_graph

Model
-----

.. automodule:: talesd_gnn_reconstruction.model
   :members: PhysicsTaleSdGNN

