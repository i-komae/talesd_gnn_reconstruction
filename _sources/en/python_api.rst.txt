Python API
==========

Use the Python API for notebooks and small or medium checks. For large production training, use the Slurm submitters.
The workflow map is in :doc:`code_map`.
Direct Python calls use the same dataset, model, loss, and diagnostics code as the CLI.

Training
--------

.. code-block:: python

   from talesd_gnn_reconstruction.train import train_model

   result = train_model(
       graphs_path="/path/to/graphs/flat50000",
       output_path="/path/to/checkpoints/reconstruction.pt",
       epochs=128,
       batch_size=256,
       training_task="reconstruction",
       model_architecture="physics",
       waveform_encoder="cnn-gru",
       loss_mode="physics",
       quality_prediction=True,
       split_mode="source-stratified",
       val_fraction=0.05,
       test_fraction=0.10,
       device="cuda",
   )

``train_model`` runs dataset initialization, splitting, scaler fitting, DataLoader setup, training, validation, best checkpoint saving, final test evaluation, and diagnostics.
The returned dictionary includes checkpoint paths, metrics JSON paths, diagnostics directories, and stage timings.
For notebooks, prefer small graphs or short checks instead of full production training.

HDF5 dataset loading
--------------------

.. code-block:: python

   from talesd_gnn_reconstruction.dataset import H5GraphDataset

   dataset = H5GraphDataset(
       "/path/to/graphs/flat50000",
       require_target=True,
       load_particle_label=True,
   )

   sample = dataset[0]
   dataset.close()

``sample`` contains ``node_features``, ``edge_features``, ``waveform_features``, ``target``, and metadata.
This is the closest API to the tensors later consumed by the model, so it is the first place to inspect schema or input-debugging issues.

Prediction
----------

.. code-block:: python

   from talesd_gnn_reconstruction.predict import predict_graphs

   csv_path = predict_graphs(
       graphs_path="/path/to/graphs/data_graphs.h5",
       checkpoint_path="/path/to/checkpoints/reconstruction.pt",
       output_csv="/path/to/predictions.csv",
       batch_size=256,
       device="cuda",
   )

Input distributions
-------------------

.. code-block:: python

   from talesd_gnn_reconstruction.feature_analysis import save_input_distributions

   summary = save_input_distributions(
       graphs_path="/path/to/graphs/flat50000",
       output_dir="/path/to/input_distributions",
       max_graphs=100000,
       show_progress=True,
   )

``summary`` includes output paths and basic statistics for each feature.
For large HDF5 datasets, run this on the server and sync only the generated PDF/JSON outputs.

Feature group importance
------------------------

.. code-block:: python

   from talesd_gnn_reconstruction.feature_analysis import save_feature_group_importance

   result = save_feature_group_importance(
       graphs_path="/path/to/graphs/flat50000",
       checkpoint_path="/path/to/checkpoints/reconstruction.pt",
       output_dir="/path/to/feature_importance",
       split="validation",
       max_graphs=50000,
       batch_size=256,
       device="cuda",
   )

``result`` contains metric degradation by feature group.
This estimates input-group contribution from a trained checkpoint without running many retraining jobs.

Manual HDF5 graph writing
-------------------------

Normally use ``talesd-gnn export``. The low-level writer API is:

.. code-block:: python

   from talesd_gnn_reconstruction.graph_io import create_graph_file, write_graph

   with create_graph_file("/path/to/graphs.h5", config={"kind": "mc"}) as handle:
       write_graph(handle, 0, graph_event)

``graph_event`` is the ``GraphEvent`` returned by ``event_graph.build_graph_event``.
