Code Reading Map
================

This page maps the execution workflow to the source files. Detailed model concepts and equations are described in :doc:`code_walkthrough`.

Recommended reading order
-------------------------

For production runs, read the code in this order:

.. code-block:: text

   DST
     -> talesd-gnn export
       -> HDF5 graph
         -> H5GraphDataset
           -> train_model
             -> PhysicsTaleSdGNN
               -> loss / metrics / diagnostics

.. list-table::
   :header-rows: 1

   * - Stage
     - Main files
     - What is decided there
   * - CLI entry point
     - ``src/talesd_gnn_reconstruction/cli.py``
     - Converts command-line arguments into Python API calls.
   * - DST reading
     - ``src/talesd_gnn_reconstruction/dst_reader.py``
     - Iterates DST banks as Python records.
   * - Event graph creation
     - ``src/talesd_gnn_reconstruction/event_graph.py``
     - Builds accepted-pulse nodes, edges, waveforms, targets, and metadata.
   * - HDF5 writing
     - ``src/talesd_gnn_reconstruction/graph_io.py``
     - Writes graph shards and stores schema attributes.
   * - HDF5 loading
     - ``src/talesd_gnn_reconstruction/dataset.py``
     - Presents many HDF5 shards as one dataset for DataLoader.
   * - Training workflow
     - ``src/talesd_gnn_reconstruction/train.py``
     - Handles split, scalers, DataLoader, training loop, losses, checkpoints, and evaluation.
   * - Model
     - ``src/talesd_gnn_reconstruction/model.py``
     - Embeds node, edge, and waveform inputs, applies message passing, and produces heads.
   * - Diagnostics
     - ``src/talesd_gnn_reconstruction/diagnostics.py``
     - Writes learning curves, energy dependence plots, quality-cut plots, and mass diagnostics.
   * - Feature analysis
     - ``src/talesd_gnn_reconstruction/feature_analysis.py``
     - Writes input distributions and post-hoc group-ablation importance.

CLI as a thin wrapper
---------------------

The ``talesd-gnn`` commands are mostly wrappers. ``cli.py`` parses arguments and calls the implementation functions.

.. list-table::
   :header-rows: 1

   * - Command
     - Function in ``cli.py``
     - Main implementation
   * - ``export``
     - ``_cmd_export``
     - Reads DST records, builds graphs in ``event_graph``, and writes HDF5 files in ``graph_io``.
   * - ``train``
     - ``_cmd_train``
     - Calls ``train.train_model``.
   * - ``predict``
     - ``_cmd_predict``
     - Calls ``predict.predict_graphs``.
   * - ``input-distributions``
     - ``_cmd_input_distributions``
     - Calls ``feature_analysis.save_input_distributions``.
   * - ``feature-importance``
     - ``_cmd_feature_importance``
     - Calls ``feature_analysis.save_feature_group_importance``.
   * - ``visualize``
     - ``_cmd_visualize``
     - Calls ``visualize.visualize_graphs``.

HDF5 graph export
-----------------

``talesd-gnn export`` creates training HDF5 graphs from DST files:

.. code-block:: text

   cli._cmd_export()
     -> dst_reader.iter_dst_banks()
     -> event_graph.build_graph_event()
     -> graph_io.create_graph_file() / graph_io.write_graph()

``event_graph.build_graph_event`` returns a ``GraphEvent`` object for one event.

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
     - Later use
   * - ``node_features``
     - Scalar features for accepted-pulse nodes.
     - Input to the node encoder.
   * - ``edge_index``
     - Directed graph connectivity.
     - Determines message-passing edges.
   * - ``edge_features``
     - Pulse, detector, timing, geometry, and signal relations.
     - Input to the edge encoder and message layers.
   * - ``waveform_features``
     - Rise-aligned raw waveform windows and accepted masks.
     - Input to the waveform encoder.
   * - ``target``
     - Supervised MC truth values.
     - Used by reconstruction losses and metrics.
   * - ``metadata``
     - Source path, particle label, event id, and related fields.
     - Used for split, diagnostics, and mass labels.

HDF5 files also store schema metadata such as ``columns_json`` and ``waveform_schema``.
Files with incompatible schemas should be re-exported instead of silently reused.

Dataset loading
---------------

``H5GraphDataset`` treats one HDF5 file or many shards as one dataset. Initialization checks shard lengths, metadata availability, and schema compatibility. Graph arrays are loaded lazily in ``__getitem__``.

.. code-block:: text

   H5GraphDataset.__init__()
     -> resolve shard list
     -> check columns_json / waveform_schema
     -> build cumulative lengths

   H5GraphDataset.__getitem__(index)
     -> map global index to shard/local index
     -> read node/edge/waveform/target/metadata
     -> return a sample dict for DataLoader

``max_open_files`` limits the number of simultaneously opened HDF5 files and is relevant for large sharded datasets.

Split and scalers
-----------------

``train_model`` creates the dataset and then splits indices into train, validation, and test.
The production setting uses ``source-stratified`` splitting.
The split key is a source group, not always the raw ``source_path``.
For split CORSIKA DST files named like ``DAT??????_gea_trg_XXX.dst.gz``, the ``XXX`` chunks are grouped by the common ``DAT??????`` shower id so that one shower cannot cross split boundaries.

Scalers are fit only on the training split:

.. code-block:: text

   train_model()
     -> H5GraphDataset(...)
     -> create split indices
     -> fit_scalers(dataset, train_indices)
     -> create DataLoaders

This prevents validation and test information from entering the normalization constants.

Model inputs and outputs
------------------------

The current physics model is ``PhysicsTaleSdGNN``. It receives a batch dictionary with node, edge, waveform, batch-index, target, and metadata tensors.

``model.py`` embeds node scalars, edge scalars, and waveform tensors separately. It then applies multiple ``GatedEdgeMessageLayer`` layers and a graph-level readout. Output heads depend on the task settings.

.. list-table::
   :header-rows: 1

   * - Head
     - Condition
     - Output
   * - Reconstruction
     - ``training_task=reconstruction``
     - Energy, core x/y, and direction vector.
   * - Quality
     - ``quality_prediction=True``
     - Reconstruction-quality target for the event.
   * - Mass
     - ``mass_classification=True`` or ``training_task=mass``
     - Iron/proton classification logit.
   * - Error
     - ``error_prediction=True``
     - Auxiliary error estimate. It is off in the current main setting.

Losses and evaluation
---------------------

The model forward pass and loss calculation are separate. The model returns predictions; ``train.py`` compares them with targets.

.. list-table::
   :header-rows: 1

   * - Process
     - Main functions
     - Role
   * - Reconstruction loss
     - ``_reconstruction_loss`` / ``_reconstruction_training_loss``
     - Supervised energy, core, and direction loss.
   * - Angular loss
     - ``_angular_loss_from_vectors``
     - Uses the angular distance between direction vectors.
   * - Energy-bias penalty
     - ``_energy_bin_bias_loss`` / ``_energy_particle_bias_loss``
     - Penalizes mean energy bias by true-energy bin and particle species.
   * - Quality loss
     - ``_quality_prediction_loss``
     - Learns a quality target derived from reconstruction errors.
   * - Mass loss
     - ``_mass_classification_loss``
     - Computes BCE, focal, and ranking terms according to settings.

After training, the best validation checkpoint is loaded again for validation/test prediction. Metrics and plots are written by ``metrics.py`` and ``diagnostics.py``.

Server submitters
-----------------

On Slurm, use ``scripts/submit_server_*.sh`` instead of invoking ``talesd-gnn train`` directly. Submitters collect environment variables and configure resources, run directories, local graph cache, and runtime copy.

.. code-block:: text

   submit_server_reco_mass_training.sh
     -> set task-specific environment variables
     -> submit_server_waveform_full_training.sh
       -> choose Slurm resources
       -> prepare local graph/runtime cache
       -> run train_large_existing_graphs.sh
         -> talesd-gnn train

Task-specific submitters such as ``submit_server_reco_mass_training.sh`` and ``submit_server_mass_only_training.sh`` set defaults before calling the common training submitter. For comparisons, inspect the final run configuration rather than only the wrapper script name.

Diagnostics and post-processing
-------------------------------

During training, ``save_learning_progress`` updates the learning curve. If enabled, best-validation diagnostics are also updated when the best epoch changes. After training, ``save_training_diagnostics`` writes validation/test figures and ``summary.json``.

Input distributions and feature importance are handled by ``feature_analysis.py``. Feature importance is post-hoc ablation: it replaces input groups and measures metric changes without retraining.
