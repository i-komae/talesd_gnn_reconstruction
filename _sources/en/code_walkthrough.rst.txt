Code walkthrough
================

This page explains which source files and functions are used during HDF5 graph export, training, prediction, diagnostics, and feature analysis.
It is intended as a reading guide for the code, not only as a command reference.

Top-level entry point
---------------------

The command-line entry point is defined in ``pyproject.toml``.

.. code-block:: text

   talesd-gnn -> talesd_gnn_reconstruction.cli:main

``src/talesd_gnn_reconstruction/cli.py`` parses command-line options and dispatches them to the Python API.

.. list-table::
   :header-rows: 1

   * - Workflow
     - CLI function
     - Main implementation
   * - HDF5 graph export
     - ``cli._cmd_export`` : ``cli.py:1227``
     - ``dst_reader.iter_dst_banks``, ``event_graph.build_graph_event``, ``graph_io.write_graph``
   * - Training
     - ``cli._cmd_train`` : ``cli.py:1450``
     - ``train.train_model`` : ``train.py:1173``
   * - Prediction CSV
     - ``cli._cmd_predict`` : ``cli.py:1551``
     - ``predict.predict_graphs``
   * - Input distributions
     - ``cli._cmd_input_distributions`` : ``cli.py:1566``
     - ``feature_analysis.save_input_distributions`` : ``feature_analysis.py:133``
   * - Feature-group importance
     - ``cli._cmd_feature_importance`` : ``cli.py:1581``
     - ``feature_analysis.save_feature_group_importance`` : ``feature_analysis.py:352``
   * - Graph visualization
     - ``cli._cmd_visualize``
     - ``visualize.py``

HDF5 graph export
-----------------

Graph export is split into three layers: DST reading, event-to-graph conversion, and HDF5 writing.

.. code-block:: text

   talesd-gnn export
     -> cli._cmd_export()
       -> dst_reader.iter_dst_banks()
       -> event_graph.build_graph_event()
       -> graph_io.create_graph_file()
       -> graph_io.write_graph()

DST reading
~~~~~~~~~~~

``src/talesd_gnn_reconstruction/dst_reader.py`` streams DST banks.
The main entry point is ``iter_dst_banks`` : ``dst_reader.py:222``.

It opens DST files through ``dstio``, applies filters such as ``kind``, ``max_events``, trigger mode, source indices, and event date, and yields ``BankRecord`` objects.
For MC input, ``rusdraw``-style information is converted into the TALE-SD calibev-like representation used by the graph builder.

At this stage no graph nodes or edges have been created yet. The function only streams event-level records to the next layer.

GraphEvent creation
~~~~~~~~~~~~~~~~~~~

``src/talesd_gnn_reconstruction/event_graph.py`` converts one DST event into one GNN graph.
The main entry point is ``build_graph_event`` : ``event_graph.py:688``.

The function does the following:

1. ``_combine_sub_waveforms`` : ``event_graph.py:159`` merges waveform segments.
2. ``_extract_hit`` : ``event_graph.py:293`` extracts detector hits, pulses, FADC waveforms, timing, and charge.
3. ``_merge_hits_by_lid`` : ``event_graph.py:376`` merges records with the same detector ID.
4. ``_build_node_features`` : ``event_graph.py:471`` treats accepted pulses as nodes and builds node, pulse, and waveform features.
5. ``_build_edges`` : ``event_graph.py:610`` builds directed edges and edge features between detectors.
6. ``_target_from_sim`` : ``event_graph.py:419`` builds reconstruction targets from MC truth.
7. ``_particle_label_from_sim`` : ``event_graph.py:460`` builds the mass-classification label.
8. A ``GraphEvent`` object is returned.

Main fields in ``GraphEvent`` : ``event_graph.py:41``:

- ``node_features``: detector position, barycenter-relative position, accepted-pulse arrival time, the pulse's own charge, summed accepted-pulse charge in the same detector, accepted-pulse count, waveform length, FADC peak, and related quantities.
- ``edge_index``: source and destination node indices for directed edges.
- ``edge_features``: distance, time difference, spatial weight, causal direction, signal weight, and normalization terms.
- ``pulse_features``: the ``node_index`` mapping for the corresponding accepted-pulse node. Additional pulse-level scalar inputs are disabled in the current schema.
- ``waveform_features``: waveform inputs for the waveform encoder; currently upper/lower rise-aligned raw windows plus accepted-pulse masks.
- ``target``: MC truth targets ``log10_energy_eV, core_x_km, core_y_km, dir_x, dir_y, dir_z``.
  The core target is two-dimensional on the ground plane; the arrival direction is stored as a three-component unit vector.
- ``particle_label``: label for mass classification.
- ``metadata``: ``source_path``, ``source_index``, part type, and related split/diagnostic metadata.

Node and edge features
~~~~~~~~~~~~~~~~~~~~~~

``_build_node_features`` creates one node per accepted pulse.
The same detector ID may appear multiple times if a detector contributes multiple pulses.
The node columns deliberately separate pulse-level quantities, detector-level accepted-pulse aggregates, and event-context quantities.
Examples are ``log10_pulse_rho`` for the pulse itself, ``log10_detector_sum_pulse_rho`` for the accepted-pulse charge sum in the same detector, and ``dx_from_signal_bary_km`` for the event context.
The column definitions are exposed by ``graph_columns`` : ``event_graph.py:755`` and are written into the HDF5 file.
Because the physical meaning of several columns has changed, old HDF5 files are not silently migrated to the current schema; they must be re-exported.

``_build_edges`` scans node pairs from different detectors, removes pairs outside the distance and timing cuts, and writes bidirectional directed edges.
Changing edge features affects the HDF5 schema, ``edge_feature_dim``, and checkpoint compatibility.

HDF5 schema
~~~~~~~~~~~

``src/talesd_gnn_reconstruction/graph_io.py`` defines the HDF5 schema.

``create_graph_file`` : ``graph_io.py:14`` writes root attributes such as ``format=talesd_gnn_graphs``, ``format_version``, column definitions, and config metadata.
It also creates ``events`` and ``metadata`` groups.

``write_graph`` : ``graph_io.py:53`` writes one graph under ``events/%08d``.
It stores ``node_features``, ``node_positions_km``, ``node_lids``, ``edge_index``, ``edge_features``, ``pulse_features``, ``waveform_features``, optional ``target``, optional ``particle_label``, and per-event metadata.

The metadata is used later by source-aware train/validation/test splitting.

Energy-flat sampling and shards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Energy-flat export logic lives in ``cli.py`` around the export command.

- ``_scan_energy_candidates_for_file`` : ``cli.py:195`` scans candidates in each file.
- ``_add_energy_sample`` : ``cli.py:1075`` adds candidates to per-bin reservoirs.
- ``_sampled_graphs_from_reservoirs`` : ``cli.py:1101`` materializes the sampled event plan.
- ``_write_selected_graph_shard`` : ``cli.py:439`` writes selected events to a shard.
- ``_write_ordered_selected_graph_shard`` : ``cli.py:577`` writes events using a shuffled write plan.

The shuffled object is the lightweight selection/write plan, not fully materialized graph objects.
This avoids reading and storing all graph contents before writing.

Dataset and batching
--------------------

``src/talesd_gnn_reconstruction/dataset.py`` adapts HDF5 graphs to PyTorch DataLoaders.

``H5GraphDataset`` : ``dataset.py:181`` expands file or directory inputs into HDF5 shards, validates schema, reads column definitions, applies particle filtering and max-graph limits, builds global-to-local index mapping, and checks target/metadata availability.
The progress label ``initialize graph shards`` corresponds to this initialization step.

``__getitem__`` reads one event group and returns NumPy arrays for node features, edge index, edge features, pulse-to-node index mapping, waveform features, target, attributes, and optional detector IDs or particle labels.
It does not return torch tensors yet.

``fit_scalers`` : ``dataset.py:549`` fits feature and target scalers using the training split only.
This avoids leaking validation/test statistics into the training normalization.

``collate_graph_arrays`` : ``dataset.py:892`` merges graph samples into one batch.
It applies scalers, concatenates node arrays, remaps edge indices by node offsets, builds the graph ``batch`` vector, merges pulse/waveform arrays, and prepares target and mass-label arrays.
The wrapper ``collate_graphs`` converts the arrays into torch tensors.

Train/validation/test split
---------------------------

Splitting is implemented in ``train.py``.

.. list-table::
   :header-rows: 1

   * - Split mode
     - Function
     - Behavior
   * - ``event``
     - ``split_indices`` : ``train.py:167``
     - Random event-index split.
   * - ``source-path``
     - ``split_indices_by_source_path`` : ``train.py:194``
     - Splits by ``source_path``.
   * - ``source-stratified``
     - ``split_indices_by_stratified_source_path`` : ``train.py:489``
     - Keeps the same ``source_path`` out of multiple splits and also tries to reduce particle/energy imbalance.

The current default is ``source-stratified`` with validation fraction 0.05 and test fraction 0.10.
Therefore the effective fractions are 0.85 train, 0.05 validation, and 0.10 test.

Training
--------

``train_model`` : ``train.py:1173`` is the central training function.
CLI commands, submitter scripts, and notebooks ultimately call this function.

Its main stages are:

1. Import torch and model modules.
2. Validate training task, loss mode, mass options, and diagnostic options.
3. Resolve the target device.
4. Construct ``H5GraphDataset``.
5. Build train/validation/test splits.
6. Fit scalers on training indices.
7. Infer input dimensions from the first sample.
8. Construct ``TaleSdGNN`` or ``PhysicsTaleSdGNN``.
9. Build optimizer, learning-rate scheduler, and DataLoaders.
10. Run the epoch loop.
11. Save a best checkpoint when validation loss improves.
12. Reload the best checkpoint and run validation/test prediction plus diagnostics.

The ``stage_seconds`` log maps directly onto these phases.

Epoch loop
~~~~~~~~~~

During training, the model is put in ``train`` mode, a batch is moved to the device, ``pred_all = model(batch)`` is evaluated, the output tensor is split into active heads, losses are computed, and optimizer steps are applied.

During validation, the same losses are computed under ``model.eval()`` and ``torch.no_grad()``.
Validation loss controls best-checkpoint selection and early stopping.

Checkpoint contents
~~~~~~~~~~~~~~~~~~~

The checkpoint stores more than model weights.
It also stores model configuration, scalers, history, metrics, split indices, runtime information, and diagnostics metadata.
Prediction and feature-importance runs rely on these saved scalers and split indices.

Model
-----

The current main model is ``PhysicsTaleSdGNN`` : ``model.py:718``.

Inputs include node features, edge index, edge features, graph batch IDs, waveform features, and optionally detector IDs.
The model combines a node encoder, waveform encoder, optional detector embedding, edge-aware message passing, graph-level readout, and task-specific heads. The pulse encoder is inactive in the current schema because ``pulse_features`` contains only ``node_index``.

Main heads:

- Reconstruction head: energy, core, and direction.
- Mass classification head: mass logit.
- Quality head: quality logit.
- Error head: raw error-prediction values.

Which heads are active is controlled by ``training_task``, ``mass_classification``, ``quality_prediction``, and ``error_prediction``.

Loss functions
--------------

Loss definitions are in ``train.py``.

``_reconstruction_loss`` : ``train.py`` computes reconstruction loss.
For ``loss_mode=physics``, energy, core, and direction are evaluated in physical units.
Energy uses SmoothL1 on log-energy difference, core uses SmoothL1 after division by ``core_loss_scale_km``, and direction uses angular error.
When enabled, the training wrapper also adds true-energy-bin bias penalties: one on the mean ``pred_logE - true_logE`` in each bin, and one on the proton/iron mean-residual difference in the same bin.

``_quality_targets_from_reconstruction`` : ``train.py`` converts reconstruction errors into a soft quality target.
``_quality_prediction_loss`` : ``train.py`` applies BCEWithLogits between the quality logit and that soft target.
This target is not the mass label.

``_mass_classification_loss`` : ``train.py`` computes mass-only or mass-head loss.
It supports BCE, focal loss, and optional ranking loss.

Diagnostics
-----------

``src/talesd_gnn_reconstruction/diagnostics.py`` produces learning curves, reconstruction resolution plots, quality-cut plots, mass ROC/confusion/score plots, and JSON summaries.
The main entry point is ``save_training_diagnostics`` : ``diagnostics.py:2463``.

Training reloads the best validation checkpoint before producing final validation/test metrics.
The final diagnostics therefore correspond to the best validation epoch, not necessarily the last epoch.

Input distributions and feature importance
------------------------------------------

``feature_analysis.save_input_distributions`` : ``feature_analysis.py:133`` scans HDF5 graphs and writes feature distribution PDFs and JSON.

``feature_analysis.save_feature_group_importance`` : ``feature_analysis.py:352`` restores the checkpoint, model config, scalers, and split indices, computes baseline metrics, replaces or zeros one feature group at a time, recomputes metrics, and reports the degradation.
This is a post-hoc ablation of the trained model, not a full retraining experiment.

Where to read when changing code
--------------------------------

.. list-table::
   :header-rows: 1

   * - Change
     - Start here
     - Notes
   * - Node features
     - ``event_graph._build_node_features``, ``event_graph.graph_columns``
     - Changes HDF5 schema and checkpoint compatibility.
   * - Edge features
     - ``event_graph._build_edges``
     - Changes ``edge_feature_dim``.
   * - Waveform handling
     - ``event_graph`` waveform extraction, ``model.WaveformEncoder``
     - Also check collated waveform shapes.
   * - Split logic
     - ``train.split_indices*``
     - Always record split mode when comparing runs.
   * - Loss logic
     - ``train._reconstruction_loss``, ``train._quality_prediction_loss``, ``train._mass_classification_loss``
     - Run a small forward/loss smoke test before full training.
   * - Model heads
     - ``model.PhysicsTaleSdGNN``
     - Also check output splitting and loss code.
   * - Diagnostics
     - ``diagnostics.save_training_diagnostics`` and nearby plot functions
     - Affects evaluation interpretation, not training updates.
   * - Slurm submission
     - ``scripts/submit_server_*``
     - Do not accidentally change graph, task, loss, or epoch settings.

Recommended reading order
-------------------------

1. Read ``cli.py`` ``build_parser`` and ``_cmd_train``.
2. Read ``train.py`` ``train_model`` in stage order.
3. Read ``dataset.py`` ``H5GraphDataset`` and ``collate_graph_arrays``.
4. Read ``model.py`` ``PhysicsTaleSdGNN``.
5. Read loss functions in ``train.py``.
6. Read ``dst_reader.py``, ``event_graph.py``, and ``graph_io.py`` for HDF5 export.
7. Read ``diagnostics.py`` and ``feature_analysis.py`` for evaluation and interpretation.
