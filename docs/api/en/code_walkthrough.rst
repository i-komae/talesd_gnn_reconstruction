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
     - ``train.train_model`` : ``train.py:1385``
   * - Prediction CSV
     - ``cli._cmd_predict`` : ``cli.py:1557``
     - ``predict.predict_graphs``
   * - Input distributions
     - ``cli._cmd_input_distributions`` : ``cli.py:1572``
     - ``feature_analysis.save_input_distributions`` : ``feature_analysis.py:139``
   * - Feature-group importance
     - ``cli._cmd_feature_importance`` : ``cli.py:1587``
     - ``feature_analysis.save_feature_group_importance`` : ``feature_analysis.py:369``
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
The main entry point is ``build_graph_event`` : ``event_graph.py:697``.

The function does the following:

1. ``_combine_sub_waveforms`` : ``event_graph.py:159`` merges waveform segments.
2. ``_extract_hit`` : ``event_graph.py:298`` extracts detector hits, pulses, FADC waveforms, timing, and charge.
3. ``_merge_hits_by_lid`` : ``event_graph.py:381`` merges records with the same detector ID.
4. ``_build_node_features`` : ``event_graph.py:474`` treats accepted pulses as nodes and builds node, pulse, and waveform features.
5. ``_build_edges`` : ``event_graph.py:619`` builds directed edges and edge features between detectors.
6. ``_target_from_sim`` : ``event_graph.py:424`` builds reconstruction targets from MC truth.
7. ``_particle_label_from_sim`` : ``event_graph.py:463`` builds the mass-classification label.
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
The column definitions are exposed by ``graph_columns`` : ``event_graph.py:764`` and are written into the HDF5 file.
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

``H5GraphDataset`` : ``dataset.py:202`` expands file or directory inputs into HDF5 shards, validates schema, reads column definitions, applies particle filtering and max-graph limits, builds global-to-local index mapping, and checks target/metadata availability.
The progress label ``initialize graph shards`` corresponds to this initialization step.

``__getitem__`` reads one event group and returns NumPy arrays for node features, edge index, edge features, pulse-to-node index mapping, waveform features, target, attributes, and optional detector IDs or particle labels.
It does not return torch tensors yet.

``fit_scalers`` : ``dataset.py:599`` fits feature and target scalers using the training split only.
This avoids leaking validation/test statistics into the training normalization.

``collate_graph_arrays`` : ``dataset.py:942`` merges graph samples into one batch.
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
     - ``split_indices`` : ``train.py:173``
     - Random event-index split.
   * - ``source-path``
     - ``split_indices_by_source_path`` : ``train.py:200``
     - Splits by ``source_path``.
   * - ``source-stratified``
     - ``split_indices_by_stratified_source_path`` : ``train.py:564``
     - Keeps the same ``source_path`` out of multiple splits and also tries to reduce particle/energy imbalance.

The current default is ``source-stratified`` with validation fraction 0.05 and test fraction 0.10.
Therefore the effective fractions are 0.85 train, 0.05 validation, and 0.10 test.

Training
--------

``train_model`` : ``train.py:1385`` is the central training function.
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

The current main model is ``PhysicsTaleSdGNN`` : ``model.py:730``.

Inputs include node features, edge index, edge features, graph batch IDs, waveform features, and optionally detector IDs.
The model combines a node encoder, waveform encoder, optional detector embedding, edge-aware message passing, graph-level readout, and task-specific heads. The pulse encoder is inactive in the current schema because ``pulse_features`` contains only ``node_index``.

What the GNN approximates
~~~~~~~~~~~~~~~~~~~~~~~~~

The model learns a graph-level function

.. math::

   f_{\theta}(G)
   =
   (\log_{10}E,\ x_c,\ y_c,\ n_x,\ n_y,\ n_z,\ \ldots),
   \qquad
   G=(V,E).

``V`` is the set of accepted-pulse nodes and ``E`` is the set of directed edges between pulses on different detectors.
The node order has no physical meaning, so the model must depend on pulse features and pairwise relations rather than on an arbitrary array order.
This is why the implementation uses message passing followed by permutation-invariant graph readout.

Conceptually, the learned mapping is

.. math::

   \{x_i\}_{i\in V},\ \{e_{ij}\}_{(i,j)\in E}
   \longrightarrow
   \hat{y}_{\mathrm{event}}.

``x_i`` contains the accepted-pulse, detector, and event-context features for node ``i``.
``e_ij`` contains the distance, timing, signal-difference, and edge-weight features for the directed relation ``i -> j``.
The output is one prediction per event graph, not one prediction per node.

Relation to shower reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The network does not explicitly solve a plane fit or an LDF fit.
However, the inputs expose the information those procedures depend on:

.. list-table::
   :header-rows: 1

   * - Physical information
     - Code-level input
     - Main affected outputs
   * - Relative arrival timing
     - ``pulse_arrival_usec_rel``, edge ``dt_usec``, ``dt_per_km``
     - Direction, core
   * - Detector geometry
     - Node positions, barycenter-relative position, edge distance
     - Core, direction
   * - Signal size and lateral pattern
     - ``log10_pulse_rho``, detector sum/max rho, edge signal differences
     - Energy, core, mass
   * - Waveform shape
     - Raw waveform windows, accepted masks, waveform encoder output
     - Mass, quality, reconstruction support

Message passing lets each pulse representation depend on whether neighboring pulses are consistent in time, distance, and signal size.
A large charge at one detector is not enough by itself; the surrounding pattern tells the model whether it is close to the core, consistent with the shower front, or an outlying contribution.

Node, waveform, and edge encoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The initial node state is

.. math::

   h_i^{(0)}
   =
   \phi_{\mathrm{node}}([x_i,\ w_i,\ d_i,\ p_i]).

``x_i`` is the scalar node feature vector, ``w_i`` is the waveform embedding, ``d_i`` is the detector-ID embedding, and ``p_i`` is the pulse-scalar summary.
In the current default, detector embedding and pulse-scalar encoding are disabled, so the active inputs are mostly ``x_i`` and ``w_i``.

``WaveformEncoder`` compresses the four-channel waveform into a node embedding.
For ``cnn-gru``, the code applies Conv1D to local waveform shapes, then a GRU to summarize time ordering, and finally projects the recurrent last state together with mean and max pooled encoded features.
The current waveform schema keeps raw upper/lower windows and accepted-pulse masks on the same time axis, so the model can use both the raw waveform shape and the accepted-pulse region.

``EdgeTimeDeltaEncoder`` separately encodes

.. math::

   \tau_{ij}
   =
   (\Delta t_{ij},\ |\Delta t_{ij}|,\ \Delta t_{ij}/r_{ij})

and averages the encoded timing messages into each destination node before the main message-passing layers.
This keeps arrival-time information from being diluted among many other edge columns.

Message passing
~~~~~~~~~~~~~~~

``GatedEdgeMessageLayer`` builds a directed message from source node state, destination node state, and edge features:

.. math::

   m_{i\rightarrow j}
   =
   M_{\mathrm{msg}}([h_i,h_j,e_{ij}])
   \cdot
   \sigma(M_{\mathrm{gate}}([h_i,h_j,e_{ij}])).

The message MLP controls what information is sent.
The gate MLP controls how strongly that edge contributes.
The destination node state is included because the same source pulse can have a different meaning depending on the pulse that receives the message.

For each destination node, messages are aggregated by both mean and max.
Mean captures the typical neighborhood trend, while max preserves strong local evidence that would be diluted by averaging.
The update is residual and layer-normalized:

.. math::

   h_j' =
   \mathrm{LayerNorm}
   \left(
     h_j + M_{\mathrm{update}}([h_j,\overline{m}_j,m_j^{\max}])
   \right).

This layer is repeated five times in the standard configuration.
After repeated message passing, a node representation no longer describes only one pulse; it also contains information from nearby pulses connected through the edge graph.

Readout and heads
~~~~~~~~~~~~~~~~~

After message passing, the model still has node-level vectors.
Energy, core, direction, quality, and mass are event-level quantities, so the node vectors must be pooled into a graph vector.
The implementation concatenates mean pooling, max pooling, and multi-head attentive readout.

For attention head ``h``,

.. math::

   a_{ih}
   =
   \frac{\exp s_h(h_i)}
        {\sum_{k\in G}\exp s_h(h_k)},
   \qquad
   g_h
   =
   \sum_{i\in G} a_{ih} h_i.

Different heads can learn to emphasize different parts of the event.
These heads are not automatically labeled by physics meaning; interpretation requires feature importance or attention diagnostics.

Implementation excerpts
~~~~~~~~~~~~~~~~~~~~~~~

The following classes are the main model components described above.

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/model.py
   :language: python
   :pyobject: WaveformEncoder
   :linenos:

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/model.py
   :language: python
   :pyobject: EdgeTimeDeltaEncoder
   :linenos:

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/model.py
   :language: python
   :pyobject: GatedEdgeMessageLayer
   :linenos:

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/model.py
   :language: python
   :pyobject: AttentiveReadout
   :linenos:

Main heads:

- Reconstruction head: energy, core, and direction.
- Mass classification head: mass logit.
- Quality head: quality logit.
- Error head: raw error-prediction values.

Which heads are active is controlled by ``training_task``, ``mass_classification``, ``quality_prediction``, and ``error_prediction``.

Loss functions
--------------

Loss definitions are in ``train.py``.

Mathematical helpers
~~~~~~~~~~~~~~~~~~~~

``SmoothL1`` behaves like a quadratic loss for small residuals and like an absolute-error loss for large residuals:

.. math::

   \mathrm{SmoothL1}_{\beta}(r)
   =
   \begin{cases}
     \frac{1}{2}r^2/\beta, & |r| < \beta,\\
     |r|-\frac{1}{2}\beta, & |r| \ge \beta.
   \end{cases}

This limits the influence of large outliers while keeping a smooth local gradient near zero residual.

The mass and quality heads emit logits ``z`` rather than probabilities.
The sigmoid

.. math::

   \sigma(z)=\frac{1}{1+\exp(-z)}

is used only when converting a logit to a probability-like score.
``BCEWithLogits`` combines sigmoid and binary cross entropy in a numerically stable form:

.. math::

   \mathrm{BCEWithLogits}(z,y)
   =
   \mathrm{softplus}(z)-yz,
   \qquad
   \mathrm{softplus}(z)=\log(1+\exp z).

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

``feature_analysis.save_input_distributions`` : ``feature_analysis.py:139`` scans HDF5 graphs and writes feature distribution PDFs and JSON.

``feature_analysis.save_feature_group_importance`` : ``feature_analysis.py:369`` restores the checkpoint, model config, scalers, and split indices, computes baseline metrics, replaces or zeros one feature group at a time, recomputes metrics, and reports the degradation.
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
