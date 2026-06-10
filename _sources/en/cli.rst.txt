CLI API
=======

The command-line entry point is ``talesd-gnn``. In ``pyproject.toml`` it maps to
``talesd_gnn_reconstruction.cli:main``.
The implementation map is described in :doc:`code_map`.
This page focuses on the commands, their inputs, their outputs, and how the outputs are used later.

HDF5 graph export
-----------------

Read DST files and write GNN-ready HDF5 graph shards.
The exported HDF5 files are the common input for training, input-distribution checks, visualization, and feature importance.
For MC training, they contain targets and particle labels.

.. code-block:: bash

   .venv/bin/talesd-gnn export \
     --input-dir /path/to/mc_dst \
     --kind mc \
     --const-dst /path/to/talesdconst_pass2.dst \
     --mc-calib-dir /path/to/tale_mc_calib \
     --energy-sample-per-bin 50000 \
     --energy-bin-width 0.1 \
     --workers 32 \
     --shard-size 50000 \
     --skip-errors \
     -o /path/to/graphs/flat50000/flat50000.h5

Implementation path:

.. code-block:: text

   talesd-gnn export
     -> cli._cmd_export()
       -> dst_reader.py
       -> event_graph.py
       -> graph_io.py

Heterogeneous HDF5 graph export
-------------------------------

``export-hetero`` writes the newer ``dstio.tale.graph`` schema.
The output is a training/cache format, not a mandatory step for one-pass reconstruction.
The graph contains detector nodes, pulse nodes, detector-level waveforms, and typed relations.
For MC training, ``--require-reference-core`` is the normal setting because core-relative pulse features are valid only when the Ising reference core exists.
For balanced large MC datasets, add ``--energy-sample-per-bin`` with ``--energy-sample-stratify-particle``.
In that mode, the command delegates source manifest scans, ``DAT??????`` source-group management, zenith-stratified source allocation, deterministic random event-index selection, graphable-event refill, graph construction, and HDF5 writing to ``dstio.tale.graph.write_balanced_graph_h5``.
The GNN repository does not add, remove, or redefine graph edges during export.

.. code-block:: bash

   .venv/bin/talesd-gnn export-hetero \
     --input-dir /path/to/mc_dst \
     --kind mc \
     --const-dst /path/to/talesdconst_pass2.dst \
     --mc-calib-dir /path/to/tale_mc_calib \
     --require-reference-core \
     --skip-errors \
     --skip-missing-mc-calibration \
     --energy-sample-per-bin 50000 \
     --energy-sample-stratify-particle \
     --selection-summary /path/to/graphs/hetero/summaries/hetero_selection_summary.json \
     --workers 32 \
     --scan-workers 32 \
     --selection-workers 1 \
     --h5-backend auto \
     --write-block-size 2048 \
     --shard-size 100000 \
     -o /path/to/graphs/hetero

Implementation path:

.. code-block:: text

   talesd-gnn export-hetero
     -> cli._cmd_export_hetero()
       -> dstio.tale.graph.write_balanced_graph_h5()  # with --energy-sample-per-bin
          or dstio.tale.graph.write_graph_h5()        # without balanced selection
       -> dstio-owned GraphEvent build and HDF5 shard write

Training
--------

Train from existing HDF5 graphs.
``train`` does not modify the graph files. It writes split-dependent scalers, checkpoints, metrics, and diagnostics under the run output.

.. code-block:: bash

   .venv/bin/talesd-gnn train \
     --graphs /path/to/graphs/flat50000 \
     -o /path/to/output/checkpoints/reconstruction.pt \
     --training-task reconstruction \
     --model-architecture physics \
     --waveform-encoder cnn-gru \
     --loss-mode physics \
     --quality-prediction \
     --split-mode source-stratified \
     --val-fraction 0.05 \
     --test-fraction 0.10 \
     --epochs 128

Implementation path:

.. code-block:: text

   talesd-gnn train
     -> cli._cmd_train()
       -> train.train_model()
         -> dataset.H5GraphDataset
         -> model.PhysicsTaleSdGNN
         -> loss functions in train.py
         -> metrics.py / diagnostics.py

Heterogeneous training
----------------------

``train-hetero`` trains from HDF5 files written by ``export-hetero``.
The training dataset converts each event sample to PyG ``HeteroData`` with
``hetero_data.sample_to_hetero_data`` so ``torch_geometric.loader.DataLoader``
can batch variable-size detector/pulse graphs. The model then converts the
batched ``HeteroData`` back to the repository's explicit tensor dictionary with
``hetero_data.hetero_data_to_tensors``.

.. code-block:: bash

   .venv/bin/talesd-gnn train-hetero \
     --graphs /path/to/graphs/hetero \
     -o /path/to/output/checkpoints/hetero_reco_mass.pt \
     --epochs 128 \
     --batch-size 128 \
     --model-architecture hetero_attention \
     --waveform-encoder transformer \
     --loss-mode physics \
     --mass-classification \
     --split-mode source-stratified \
     --val-fraction 0.05 \
     --test-fraction 0.10 \
     --diagnostics

Implementation path:

.. code-block:: text

   talesd-gnn train-hetero
     -> cli._cmd_train_hetero()
       -> hetero_training.train_hetero_model()
         -> hetero_graph_io.H5HeteroGraphDataset
         -> hetero_data.sample_to_hetero_data()
         -> hetero_model.MinimalHeteroTaleSdGNN
         -> metrics.py / diagnostics.py

For the current planned heterogeneous comparison, use ``WAVEFORM_ENCODER=transformer`` first.
The ``cnn-gru`` waveform encoder is a later ablation under the selected condition, not a simultaneous six-job sweep.
If a relation-definition ablation is needed, request or implement that in
``dstio`` and export a separate HDF5 graph dataset. ``train-hetero`` does not
edit graph connectivity.

Prediction
----------

Create a CSV from a trained checkpoint.
For MC graphs, truth columns can be included unless ``--no-truth`` is used. For data graphs, prediction-only output is the normal use.

.. code-block:: bash

   .venv/bin/talesd-gnn predict \
     --graphs /path/to/graphs/data_graphs.h5 \
     --checkpoint /path/to/checkpoints/reconstruction.pt \
     -o /path/to/predictions.csv

Direct DST reconstruction
-------------------------

``reconstruct-dst`` uses a heterogeneous checkpoint and reads DST files directly.
It does not require an intermediate HDF5 graph.
It uses the same ``dstio.tale.graph`` schema and the same checkpoint scalers as ``train-hetero``.

.. code-block:: bash

   .venv/bin/talesd-gnn reconstruct-dst \
     --input-dir /path/to/data_or_mc_dst \
     --kind auto \
     --checkpoint /path/to/checkpoints/hetero_reco_mass.pt \
     --const-dst /path/to/talesdconst_pass2.dst \
     --mc-calib-dir /path/to/tale_mc_calib \
     --batch-size 256 \
     --skip-errors \
     -o /path/to/reconstruction.csv

Implementation path:

.. code-block:: text

   talesd-gnn reconstruct-dst
     -> cli._cmd_reconstruct_dst()
       -> hetero_predict.reconstruct_dst()
         -> dstio.tale.graph.iter_graphs()
         -> hetero_data.sample_to_hetero_data()
         -> hetero_model.MinimalHeteroTaleSdGNN

Input distributions
-------------------

Save input feature distributions as PDF and JSON.
This is a dataset diagnostic used to inspect HDF5 content and split-dependent biases.
It is normally tied to the HDF5 dataset, not to a particular training run.
The command also writes redraw artifacts next to the PDFs:
``input_feature_sample_values.npz`` stores the sampled values used for plotting,
and ``input_feature_sample_values_manifest.json`` maps each NPZ array to its
feature group and column name.

.. code-block:: bash

   .venv/bin/talesd-gnn input-distributions \
     --graphs /path/to/graphs/flat50000 \
     -o /path/to/input_distributions \
     --max-graphs 100000

Feature importance
------------------

Run feature group ablation against a trained checkpoint.
This is post-hoc analysis, not retraining. It replaces input groups such as node, edge, or waveform features and measures metric degradation.
The output JSON contains the baseline metrics, per-group ablated metrics, and
deltas. ``feature_group_importance_plot_data.json`` contains the values needed
to redraw the bar plots without rerunning inference.
Here, a metric is not a single generic score. Reconstruction metrics evaluate
the physical outputs after inverse scaling: energy error
(``rmse_log10_energy`` and relative-energy metrics), core displacement in the
ground-plane xy coordinates (``core_68_km`` and related metrics), and arrival
direction opening angle (``angular_68_deg`` and related metrics). Mass metrics
are stored separately as accuracy, balanced accuracy, AUC, and confusion counts.
``baseline`` is the normal-input evaluation, each group row stores the
``ablated`` evaluation, and ``*_delta`` is ``ablated - baseline`` for every
finite numeric metric.

.. code-block:: bash

   .venv/bin/talesd-gnn feature-importance \
     --graphs /path/to/graphs/flat50000 \
     --checkpoint /path/to/checkpoints/reconstruction.pt \
     -o /path/to/feature_importance \
     --split validation \
     --max-graphs 50000

For heterogeneous checkpoints, the same command dispatches to
``hetero_feature_analysis.save_hetero_feature_group_importance``.
The default groups separate detector signal, detector geometry, detector readout context,
pulse timing/signal, Ising pulse annotations, detector waveforms, and typed edge groups.

Attention maps
--------------

For ``hetero_attention`` checkpoints, save post-hoc attention weights for a
small set of events:

.. code-block:: bash

   .venv/bin/talesd-gnn attention-maps \
     --graphs /path/to/graphs/hetero_balanced_flat10000 \
     --checkpoint /path/to/checkpoints/hetero_reco_mass.pt \
     -o /path/to/attention_maps \
     --split validation \
     --max-graphs 16

The command writes ``attention_maps.json`` and ``attention_maps.npz``. The JSON
keeps event identity, source path, prediction/target values, and the names of
the saved arrays. The NPZ keeps detector and pulse lids, positions,
``pulse_detector_index``, ``pulse_bounds``, relation attention for each
layer/relation/head, and detector/pulse readout attention weights.

These values are useful for event-display overlays and sanity checks. They are
not feature importance by themselves; use ablation or perturbation diagnostics
to test whether a feature group is necessary.

Visualization
-------------

Render an HDF5 graph as an event-display PDF.
Use this to inspect graph schema, node/edge construction, and waveform-mask behavior visually.

.. code-block:: bash

   .venv/bin/talesd-gnn visualize \
     --graphs /path/to/graphs/flat50000 \
     --index 0 \
     -o /path/to/graph_000000.pdf
