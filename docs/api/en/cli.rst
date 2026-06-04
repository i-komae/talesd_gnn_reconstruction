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

Prediction
----------

Create a CSV from a trained checkpoint.
For MC graphs, truth columns can be included unless ``--no-truth`` is used. For data graphs, prediction-only output is the normal use.

.. code-block:: bash

   .venv/bin/talesd-gnn predict \
     --graphs /path/to/graphs/data_graphs.h5 \
     --checkpoint /path/to/checkpoints/reconstruction.pt \
     -o /path/to/predictions.csv

Input distributions
-------------------

Save input feature distributions as PDF and JSON.
This is a dataset diagnostic used to inspect HDF5 content and split-dependent biases.
It is normally tied to the HDF5 dataset, not to a particular training run.

.. code-block:: bash

   .venv/bin/talesd-gnn input-distributions \
     --graphs /path/to/graphs/flat50000 \
     -o /path/to/input_distributions \
     --max-graphs 100000

Feature importance
------------------

Run feature group ablation against a trained checkpoint.
This is post-hoc analysis, not retraining. It replaces input groups such as node, edge, or waveform features and measures metric degradation.

.. code-block:: bash

   .venv/bin/talesd-gnn feature-importance \
     --graphs /path/to/graphs/flat50000 \
     --checkpoint /path/to/checkpoints/reconstruction.pt \
     -o /path/to/feature_importance \
     --split validation \
     --max-graphs 50000

Visualization
-------------

Render an HDF5 graph as an event-display PDF.
Use this to inspect graph schema, node/edge construction, and waveform-mask behavior visually.

.. code-block:: bash

   .venv/bin/talesd-gnn visualize \
     --graphs /path/to/graphs/flat50000 \
     --index 0 \
     -o /path/to/graph_000000.pdf
