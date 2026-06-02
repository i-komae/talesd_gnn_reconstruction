CLI API
=======

CLIの入口は ``talesd-gnn`` です。``pyproject.toml`` では
``talesd_gnn_reconstruction.cli:main`` に対応しています。

HDF5 graph export
-----------------

DSTを読み、GNN用HDF5 graph shardを作成します。

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

実装の流れ:

.. code-block:: text

   talesd-gnn export
     -> cli._cmd_export()
       -> dst_reader.py
       -> event_graph.py
       -> graph_io.py

Training
--------

既存HDF5 graphを使って学習します。

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

実装の流れ:

.. code-block:: text

   talesd-gnn train
     -> cli._cmd_train()
       -> train.train_model()
         -> dataset.H5GraphDataset
         -> model.PhysicsTaleSdGNN
         -> train.py の loss 関数
         -> metrics.py / diagnostics.py

Prediction
----------

学習済みcheckpointでCSVを作成します。

.. code-block:: bash

   .venv/bin/talesd-gnn predict \
     --graphs /path/to/graphs/data_graphs.h5 \
     --checkpoint /path/to/checkpoints/reconstruction.pt \
     -o /path/to/predictions.csv

Input distributions
-------------------

入力特徴量の分布をPDFとJSONに保存します。

.. code-block:: bash

   .venv/bin/talesd-gnn input-distributions \
     --graphs /path/to/graphs/flat50000 \
     -o /path/to/input_distributions \
     --max-graphs 100000

Feature importance
------------------

checkpointに対してfeature group ablationを行います。

.. code-block:: bash

   .venv/bin/talesd-gnn feature-importance \
     --graphs /path/to/graphs/flat50000 \
     --checkpoint /path/to/checkpoints/reconstruction.pt \
     -o /path/to/feature_importance \
     --split validation \
     --max-graphs 50000

Visualization
-------------

HDF5 graphをevent displayとしてPDF出力します。

.. code-block:: bash

   .venv/bin/talesd-gnn visualize \
     --graphs /path/to/graphs/flat50000 \
     --index 0 \
     -o /path/to/graph_000000.pdf

