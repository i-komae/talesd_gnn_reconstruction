CLI API
=======

CLIの入口は ``talesd-gnn`` です。``pyproject.toml`` では
``talesd_gnn_reconstruction.cli:main`` に対応しています。
各 command の実装上の位置づけは :doc:`code_map` にまとめています。
このページでは、実際に使う command、主な入力、出力、後段での使い道を確認します。

HDF5 graph export
-----------------

DSTを読み、GNN用HDF5 graph shardを作成します。
出力される HDF5 は学習、入力分布、可視化、feature importance の共通入力になります。
MC 学習用の場合、``target`` と ``particle_label`` を含むため、再構成と mass 分類の教師データとして使えます。

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
``train`` は graph を直接変更しません。split、scaler、model、checkpoint、metrics、diagnostics を run directory 側に作ります。

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
``--no-truth`` を使わない場合、MC graph では truth も CSV に含められます。
実データ graph では、truth を使わず prediction だけを出す用途になります。

.. code-block:: bash

   .venv/bin/talesd-gnn predict \
     --graphs /path/to/graphs/data_graphs.h5 \
     --checkpoint /path/to/checkpoints/reconstruction.pt \
     -o /path/to/predictions.csv

Input distributions
-------------------

入力特徴量の分布をPDFとJSONに保存します。
HDF5 の内容と split の偏りを確認するための診断です。
学習ごとに変わるものではなく、基本的には HDF5 作成後に同じ graph に対して確認します。

.. code-block:: bash

   .venv/bin/talesd-gnn input-distributions \
     --graphs /path/to/graphs/flat50000 \
     -o /path/to/input_distributions \
     --max-graphs 100000

Feature importance
------------------

checkpointに対してfeature group ablationを行います。
これは再学習ではありません。学習済み model に対して、node/edge/waveform などの入力 group を置換し、
metrics がどれだけ悪化するかを見ます。

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
graph schema、node/edge の作り方、waveform mask の妥当性を目で確認するために使います。

.. code-block:: bash

   .venv/bin/talesd-gnn visualize \
     --graphs /path/to/graphs/flat50000 \
     --index 0 \
     -o /path/to/graph_000000.pdf
