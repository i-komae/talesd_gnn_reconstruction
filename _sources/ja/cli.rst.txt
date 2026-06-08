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

Heterogeneous HDF5 graph export
-------------------------------

``export-hetero`` は、新しい ``dstio.tale.graph`` schema のHDF5を書きます。
このHDF5は学習用cacheであり、一回通しの再構成で必ず作るものではありません。
graph は detector node、pulse node、detector-level waveform、型付き relation を持ちます。
MC 学習用では、Ising reference core がある graph だけを使うため
``--require-reference-core`` を付けるのが標準です。
大規模MCを偏りにくく作る場合は ``--energy-sample-per-bin`` と ``--energy-sample-stratify-particle`` を付けます。
balanced selector は ``DAT??????`` source group、zenith、azimuth、core位置、event時刻binに候補を分散させ、``--selection-summary`` で選択分布を書き出せます。

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
     --shard-size 50000 \
     -o /path/to/graphs/hetero/hetero.h5

実装の流れ:

.. code-block:: text

   talesd-gnn export-hetero
     -> cli._cmd_export_hetero()
       -> dstio.tale.graph.iter_graphs()
       -> hetero_graph_io.py

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

Heterogeneous training
----------------------

``train-hetero`` は ``export-hetero`` で作成したHDF5を読みます。
学習用 dataset は各 event sample を ``hetero_data.sample_to_hetero_data`` で
PyG ``HeteroData`` に変換します。これにより
``torch_geometric.loader.DataLoader`` が detector/pulse 数の違う graph を batch 化できます。
model は batch 化された ``HeteroData`` を ``hetero_data.hetero_data_to_tensors`` で
この repository の明示的な tensor dict に戻して処理します。

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

実装の流れ:

.. code-block:: text

   talesd-gnn train-hetero
     -> cli._cmd_train_hetero()
       -> hetero_training.train_hetero_model()
         -> hetero_graph_io.H5HeteroGraphDataset
         -> hetero_data.sample_to_hetero_data()
         -> hetero_model.MinimalHeteroTaleSdGNN
         -> metrics.py / diagnostics.py

現在予定している heterogeneous 比較では、まず ``WAVEFORM_ENCODER=transformer`` を使います。
``cnn-gru`` は同時に6本投げるのではなく、transformer の結果を見て採用条件を決めた後の ablation として比較します。

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

Direct DST reconstruction
-------------------------

``reconstruct-dst`` は heterogeneous checkpoint を使い、DST を直接読みます。
中間HDF5 graphは不要です。
``train-hetero`` と同じ ``dstio.tale.graph`` schema と checkpoint scaler を使います。

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

実装の流れ:

.. code-block:: text

   talesd-gnn reconstruct-dst
     -> cli._cmd_reconstruct_dst()
       -> hetero_predict.reconstruct_dst()
         -> dstio.tale.graph.iter_graphs()
         -> hetero_data.sample_to_hetero_data()
         -> hetero_model.MinimalHeteroTaleSdGNN

Input distributions
-------------------

入力特徴量の分布をPDFとJSONに保存します。
HDF5 の内容と split の偏りを確認するための診断です。
学習ごとに変わるものではなく、基本的には HDF5 作成後に同じ graph に対して確認します。
PDF の横には再描画用 artifact も保存します。
``input_feature_sample_values.npz`` は実際に描画に使った sampled values、
``input_feature_sample_values_manifest.json`` は各 NPZ 配列がどの group / column に対応するかの対応表です。

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
出力 JSON には baseline metrics、group ごとの ablated metrics、delta が入ります。
``feature_group_importance_plot_data.json`` には bar plot を推論なしで再描画するための値を保存します。

.. code-block:: bash

   .venv/bin/talesd-gnn feature-importance \
     --graphs /path/to/graphs/flat50000 \
     --checkpoint /path/to/checkpoints/reconstruction.pt \
     -o /path/to/feature_importance \
     --split validation \
     --max-graphs 50000

heterogeneous checkpoint の場合も同じ command を使います。
checkpoint の runtime/model config から判定し、
``hetero_feature_analysis.save_hetero_feature_group_importance`` にdispatchします。
group は detector signal、detector geometry、detector readout context、
pulse timing/signal、Ising pulse annotation、detector waveform、型付き edge group に分かれます。

Visualization
-------------

HDF5 graphをevent displayとしてPDF出力します。
graph schema、node/edge の作り方、waveform mask の妥当性を目で確認するために使います。

.. code-block:: bash

   .venv/bin/talesd-gnn visualize \
     --graphs /path/to/graphs/flat50000 \
     --index 0 \
     -o /path/to/graph_000000.pdf
