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
この mode では、source manifest scan、``DAT??????`` source group 管理、zenith-stratified source allocation、
deterministic random event-index selection、graphable event の refill、graph 作成、HDF5 書き込みを
``dstio.tale.graph.write_balanced_graph_h5`` に委譲します。
GNN repository 側は export 時に graph edge を追加・削除・再定義しません。

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

実装の流れ:

.. code-block:: text

   talesd-gnn export-hetero
     -> cli._cmd_export_hetero()
       -> dstio.tale.graph.write_balanced_graph_h5()  # --energy-sample-per-bin あり
          or dstio.tale.graph.write_graph_h5()        # balanced selection なし
       -> dstio 側の GraphEvent build と HDF5 shard write

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
本番の長い学習では、まず grouped event HDF5 を single-file flat training cache に変換します。
flat cache は detector、pulse、waveform、edge、target、label を大きな連続datasetとして保存し、
event offset table で1 event分をsliceして読むため、多数の小さいgzip datasetをランダムに開く経路を避けます。

.. code-block:: bash

   .venv/bin/talesd-gnn convert-hetero-to-flat-cache \
     --input /path/to/graphs/grouped_or_shards \
     --output /path/to/graphs/hetero_flat_training_cache.h5 \
     --compression lzf

学習 DataLoader の既定は fast tensor dict 経路です。
PyG ``HeteroData`` への変換は互換性と解析用に残していますが、長い本番学習では推奨経路ではありません。

.. code-block:: bash

   .venv/bin/talesd-gnn train-hetero \
     --graphs /path/to/graphs/hetero_flat_training_cache.h5 \
     -o /path/to/output/checkpoints/hetero_reco_mass.pt \
     --epochs 128 \
     --batch-size 32 \
     --gradient-accumulation-steps 4 \
     --model-architecture hetero_attention \
     --waveform-encoder transformer \
     --waveform-transformer-max-tokens 128 \
     --training-data-format fast_tensor \
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
         -> hetero_graph_io.H5FlatHeteroGraphDataset または H5HeteroGraphDataset
         -> hetero_training.H5TensorHeteroGraphDataset
         -> hetero_model.MinimalHeteroTaleSdGNN
         -> metrics.py / diagnostics.py

現在予定している heterogeneous 比較では、まず ``WAVEFORM_ENCODER=transformer`` を使います。
``cnn-gru`` は同時に6本投げるのではなく、transformer の結果を見て採用条件を決めた後の ablation として比較します。
relation 定義の ablation が必要な場合は、``dstio`` 側へ追加仕様として入れ、
別 HDF5 graph dataset として export します。``train-hetero`` は graph connectivity を編集しません。

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
ここでいう metric は単一の汎用 score ではありません。再構成では、逆scale後の物理出力に対して、
energy のずれ（``rmse_log10_energy`` や相対energy誤差）、core の xy 平面でのずれ
（``core_68_km`` など）、到来方向の開き角（``angular_68_deg`` など）を評価します。
mass は accuracy、balanced accuracy、AUC、confusion count を別に保存します。
``baseline`` は通常入力での評価、各groupの ``reconstruction`` / ``mass`` はそのgroupを潰した時の評価、
``*_delta`` は ``ablated - baseline`` です。delta は有限な数値 metric について保存されます。

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

Attention maps
--------------

``hetero_attention`` checkpoint では、少数eventについて後処理のattention重みを保存できます。

.. code-block:: bash

   .venv/bin/talesd-gnn attention-maps \
     --graphs /path/to/graphs/hetero_balanced_flat10000 \
     --checkpoint /path/to/checkpoints/hetero_reco_mass.pt \
     -o /path/to/attention_maps \
     --split validation \
     --max-graphs 16

出力は ``attention_maps.json`` と ``attention_maps.npz`` です。
JSON には event identity、source path、prediction/target、保存した配列名を入れます。
NPZ には detector/pulse の lid と位置、``pulse_detector_index``、``pulse_bounds``、
layer/relation/head ごとの edge attention、detector/pulse readout attention を入れます。

これは event display に重ねるための診断値です。attention が大きいことだけを
feature importance や物理的説明とはみなしません。必要度の評価は ablation や
perturbation diagnostics と併用します。

Visualization
-------------

HDF5 graphをevent displayとしてPDF出力します。
graph schema、node/edge の作り方、waveform mask の妥当性を目で確認するために使います。

.. code-block:: bash

   .venv/bin/talesd-gnn visualize \
     --graphs /path/to/graphs/flat50000 \
     --index 0 \
     -o /path/to/graph_000000.pdf
