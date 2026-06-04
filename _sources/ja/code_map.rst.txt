コード読解マップ
================

このページは、リポジトリのどこで何が行われるかを読むための地図です。
細かい式や GNN の考え方は :doc:`code_walkthrough` に分け、ここでは実行フローとファイルの対応をまとめます。

読む順番
--------

本番 run の理解では、以下の順番で読むと処理の依存関係が見えます。

.. code-block:: text

   DST
     -> talesd-gnn export
       -> HDF5 graph
         -> H5GraphDataset
           -> train_model
             -> PhysicsTaleSdGNN
               -> loss / metrics / diagnostics

各段階の役割は次の通りです。

.. list-table::
   :header-rows: 1

   * - 段階
     - 主なファイル
     - ここで決まること
   * - CLI の入口
     - ``src/talesd_gnn_reconstruction/cli.py``
     - コマンドライン引数を Python API の引数へ変換する。
   * - DST 読み込み
     - ``src/talesd_gnn_reconstruction/dst_reader.py``
     - DST bank を Python の record として順に読む。
   * - event graph 作成
     - ``src/talesd_gnn_reconstruction/event_graph.py``
     - accepted pulse node、edge、waveform、target、metadata を作る。
   * - HDF5 書き込み
     - ``src/talesd_gnn_reconstruction/graph_io.py``
     - graph を shard 化し、schema 情報を HDF5 attrs に保存する。
   * - HDF5 読み込み
     - ``src/talesd_gnn_reconstruction/dataset.py``
     - shard を dataset として扱い、DataLoader 用の sample を返す。
   * - 学習全体
     - ``src/talesd_gnn_reconstruction/train.py``
     - split、scaler、DataLoader、学習 loop、loss、best checkpoint、評価を管理する。
   * - model 本体
     - ``src/talesd_gnn_reconstruction/model.py``
     - node/edge/waveform を埋め込み、message passing と readout で出力を作る。
   * - 診断図
     - ``src/talesd_gnn_reconstruction/diagnostics.py``
     - learning curve、energy dependence、quality cut、mass 診断を作る。
   * - 入力分布と重要度
     - ``src/talesd_gnn_reconstruction/feature_analysis.py``
     - 入力特徴量分布と post-hoc group ablation を作る。

CLI は薄い wrapper
------------------

``talesd-gnn`` の各 subcommand は、重い処理を直接実装しているわけではありません。
``cli.py`` が引数を解釈し、実際の処理関数へ渡します。

.. list-table::
   :header-rows: 1

   * - command
     - ``cli.py`` の関数
     - 呼び出す主な処理
   * - ``export``
     - ``_cmd_export``
     - DST を読み、``event_graph`` で graph を作り、``graph_io`` で HDF5 に書く。
   * - ``train``
     - ``_cmd_train``
     - ``train.train_model`` を呼び、学習と評価を行う。
   * - ``predict``
     - ``_cmd_predict``
     - ``predict.predict_graphs`` を呼び、checkpoint から CSV を作る。
   * - ``input-distributions``
     - ``_cmd_input_distributions``
     - ``feature_analysis.save_input_distributions`` を呼ぶ。
   * - ``feature-importance``
     - ``_cmd_feature_importance``
     - ``feature_analysis.save_feature_group_importance`` を呼ぶ。
   * - ``visualize``
     - ``_cmd_visualize``
     - ``visualize.visualize_graphs`` を呼び、event display を作る。

HDF5 graph export
-----------------

``talesd-gnn export`` は DST から学習用 HDF5 を作ります。処理は大きく 4 段階です。

.. code-block:: text

   cli._cmd_export()
     -> dst_reader.iter_dst_banks()
     -> event_graph.build_graph_event()
     -> graph_io.create_graph_file() / graph_io.write_graph()

``event_graph.build_graph_event`` が返す ``GraphEvent`` は、1 event を GNN に渡すための中間表現です。
主な中身は次の通りです。

.. list-table::
   :header-rows: 1

   * - field
     - 意味
     - 後段での使い方
   * - ``node_features``
     - accepted pulse ごとの scalar 特徴量。
     - model の node encoder へ入る。
   * - ``edge_index``
     - node 間の有向 edge。
     - message passing の接続関係を決める。
   * - ``edge_features``
     - pulse 間、検出器間、信号量差などの関係量。
     - edge encoder と message layer へ入る。
   * - ``waveform_features``
     - rise-aligned raw waveform window と accepted mask。
     - waveform encoder へ入る。
   * - ``target``
     - MC truth から作る教師値。
     - reconstruction loss と metrics に使う。
   * - ``metadata``
     - source path、particle label、event id など。
     - split、diagnostics、mass label に使う。

HDF5 には graph 本体だけでなく、``columns_json`` や ``waveform_schema`` も保存されます。
この schema が一致しない HDF5 は、現在の dataset/model で安全に読めません。

Dataset 読み込み
----------------

``H5GraphDataset`` は HDF5 shard 群を 1 つの dataset として見せる class です。
初期化時に全 graph を読むのではなく、各 shard の event 数、metadata の有無、schema を確認します。
実データは ``__getitem__`` で必要になった graph だけを読みます。

.. code-block:: text

   H5GraphDataset.__init__()
     -> shard list を解決
     -> columns_json / waveform_schema を確認
     -> cumulative length を作る

   H5GraphDataset.__getitem__(index)
     -> global index を shard index と local index へ変換
     -> node/edge/waveform/target/metadata を読む
     -> DataLoader 用の dict を返す

``max_open_files`` は同時に開く HDF5 file 数を制限します。
大きい dataset では file handle と memory の管理に関係します。

Split と scaler
---------------

``train_model`` は dataset を作った後、train/validation/test の index を作ります。
現在の本番設定では ``source-stratified`` split を使い、同じ ``source_path`` が複数 split にまたがらないようにします。

その後、train split だけを使って scaler を fit します。
validation/test の情報で標準化係数を決めないためです。

.. code-block:: text

   train_model()
     -> H5GraphDataset(...)
     -> split index 作成
     -> fit_scalers(dataset, train_indices)
     -> DataLoader 作成

scaler fitting は学習前に時間がかかりますが、node、edge、waveform、target のスケールを決める重要な段階です。

Model の入力と出力
------------------

現在の physics model は ``PhysicsTaleSdGNN`` です。入力は batch dict として渡されます。

.. code-block:: text

   batch
     node_features
     edge_index
     edge_features
     waveform_features
     batch
     target / particle_label / metadata

``model.py`` では、node scalar、edge scalar、waveform を別々に埋め込みます。
その後、``GatedEdgeMessageLayer`` を複数回通し、最後に graph 単位の readout を行います。
出力 head は task 設定で変わります。

.. list-table::
   :header-rows: 1

   * - head
     - 使う条件
     - 出力
   * - reconstruction
     - ``training_task=reconstruction``
     - energy、core x/y、direction vector。
   * - quality
     - ``quality_prediction=True``
     - その event の reconstruction quality target。
   * - mass
     - ``mass_classification=True`` または ``training_task=mass``
     - iron/proton の分類 logit。
   * - error
     - ``error_prediction=True``
     - 誤差推定用の補助出力。現状の主設定では off。

Loss と評価
-----------

loss は ``train.py`` に集約されています。重要なのは、model の forward と loss の計算が分かれている点です。
model は予測値を返し、``train.py`` が target と比較して loss を作ります。

.. list-table::
   :header-rows: 1

   * - 処理
     - 主な関数
     - 内容
   * - reconstruction loss
     - ``_reconstruction_loss`` / ``_reconstruction_training_loss``
     - energy、core、direction の supervised loss を計算する。
   * - angular loss
     - ``_angular_loss_from_vectors``
     - 方向ベクトル間の角度を loss にする。
   * - energy bias penalty
     - ``_energy_bin_bias_loss`` / ``_energy_particle_bias_loss``
     - energy bin 別、粒子種別の平均 bias を罰する。
   * - quality loss
     - ``_quality_prediction_loss``
     - reconstruction error から作る quality target を学習する。
   * - mass loss
     - ``_mass_classification_loss``
     - BCE/focal/ranking を設定に応じて計算する。

学習後は best validation checkpoint を読み戻して validation/test prediction を行い、
``metrics.py`` と ``diagnostics.py`` で数値と図を作ります。

Server submitter
----------------

Slurm では ``talesd-gnn train`` を直接打つより、``scripts/submit_server_*.sh`` を使います。
submitter は環境変数を集め、resource、run directory、local cache、runtime copy を設定してから batch job を投げます。

.. code-block:: text

   submit_server_reco_mass_training.sh
     -> task-specific env を設定
     -> submit_server_waveform_full_training.sh
       -> Slurm resource を決める
       -> local graph/runtime cache を準備する
       -> train_large_existing_graphs.sh を実行する
         -> talesd-gnn train

``submit_server_reco_mass_training.sh`` や ``submit_server_mass_only_training.sh`` は、
task 固有の既定値を設定してから共通 submitter へ渡します。
そのため、比較する時は最終的に渡された環境変数と run config を確認する必要があります。

Diagnostics と後処理
--------------------

学習中は ``save_learning_progress`` が learning curve を更新します。
best validation が更新された時には、設定に応じて best diagnostics も更新されます。
学習終了後は ``save_training_diagnostics`` が validation/test の図と ``summary.json`` を作ります。

入力分布と feature importance は ``feature_analysis.py`` が担当します。
入力分布は HDF5 と split の性質を見るための診断で、feature importance は学習済み checkpoint に対する post-hoc ablation です。
feature importance は再学習ではなく、入力 group を置換して metrics の変化を見る解析です。
