コード詳細 walkthrough
======================

このページは、HDF5作成、学習、推論、診断で「どのファイルのどの関数が何をしているか」を読むための案内です。
実行コマンドの使い方だけでなく、コードを追う順番と、各段階で作られるデータ構造を明示します。

全体の入口
----------

CLIの入口は ``pyproject.toml`` の ``[project.scripts]`` で定義されています。

.. code-block:: text

   talesd-gnn -> talesd_gnn_reconstruction.cli:main

``src/talesd_gnn_reconstruction/cli.py`` の役割は、コマンドライン引数をPython APIへ渡すことです。
主要なdispatch先は以下です。

.. list-table::
   :header-rows: 1

   * - 処理
     - CLI関数
     - 実処理の主な関数
   * - HDF5 graph作成
     - ``cli._cmd_export`` : ``cli.py:1227``
     - ``dst_reader.iter_dst_banks``、``event_graph.build_graph_event``、``graph_io.write_graph``
   * - 学習
     - ``cli._cmd_train`` : ``cli.py:1450``
     - ``train.train_model`` : ``train.py:1173``
   * - 推論CSV作成
     - ``cli._cmd_predict`` : ``cli.py:1551``
     - ``predict.predict_graphs``
   * - 入力分布
     - ``cli._cmd_input_distributions`` : ``cli.py:1566``
     - ``feature_analysis.save_input_distributions`` : ``feature_analysis.py:133``
   * - feature group重要度
     - ``cli._cmd_feature_importance`` : ``cli.py:1581``
     - ``feature_analysis.save_feature_group_importance`` : ``feature_analysis.py:352``
   * - graph可視化
     - ``cli._cmd_visualize``
     - ``visualize.py``

HDF5 graph export
-----------------

HDF5作成は、DSTを直接読む部分、eventをGNN graphへ変換する部分、HDF5へ書く部分に分かれています。

.. code-block:: text

   talesd-gnn export
     -> cli._cmd_export()
       -> dst_reader.iter_dst_banks()
       -> event_graph.build_graph_event()
       -> graph_io.create_graph_file()
       -> graph_io.write_graph()

DST読み込み
~~~~~~~~~~~

``src/talesd_gnn_reconstruction/dst_reader.py`` はDST bankを逐次読み込みます。
主要入口は ``iter_dst_banks`` : ``dst_reader.py:222`` です。

``iter_dst_banks`` が行うこと:

- ``dstio`` を使ってDST fileを開く。
- ``kind``、``max_events``、trigger mode、source index、event dateなどの条件を適用する。
- MCの場合は ``rusdraw`` 系の情報をTALE-SD calibev相当の構造へ変換する。
- 各eventを ``BankRecord`` としてyieldする。``BankRecord`` には ``bank``、``source_path``、``source_index``、``source_kind`` が入る。

この段階ではまだGNN用のnodeやedgeは作っていません。DST由来のevent単位データを、後段へ流すだけです。

GraphEvent作成
~~~~~~~~~~~~~~

``src/talesd_gnn_reconstruction/event_graph.py`` はDST bankをGNN graphへ変換します。
主要入口は ``build_graph_event`` : ``event_graph.py:688`` です。

``build_graph_event`` の流れ:

1. ``_combine_sub_waveforms`` : ``event_graph.py:159`` で波形segmentをまとめる。
2. ``_extract_hit`` : ``event_graph.py:293`` でdetectorごとのhit、pulse、FADC波形、時刻、chargeを取り出す。
3. ``_merge_hits_by_lid`` : ``event_graph.py:376`` で同一detector IDのhitを統合する。
4. ``_build_node_features`` : ``event_graph.py:471`` でpulseをnodeとして扱い、node特徴量、pulse特徴量、waveform特徴量を作る。
5. ``_build_edges`` : ``event_graph.py:610`` でdetector間のdirected edgeとedge特徴量を作る。
6. ``_target_from_sim`` : ``event_graph.py:419`` でMC truthから再構成targetを作る。
7. ``_particle_label_from_sim`` : ``event_graph.py:460`` でmass分類用labelを作る。
8. ``GraphEvent`` として返す。

``GraphEvent`` : ``event_graph.py:41`` に入る主要データ:

.. list-table::
   :header-rows: 1

   * - field
     - 内容
   * - ``node_features``
     - detector位置、barycenter相対位置、accepted pulse到着時刻、pulse自身のcharge、同一detector内のaccepted pulse合計charge、pulse数、波形長、FADC peakなど。
   * - ``edge_index``
     - directed edgeの始点・終点node index。
   * - ``edge_features``
     - detector間距離、時刻差、空間重み、因果方向、signal weightなど。
   * - ``pulse_features``
     - 対応nodeを示す ``node_index``。追加のpulse scalar入力は現行仕様では落とす。
   * - ``waveform_features``
     - waveform encoderへ渡す波形特徴量。現行仕様では上下層の rise-aligned raw window と accepted-pulse mask。
   * - ``target``
     - MC truth由来の ``log10_energy_eV, core_x_km, core_y_km, dir_x, dir_y, dir_z``。coreは地表面上の2次元位置で、到来方向は3成分単位ベクトルとして保存する。
   * - ``particle_label``
     - mass分類用label。現在の二値分類ではproton/ironを区別する。
   * - ``metadata``
     - ``source_path``、``source_index``、parttypeなど、splitや診断に使う情報。

node特徴量
~~~~~~~~~~

``_build_node_features`` は、accepted pulseを1 nodeとして扱います。
同一detectorに複数pulseがある場合でも、pulse単位のnodeとして並ぶため、``node_lids`` には同じdetector IDが複数回現れることがあります。

ここで作られる特徴量群:

- detectorの位置 ``x, y, z``。
- event内barycenterからの相対位置、半径。
- event内の最初のaccepted pulseから見た、そのpulse自身の相対時刻。
- detector trigger time。
- そのaccepted pulse自身のchargeの対数・平方根。
- 同一detector内のaccepted pulse最大charge、合計charge、accepted pulse数、accepted pulse time span。
- detector waveform segment数、waveform length。
- FADC peak、pedestal、sigma。
- accepted pulse order、first accepted pulse flag。

後でfeature selectionを変える場合は、まずこの関数と ``graph_columns`` : ``event_graph.py:755`` を確認します。
HDF5にはcolumn名も保存されるため、学習時の ``H5GraphDataset`` はcolumn名を見てschema互換性を確認します。
現行schemaでは物理定義が変わっているため、旧HDF5を黙って現行列へ読み替えず、再exportを要求します。

edge特徴量
~~~~~~~~~~

``_build_edges`` はnode pairを走査し、異なるdetector間のpairだけを候補にします。
距離や時刻差が大きすぎるpairは落とし、残ったpairからbidirectional directed edgeを作ります。

edge特徴量には次のような情報が入ります。

- detector間距離。
- 到着時刻差。
- spatial weight。
- causal direction。
- pulse信号量差 ``dlog10_pulse_rho`` とsignal weight。
- degree normalization用の量。

edge特徴量を変える場合は、``event_graph.py`` だけでなく、モデル側の ``edge_feature_dim`` とcheckpoint互換性にも影響します。

HDF5書き込み
~~~~~~~~~~~~

``src/talesd_gnn_reconstruction/graph_io.py`` がHDF5 schemaを管理します。

``create_graph_file`` : ``graph_io.py:14`` が行うこと:

- HDF5 root attributesに ``format=talesd_gnn_graphs``、``format_version``、特徴量column定義、configを保存する。
- ``events`` groupを作る。
- ``metadata`` groupに ``event_id``、``source_path``、``source_index``、``parttype``、``particle_label`` を保存できるdatasetを作る。

``write_graph`` : ``graph_io.py:53`` が行うこと:

- ``events/%08d`` groupを作る。
- ``node_features``、``node_positions_km``、``node_lids``、``edge_index``、``edge_features``、``pulse_features``、``waveform_features`` を書く。
- targetとparticle labelがある場合はそれも書く。
- metadata datasetへsource情報とparticle情報を追記する。

このmetadataは、後の ``source-stratified`` splitで重要です。
同じ ``source_path`` をtrain/val/testへまたがせないために使います。

energy-flat samplingとshard
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cli._cmd_export`` の内部には、energy binごとに候補を集め、指定数だけ選ぶ処理があります。
関連関数は ``cli.py`` 内にあります。

- ``_scan_energy_candidates_for_file`` : ``cli.py:195`` はfile内のenergy候補を数える。
- ``_add_energy_sample`` : ``cli.py:1075`` はbinごとのreservoirへcandidateを追加する。
- ``_sampled_graphs_from_reservoirs`` : ``cli.py:1101`` はreservoirから最終サンプルを作る。
- ``_write_selected_graph_shard`` : ``cli.py:439`` は選ばれたeventをshardへ書く。
- ``_write_ordered_selected_graph_shard`` : ``cli.py:577`` はshuffleされた書き込み計画に従って書く。

ここでshuffleする対象は、graph本体ではなく、どのsourceのどのeventをどの順番で書くかという軽量な選択・書き込み計画です。
graph本体を先に全部materializeするとI/Oとメモリが大きくなるためです。

学習用dataset読み込み
---------------------

``src/talesd_gnn_reconstruction/dataset.py`` はHDF5 graphをPyTorch DataLoaderへ渡すための層です。

``H5GraphDataset`` : ``dataset.py:181``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``H5GraphDataset.__init__`` が行うこと:

- ``graphs_path`` がfileかdirectoryかを判定し、HDF5 shard一覧へ展開する。
- HDF5 attrsからformat、waveform schema、column定義を読む。
- ``particle_filter`` や ``max_graphs`` を適用する。
- shardごとのgraph数を読み、global indexからlocal indexへ変換するための累積長を作る。
- metadata、target、particle labelの有無を確認する。
- node feature column selectionを解決する。
- progress label ``initialize graph shards`` を出す。

``_handle`` はHDF5 file handleのLRU cacheを持ちます。
open file数は環境変数 ``TALESD_GNN_H5_MAX_OPEN_FILES`` で制御され、既定は4です。

``__getitem__`` が返すsample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``H5GraphDataset.__getitem__`` は、global indexを ``(path_index, local_index, key)`` へ変換し、HDF5から1 eventを読みます。
返るdictには以下が入ります。

- ``node_features``
- ``edge_index``
- ``edge_features``
- ``pulse_features``
- ``waveform_features``
- ``target``
- ``attrs``
- 必要な場合は ``detector_lids``、``particle_label``、``node_positions``

この段階ではまだtorch tensorではありません。numpy配列中心のsampleです。

Scaler fitting
~~~~~~~~~~~~~~

``fit_scalers`` : ``dataset.py:549`` はtrain splitだけからscalerを推定します。

対象:

- node features。
- edge features。
- pulse featuresは、追加scalarがある旧仕様の場合だけ対象になる。現行仕様では ``node_index`` のみなのでpulse scalerは作らない。
- target。

これはvalidation/testの情報を標準化へ混ぜないためです。
``fit scalers`` が長い場合、HDF5 shardのI/O、worker数、local cacheの状態を確認します。

collate
~~~~~~~

``collate_graph_arrays`` : ``dataset.py:892`` は複数event sampleを1 batchへまとめます。
内部ではPython backendまたはC++ extension backendを使います。

collateで行うこと:

- scalerを適用する。
- eventごとのnode配列を連結する。
- edge indexをnode offsetで付け替える。
- ``batch`` 配列を作り、nodeがどのgraphに属するかを示す。
- ``waveform_features`` を連結する。``pulse_features`` は現行仕様では ``node_index`` だけなので、pulse scalar tensorは空になる。
- targetやmass labelをbatch配列にする。
- torch tensorへ変換する。

DataLoader workerが落ちる場合は、この段階のHDF5読み込み、collate、メモリ使用量を優先して確認します。

train/val/test split
--------------------

splitは ``train.py`` にあります。

.. list-table::
   :header-rows: 1

   * - split mode
     - 関数
     - 内容
   * - ``event``
     - ``split_indices`` : ``train.py:167``
     - event index単位でrandom splitする。
   * - ``source-path``
     - ``split_indices_by_source_path`` : ``train.py:194``
     - ``source_path`` 単位で分ける。
   * - ``source-stratified``
     - ``split_indices_by_stratified_source_path`` : ``train.py:489``
     - ``source_path`` をまたがせず、particleやenergy分布も偏りにくくする。

現在の既定は ``source-stratified`` です。
validation fractionは0.05、test fractionは0.10、train fractionは0.85です。
同じ ``source_path`` はtrain/val/testにまたがりません。

学習本体
--------

``train_model`` : ``train.py:1173`` が学習の中心です。
CLI、submitter、notebookから最終的にここへ入ります。

``train_model`` の主な段階:

1. torchとmodel moduleをimportする。
2. ``training_task``、loss mode、mass option、diagnostic optionを検証する。
3. deviceを解決する。
4. ``H5GraphDataset`` を作る。
5. splitを作る。
6. train splitでscalerをfitする。
7. 最初のsampleからinput dimensionを推定する。
8. ``TaleSdGNN`` または ``PhysicsTaleSdGNN`` を作る。
9. optimizer、LR scheduler、DataLoaderを作る。
10. epoch loopでtrainとvalidationを回す。
11. validation lossが改善したらbest checkpointを保存する。
12. 最後にbest checkpointを読み戻し、validation/test predictionとdiagnosticsを作る。

この流れはlogの ``stage_seconds`` と対応します。
例えば ``dataset_init``、``fit_scalers``、``model_and_loaders``、``epochs``、``diagnostics`` は、この段階の所要時間です。

epoch loop
~~~~~~~~~~

各epochでは以下を行います。

train側:

- ``model.train()`` にする。
- DataLoaderからbatchを受け取る。
- ``batch_to_device`` でGPUへ送る。
- ``pred_all = model(batch)`` でforwardする。
- 出力をreconstruction、mass、quality、error headへ分ける。
- 有効なtaskに応じてlossを計算する。
- ``backward``、gradient clipping、optimizer stepを行う。

validation側:

- ``model.eval()`` と ``torch.no_grad()`` を使う。
- 同じlossを計算するが、optimizer stepはしない。
- val lossをbest判定に使う。

checkpoint
~~~~~~~~~~

checkpointにはmodel重みだけでなく、以下も入ります。

- model config。
- scaler。
- history。
- metrics。
- split index。
- runtime information。
- diagnostics summary。

そのため、推論やfeature importanceではcheckpointからsplitやscalerを復元できます。

モデル構造
----------

現在の主モデルは ``PhysicsTaleSdGNN`` : ``model.py:718`` です。

入力
~~~~

``PhysicsTaleSdGNN.forward`` が受け取るbatchには、少なくとも以下があります。

- ``node_features``: node encoderへ入る。
- ``edge_index``、``edge_features``: message passingへ入る。
- ``batch``: graph poolingで使う。
- ``pulse_features``: 現行仕様では ``node_index`` だけで、pulse encoderは無効。旧仕様の追加pulse scalarがある場合だけnode表現へ足される。
- ``waveform_features``: waveform encoderでnodeまたはgraph表現へ入る。
- ``detector_lids``: detector embeddingを使う場合に入る。

node / pulse / waveform
~~~~~~~~~~~~~~~~~~~~~~~

``PhysicsTaleSdGNN`` では、node特徴量とwaveform特徴量を使います。現行仕様では追加のpulse scalar特徴量は無効です。

- node encoderは静的なdetector/event特徴量をhidden dimへ写す。
- pulse encoderは、旧仕様の追加pulse scalarがある場合だけ使われます。現行仕様では ``pulse_dim=0`` です。
- waveform encoderは ``cnn-gru`` などでFADC波形由来の特徴量を抽出する。
- detector embeddingを有効にすると、detector ID固有の埋め込みも加わる。

message passing
~~~~~~~~~~~~~~~

edge情報は ``GatedEdgeMessageLayer`` などで使われます。
各layerはedge特徴量と隣接node表現からmessageを作り、node表現を更新します。
``num_layers`` が5なら、この更新を5層分行います。

readoutとhead
~~~~~~~~~~~~~

node表現はgraph単位にpoolingされます。
``PhysicsTaleSdGNN`` はmean、max、attention readout、graph contextを組み合わせます。

出力head:

- reconstruction head: energy、core、directionを出す。
- mass classification head: mass logitを出す。
- quality head: quality logitを出す。
- error head: error prediction用のraw値を出す。

どのheadを使うかは ``training_task``、``mass_classification``、``quality_prediction``、``error_prediction`` で決まります。

loss
----

loss定義は ``train.py`` にまとまっています。

reconstruction loss
~~~~~~~~~~~~~~~~~~~

``_reconstruction_loss`` : ``train.py`` が再構成lossです。
``loss_mode=physics`` では、targetを物理量へ戻してから、energy、core、directionを別々に評価します。

- energy: log energy差へSmoothL1を使う。
- core: xy core差を ``core_loss_scale_km`` で割ってSmoothL1を使う。
- direction: angular errorを角度scaleで正規化してSmoothL1を使う。
- energy bias: ``--energy-bias-loss-weight`` が正なら、true energy binごとの平均 ``pred_logE - true_logE`` を0へ寄せる。
- particle energy bias: ``--energy-particle-bias-loss-weight`` が正なら、同じtrue energy bin内のproton/iron平均logE residual差を0へ寄せる。

core scaleを変えると、同じcore誤差でもloss上の重みが変わります。
現在の既定は ``0.05 km`` です。

quality loss
~~~~~~~~~~~~

``_quality_targets_from_reconstruction`` : ``train.py`` が、再構成誤差からsoft quality targetを作ります。
``_quality_prediction_loss`` : ``train.py`` はquality logitとsoft targetのBCEWithLogitsです。

ここでのtargetはmass labelではありません。
再構成がどの程度良いかを表す連続値です。

mass loss
~~~~~~~~~

``_mass_classification_loss`` : ``train.py`` がmass分類lossです。
``bce`` と ``focal`` を扱います。
必要ならranking lossも足します。

mass-onlyではreconstruction targetを答えとして使うのではなく、``particle_label`` だけで分類します。
mass accuracyが頭打ちになる場合は、score separation、energy bin別accuracy、input feature group寄与、classification head容量を確認します。

診断と評価
----------

``src/talesd_gnn_reconstruction/diagnostics.py`` は学習後の図とsummaryを作ります。
主要入口は ``save_training_diagnostics`` : ``diagnostics.py:2463`` です。

主な出力:

- learning curve。
- reconstruction精度のenergy dependence。
- angular/core/energy resolution。
- quality cut performance。
- mass ROC、confusion matrix、score distribution。
- JSON summary。

best validation checkpointを読み戻してからvalidation/testの評価を行うため、最後のepochではなくbest epochの性能がdiagnosticsに反映されます。

入力分布とfeature importance
--------------------------------

``src/talesd_gnn_reconstruction/feature_analysis.py`` は、再学習を大量に回さず入力特徴量の確認をするためのmoduleです。

入力分布
~~~~~~~~

``save_input_distributions`` : ``feature_analysis.py:133`` はHDF5 graphを読み、node、edge、pulse、targetなどの分布をPDF/JSONへ出します。
featureを削る前に、まずここで分布、外れ値、energy/particleとの関係を見ます。

feature group重要度
~~~~~~~~~~~~~~~~~~~

``save_feature_group_importance`` : ``feature_analysis.py:352`` はcheckpointを読み、validation/test split上でfeature group ablationを行います。

流れ:

1. checkpointからmodel config、scaler、splitを復元する。
2. 通常入力でbaseline metricsを計算する。
3. feature groupごとに値を平均置換またはzero化する。
4. metricsの悪化量を計算する。
5. JSON/PDFへ保存する。

これは再学習を伴わないpost-hoc評価です。
相関の強い入力群がすべて必要かを調べる時は、まずこの結果を見ます。
ただし、ablationは「その学習済みモデルにとっての使用度」を見るもので、特徴量を削って再学習した時の最終性能を直接保証するものではありません。

変更したい時に読む場所
----------------------

.. list-table::
   :header-rows: 1

   * - 変更したい内容
     - 最初に読む場所
     - 注意
   * - node特徴量を増減する
     - ``event_graph._build_node_features``、``event_graph.graph_columns``
     - HDF5 schemaとcheckpoint互換性に影響する。
   * - edge特徴量を変える
     - ``event_graph._build_edges``
     - ``edge_feature_dim`` とmodel入力に影響する。
   * - waveformの扱いを変える
     - ``event_graph`` のwaveform抽出、``model.WaveformEncoder``
     - collateの ``waveform_features`` shapeも確認する。
   * - splitを変える
     - ``train.split_indices*``
     - 旧runとの比較ではsplit modeを必ず記録する。
   * - lossを変える
     - ``train._reconstruction_loss``、``train._quality_prediction_loss``、``train._mass_classification_loss``
     - full training前に小さいbatchでforward/loss smokeを通す。
   * - model headを変える
     - ``model.PhysicsTaleSdGNN``
     - checkpoint config、出力分割、loss側も同時に確認する。
   * - diagnostics図を変える
     - ``diagnostics.save_training_diagnostics`` と周辺のplot関数
     - 学習本体に影響しないが、評価解釈に影響する。
   * - Slurm投入条件を変える
     - ``scripts/submit_server_*``
     - graph、task、loss、epochを意図せず変えない。

コードを読む推奨順
------------------

初めて読む場合は、次の順で読むと依存関係を追いやすいです。

1. ``cli.py`` の ``build_parser`` と ``_cmd_train`` を読む。
2. ``train.py`` の ``train_model`` のstage順を読む。
3. ``dataset.py`` の ``H5GraphDataset`` と ``collate_graph_arrays`` を読む。
4. ``model.py`` の ``PhysicsTaleSdGNN`` を読む。
5. ``train.py`` のloss関数を読む。
6. HDF5作成を理解するために ``dst_reader.py``、``event_graph.py``、``graph_io.py`` を読む。
7. 評価図を理解するために ``diagnostics.py`` と ``feature_analysis.py`` を読む。

この順番なら、学習で使われる入力、モデル、loss、出力、diagnosticsの対応が見えます。
