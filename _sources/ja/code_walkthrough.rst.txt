コード詳細 walkthrough
======================

このページは、HDF5作成、学習、推論、診断で「どのファイルのどの関数が何をしているか」を読むための案内です。
実行コマンドの使い方だけでなく、コードを追う順番と、各段階で作られるデータ構造を明示します。

このページの読み方
------------------

このリポジトリの処理は、大きく分けると次の順番で進みます。

.. code-block:: text

   DST event
     -> accepted pulseをnodeにしたHDF5 graph
     -> HDF5 graph dataset
     -> scalerで標準化されたmini-batch
     -> PhysicsTaleSdGNN
     -> reconstruction / quality / mass prediction
     -> loss
     -> checkpointとdiagnostics

この順番は、コードを読む時にも有効です。
先にmodelだけを見ると、``edge_index`` や ``batch`` の意味が分かりにくくなります。
逆に、DST exportだけを見ると、GNNがその情報をどう使うのかが見えません。
したがって、まず「1 eventを1 graphへ変換する」、次に「graphをbatchへまとめる」、最後に「GNNがgraphをどう変換する」という順で読むと、入力、モデル、loss、diagnosticsの対応が追いやすくなります。

GNNを読むための基礎
-------------------

Graph Neural Network（GNN）は、graph形式の入力を扱うニューラルネットワークです。
ここでいうgraphは、点と線からなるデータ構造です。
点をnode、線をedgeと呼びます。
本解析では、1つの空気シャワーeventを1つのgraphとして扱います。

本解析での対応は次の通りです。

.. list-table::
   :header-rows: 1

   * - 一般的なGNN用語
     - 本解析での意味
   * - graph
     - 1つの空気シャワーevent。
   * - node
     - Ising noise filter後に残ったaccepted pulse。detectorそのものではない。
   * - edge
     - 異なるdetectorに属するaccepted pulse間の時空間的な関係。
   * - node feature
     - accepted pulseに付く数値ベクトル。pulse自身、detector由来量、event文脈を含む。
   * - edge feature
     - 2つのnode間の距離、時刻差、信号量差、Ising weightなど。
   * - graph-level prediction
     - event全体のenergy、core、arrival direction、quality、massを予測すること。

画像ではpixelが規則格子上に並ぶため、近傍の定義はほぼ固定です。
一方、TALE-SDのeventでは、hitしたdetector数も位置もeventごとに変わります。
そのため、固定サイズの画像として扱うより、node数とedge数がeventごとに変わるgraphとして扱う方が自然です。

tensorとして見ると、1 eventは概ね次の形を持ちます。

.. code-block:: text

   node_features:     [N_v, F_node]
   edge_index:        [2, N_e]
   edge_features:     [N_e, F_edge]
   waveform_features: [N_v, 4, T]
   target:            [F_target]

``N_v`` はnode数、``N_e`` はdirected edge数です。
``edge_index`` の1行目はedgeの始点node、2行目は終点nodeを表します。
例えば ``edge_index[:, k] = [3, 8]`` なら、k番目のedgeはnode 3からnode 8へ向かうdirected edgeです。

mini-batchでは複数eventのnodeを縦に連結します。
そのため、各nodeがどのeventに属するかを示す ``batch`` 配列が必要になります。
後でgraph-level predictionを作る時、``batch`` を使ってnode表現をeventごとに集約します。

基本用語
~~~~~~~~

特徴量
  モデルへ入力する数値です。
  位置、時刻、信号量、波形、edgeの時刻差などが特徴量です。

表現、埋め込み、embedding
  ニューラルネットワーク内部で作られるベクトルです。
  入力特徴量そのものではなく、予測に使いやすい形へ変換された中間表現です。
  本モデルでは、各nodeを最初に192次元のnode表現へ変換します。

MLP
  Multi-Layer Perceptronの略です。
  線形変換と非線形関数を重ねた小さなニューラルネットワークです。
  本モデルでは、node encoder、message生成、出力headなどで使います。

標準化
  学習データの平均を引き、標準偏差で割る変換です。

  .. math::

     x' = \frac{x-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}

  大きさの違う特徴量を同じ程度の数値範囲へそろえ、最適化を安定させます。
  ``\mu`` と ``\sigma`` はtrain splitだけから推定します。
  validation/testの統計を混ぜると、評価データの情報を学習側へ漏らすことになるためです。

非線形関数とloss関数
~~~~~~~~~~~~~~~~~~~~

ニューラルネットワークは線形変換だけでは複雑な関係を表せません。
そのため、層の間に非線形関数を入れます。
また、予測と正解のずれを数値化するためにloss関数を使います。

Sigmoid
  実数を0から1へ写す関数です。

  .. math::

     \sigma(z)=\frac{1}{1+\exp(-z)}

  ``z=0`` で ``0.5`` になり、``z`` が大きいほど1に近づきます。
  mass分類では、modelが出すlogit ``z`` を ``p(Fe)=\sigma(z)`` として確率のように解釈します。
  quality headでも、quality logitを0から1のscoreへ変換する時に使います。

Logit
  確率 ``p`` を実数へ戻す変換です。

  .. math::

     \mathrm{logit}(p)=\log\frac{p}{1-p}

  ニューラルネットワークでは、sigmoidに入れる前の生の出力もlogitと呼びます。
  mass headの出力がlogitである、という時はこの意味です。

SiLU
  本モデルのMLPで使う活性化関数です。

  .. math::

     \mathrm{SiLU}(x)=x\sigma(x)

  負の入力を完全には切り捨てず、正の入力はほぼそのまま通します。
  ReLUより滑らかなため、回帰問題で扱いやすいことがあります。

Softplus
  滑らかに正値を作る関数です。

  .. math::

     \mathrm{softplus}(x)=\log(1+\exp x)

  常に正で、``x`` が大きい時はほぼ ``x`` になります。
  ranking lossや、誤差予測headの正値化で使います。

SmoothL1
  小さい誤差では二乗誤差、大きい誤差では絶対値誤差に近い振る舞いをするlossです。

  .. math::

     \mathrm{SmoothL1}_{\beta}(r)=
     \begin{cases}
       r^2/(2\beta), & |r| < \beta,\\
       |r|-\beta/2, & |r| \ge \beta.
     \end{cases}

  二乗誤差だけを使うと、大きな外れ値がlossを強く支配します。
  SmoothL1は外れ値の影響を抑えつつ、小さい誤差では滑らかに学習できます。

BCEWithLogits
  Binary Cross Entropyとsigmoidをまとめた二値分類用lossです。

  .. math::

     \mathrm{BCEWithLogits}(z,y)
     =
     -y\log\sigma(z) -(1-y)\log(1-\sigma(z)).

  入力 ``z`` は確率ではなくlogitです。
  ``y`` は0または1のlabelだけでなく、0から1のsoft targetでも使えます。
  本リポジトリでは、mass分類のproton/iron labelと、quality scoreのsoft targetの両方に使います。

GNNの中心: message passing
~~~~~~~~~~~~~~~~~~~~~~~~~~

GNNの特徴は、nodeが周囲のnodeから情報を受け取って自分の表現を更新することです。
この処理をmessage passingと呼びます。

一般形は次のように書けます。

.. math::

   h_i^{(k)}
   =
   \gamma^{(k)}
   \left(
     h_i^{(k-1)},
     \bigoplus_{j\in\mathcal{N}(i)}
     \phi^{(k)}(h_i^{(k-1)},h_j^{(k-1)},e_{j,i})
   \right).

ここで、``h_i`` はnode ``i`` の表現、``e_{j,i}`` はedge特徴量、``\phi`` はmessageを作る関数、``\oplus`` はmeanやmaxなどの集約、``\gamma`` はnode表現を更新する関数です。

本モデルでは、``GatedEdgeMessageLayer`` がこの役割を担います。
directed edge ``i -> j`` について、次を連結してmessageの入力にします。

.. code-block:: text

   [h_i, h_j, edge_attr_ij]

その後、message MLPとgate MLPを使って

.. math::

   m_{i\rightarrow j}
   =
   M_{\mathrm{msg}}([h_i,h_j,e_{ij}])
   \cdot
   \sigma(M_{\mathrm{gate}}([h_i,h_j,e_{ij}]))

を作ります。
gateは、そのedgeから来る情報をどの程度通すかを学習する重みです。
物理的に自然な近傍だけを人間が完全に決めるのではなく、edge特徴量を見ながらmodelが通し方を調整します。

node ``j`` は、入ってくるmessageをmeanとmaxで集約します。
meanは周囲全体の平均的な状況を表し、maxは特に強い近傍の影響を拾います。
更新後のnode表現は、元のnode表現と集約messageを組み合わせ、LayerNormとfeed-forward networkで整えます。

本モデルではこの更新を5層繰り返します。
1層目では直接つながったnodeの情報だけが入ります。
2層目では、その近傍が受け取った情報も入ります。
層を重ねるほど、各node表現はより広い範囲の時空間パターンを含むようになります。

ただし、edgeがないnode pairの情報は直接は混ざりません。
したがって、graphの作り方、つまりedge候補とedge特徴量は、GNNの性質を決める重要な設計です。

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
     - ``train.train_model`` : ``train.py:1385``
   * - 推論CSV作成
     - ``cli._cmd_predict`` : ``cli.py:1557``
     - ``predict.predict_graphs``
   * - 入力分布
     - ``cli._cmd_input_distributions`` : ``cli.py:1572``
     - ``feature_analysis.save_input_distributions`` : ``feature_analysis.py:139``
   * - feature group重要度
     - ``cli._cmd_feature_importance`` : ``cli.py:1587``
     - ``feature_analysis.save_feature_group_importance`` : ``feature_analysis.py:369``
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
主要入口は ``build_graph_event`` : ``event_graph.py:697`` です。

``build_graph_event`` の流れ:

1. ``_combine_sub_waveforms`` : ``event_graph.py:159`` で波形segmentをまとめる。
2. ``_extract_hit`` : ``event_graph.py:298`` でdetectorごとのhit、pulse、FADC波形、時刻、chargeを取り出す。
3. ``_merge_hits_by_lid`` : ``event_graph.py:381`` で同一detector IDのhitを統合する。
4. ``_build_node_features`` : ``event_graph.py:474`` でpulseをnodeとして扱い、node特徴量、pulse特徴量、waveform特徴量を作る。
5. ``_build_edges`` : ``event_graph.py:619`` でdetector間のdirected edgeとedge特徴量を作る。
6. ``_target_from_sim`` : ``event_graph.py:424`` でMC truthから再構成targetを作る。
7. ``_particle_label_from_sim`` : ``event_graph.py:463`` でmass分類用labelを作る。
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

実コード抜粋: ``GraphEvent`` と ``build_graph_event``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下が、DST由来の1 eventを、学習コードが読める ``GraphEvent`` へまとめる実コードです。
``GraphEvent`` は単なる入れ物ではなく、HDF5へ保存される配列の境界を決めています。
``build_graph_event`` は、hit抽出、node作成、edge作成、target作成、metadata作成を1か所で接続します。
ここを読むと、「どの情報がHDF5へ入り、どの情報は捨てられるか」が分かります。

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/event_graph.py
   :language: python
   :pyobject: GraphEvent
   :linenos:

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/event_graph.py
   :language: python
   :pyobject: build_graph_event
   :linenos:

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

後でfeature selectionを変える場合は、まずこの関数と ``graph_columns`` : ``event_graph.py:764`` を確認します。
HDF5にはcolumn名も保存されるため、学習時の ``H5GraphDataset`` はcolumn名を見てschema互換性を確認します。
現行schemaでは物理定義が変わっているため、旧HDF5を黙って現行列へ読み替えず、再exportを要求します。

実コード抜粋: accepted pulseをnodeへ展開する部分
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``_build_node_features`` は、detectorごとのhitをそのままnodeにしているわけではありません。
まず各detector内のpulseを時刻順に並べ、閾値を満たすaccepted pulseだけを残します。
その後、accepted pulseごとにnodeを作り、pulse自身の量、同じdetector内accepted pulseの集約量、detector状態、event内の相対位置を同じ ``node_features`` 行に入れます。
このため、同じdetectorにaccepted pulseが複数ある場合は、同じ ``lid`` を持つnodeが複数できます。

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/event_graph.py
   :language: python
   :pyobject: _build_node_features
   :linenos:

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

実コード抜粋: edge候補とedge特徴量
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``_build_edges`` は、全node pairを無条件につなぐ関数ではありません。
同一detector内のpulse pairはedgeにせず、異なるdetector間で距離と時刻差の条件を満たすpairだけを残します。
その後、1つのundirected pairから ``i -> j`` と ``j -> i`` の2本のdirected edgeを作ります。
edge featureには、位置差、時刻差、距離で割った時刻差、信号量差、Ising風のweightとその構成要素が入ります。

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/event_graph.py
   :language: python
   :pyobject: _build_edges
   :linenos:

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

``H5GraphDataset`` : ``dataset.py:202``
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

実コード抜粋: HDF5から1 eventを読む部分
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``__getitem__`` は、DataLoader workerが実際に呼ぶ読み出し関数です。
ここではまだmini-batch化は行いません。
1つのHDF5 event groupから、node、edge、pulse、waveform、target、particle labelをnumpy配列として取り出します。
schema互換性がない場合や、必要なtarget/labelがない場合は、この層で止まります。

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/dataset.py
   :language: python
   :pyobject: H5GraphDataset.__getitem__
   :linenos:

Scaler fitting
~~~~~~~~~~~~~~

``fit_scalers`` : ``dataset.py:599`` はtrain splitだけからscalerを推定します。

対象:

- node features。
- edge features。
- pulse featuresは、追加scalarがある旧仕様の場合だけ対象になる。現行仕様では ``node_index`` のみなのでpulse scalerは作らない。
- target。

これはvalidation/testの情報を標準化へ混ぜないためです。
``fit scalers`` が長い場合、HDF5 shardのI/O、worker数、local cacheの状態を確認します。

collate
~~~~~~~

``collate_graph_arrays`` : ``dataset.py:942`` は複数event sampleを1 batchへまとめます。
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

実コード抜粋: graph sampleをmini-batchへ連結する部分
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``collate_graph_arrays`` は、通常はC++ extension backendへ処理を渡します。
Python backendの ``_collate_graph_arrays_python`` を読むと、batch化で何をしているかが分かります。
重要なのは、node配列を単純に縦連結するだけではなく、edge indexに ``node_offset`` を足して、複数eventを1つの大きなdisconnected graphとして扱える形へ直している点です。
また、``batch`` 配列を作ることで、後段のreadoutがnodeをeventごとに集約できるようにしています。

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/dataset.py
   :language: python
   :pyobject: collate_graph_arrays
   :linenos:

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/dataset.py
   :language: python
   :pyobject: _collate_graph_arrays_python
   :linenos:

train/val/test split
--------------------

splitは ``train.py`` にあります。

.. list-table::
   :header-rows: 1

   * - split mode
     - 関数
     - 内容
   * - ``event``
     - ``split_indices`` : ``train.py:173``
     - event index単位でrandom splitする。
   * - ``source-path``
     - ``split_indices_by_source_path`` : ``train.py:200``
     - ``source_path`` 単位で分ける。
   * - ``source-stratified``
     - ``split_indices_by_stratified_source_path`` : ``train.py:564``
     - ``source_path`` をまたがせず、particleやenergy分布も偏りにくくする。

現在の既定は ``source-stratified`` です。
validation fractionは0.05、test fractionは0.10、train fractionは0.85です。
同じ ``source_path`` はtrain/val/testにまたがりません。

学習本体
--------

``train_model`` : ``train.py:1385`` が学習の中心です。
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

実コード抜粋: ``train_model`` の入口からsplit/scalerまで
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``train_model`` は長い関数ですが、先頭部分だけでも設計が見えます。
ここでは、taskやloss modeの検証、device解決、dataset初期化、split作成、scaler fittingまでを順に行います。
この順番が重要です。
splitより前にscalerをfitするとvalidation/test情報が混ざるため、必ずtrain indexが決まった後にtrain splitだけでscalerをfitします。

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/train.py
   :language: python
   :start-at: def train_model
   :end-before:     detector_lids: list[int] = []
   :linenos:

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

実コード抜粋: 1 epoch内のtrain loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下がtrain側の中心部分です。
``pred_all = model(batch)`` の後、出力tensorを有効なheadへ分け、reconstruction、quality、error、massのうち有効なlossだけを足します。
``training_task == "mass"`` の場合はmass lossが主lossになり、reco+massの場合はreconstruction lossに ``mass_loss_weight`` を掛けたmass lossが加わります。

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/train.py
   :language: python
   :start-at:     for epoch in epoch_iter:
   :end-before:         train_seconds = time.perf_counter() - train_started
   :linenos:

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

現在の主モデルは ``PhysicsTaleSdGNN`` : ``model.py:730`` です。
このモデルは、1つのevent graphを受け取り、event全体の物理量を出すgraph-level modelです。
nodeごとの分類をするmodelではありません。
最終的には、全nodeの情報を1つのevent vectorへ集約し、そのevent vectorからenergy、core、direction、quality、massを出します。

GNNとして何を近似しているか
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

このモデルは、event graph

.. math::

   G=(V,E)

を入力として、event全体の物理量

.. math::

   f_{\theta}(G)
   =
   (\log_{10}E,\ x_c,\ y_c,\ n_x,\ n_y,\ n_z,\ \ldots)

を出す関数 ``f_theta`` を学習します。
ここで ``V`` はaccepted pulse nodeの集合、``E`` は異なるdetector上のpulse間に張ったdirected edgeの集合です。
``theta`` はニューラルネットワークの重みです。

通常の全結合networkへnodeを単に一列に並べて入れると、nodeの順番に意味が生じます。
しかし、event内のaccepted pulseには「配列の何番目か」という物理的な意味はありません。
意味があるのは、pulseの時刻、charge、detector位置、edgeで表した時空間関係です。
GNNを使う理由は、node数がeventごとに変わっても、nodeの順番に依存せず、同じ局所規則を全edgeへ適用できるためです。

この実装で学習しているものは、概念的には次の写像です。

.. math::

   \{x_i\}_{i\in V},\ \{e_{ij}\}_{(i,j)\in E}
   \longrightarrow
   \hat{y}_{\mathrm{event}}.

``x_i`` はaccepted pulse node ``i`` の特徴量、``e_ij`` はnode ``i`` からnode ``j`` へのedge特徴量です。
``hat{y_event}`` はevent-level predictionです。
このため、出力はnodeごとではなくgraphごとに1つです。

物理量との対応
~~~~~~~~~~~~~~

このGNNは、plane fitやLDF fitを明示的に解くコードではありません。
ただし、入力とedgeはそれらの再構成で重要な情報に対応しています。

.. list-table::
   :header-rows: 1

   * - 物理的な情報
     - コード上の入力
     - 主に効くと期待される出力
   * - 到着時刻の相対差
     - ``pulse_arrival_usec_rel``、edgeの ``dt_usec``、``dt_per_km``
     - 到来方向、core位置
   * - 検出器位置と検出器間距離
     - node位置、barycenter相対位置、edge距離
     - core位置、到来方向
   * - 信号量と横方向分布
     - ``log10_pulse_rho``、detector内sum/max rho、edgeの信号量差
     - energy、core位置、mass
   * - 波形形状
     - raw waveform window、accepted mask、waveform encoder出力
     - mass、quality、reconstruction補助
   * - event全体の規模
     - node数、edge数、accepted pulse数、readout pooling
     - energy、quality、mass

message passingは、各pulseが「自分自身の情報」だけでなく、「周囲のdetectorのpulseと時刻・距離・信号量の関係が整合しているか」を内部表現へ入れる処理です。
例えば、あるpulseだけを見るとchargeが大きい理由は分かりません。
周辺pulseとの距離、時刻差、信号量差を見ることで、そのpulseがcore近傍の強信号なのか、時間的に外れたpulseなのかを区別しやすくなります。

入力
~~~~

``PhysicsTaleSdGNN.forward`` が受け取るbatchには、少なくとも以下があります。

- ``x``: 標準化済みnode scalar features。現行schemaでは28列。
- ``edge_index``: directed edgeの始点・終点node index。
- ``edge_attr``: 標準化済みedge features。現行schemaでは13列。
- ``batch``: 各nodeがどのevent graphに属するかを示すindex。
- ``waveform_x``: 各accepted pulse nodeに付いた4 channel waveform。
- ``detector_lids``: detector ID。detector embeddingを使う時だけ有効。
- ``pulse_x`` と ``pulse_node_index``: 旧schemaの追加pulse scalar用。現行schemaでは ``pulse_features`` は ``node_index`` のみで、pulse scalar encoderは無効。

現行schemaでの主な次元は次の通りです。

.. list-table::
   :header-rows: 1

   * - tensor
     - 形
     - 意味
   * - ``x``
     - ``[total_nodes, 28]``
     - accepted pulse nodeのscalar features。
   * - ``edge_index``
     - ``[2, total_edges]``
     - directed edgeのsource/destination。
   * - ``edge_attr``
     - ``[total_edges, 13]``
     - edgeごとの距離、時刻差、信号量差など。
   * - ``waveform_x``
     - ``[total_nodes, 4, T]``
     - upper/lower raw windowとupper/lower accepted mask。
   * - ``batch``
     - ``[total_nodes]``
     - nodeが属するevent graphの番号。

ここで ``total_nodes`` と ``total_edges`` は、mini-batch内の全event graphを連結した後の数です。

node encoder
~~~~~~~~~~~~

最初に、各nodeのscalar featuresをニューラルネットワーク内部の表現へ変換します。
現行モデルでは、node scalar featuresは28列です。
これらはすでに人間が定義した物理量です。
例えば、pulseの時刻とcharge、detectorの位置、detector内accepted pulse数、event内signal barycenterからの相対位置などです。

この28次元の値をそのままmessage passingへ入れるのではなく、``node_encoder`` でhidden dimensionへ写します。
標準設定のhidden dimensionは192です。
つまり、各accepted pulse nodeは最初に

.. code-block:: text

   28 scalar inputs + waveform embedding
     -> 192-dimensional node representation

へ変換されます。

この192次元表現は、個々の列の意味を保ったままの表ではありません。
予測に使いやすいようにmodelが学習した内部表現です。
以後のmessage passingでは、この192次元ベクトルを各nodeの状態として扱います。

数式で書くと、node ``i`` の初期状態は

.. math::

   h_i^{(0)}
   =
   \phi_{\mathrm{node}}
   \left(
     [x_i,\ w_i,\ d_i,\ p_i]
   \right)

です。
``x_i`` はnode scalar、``w_i`` はwaveform encoder出力、``d_i`` はdetector embedding、``p_i`` はpulse scalar summaryです。
現行標準では ``d_i`` と ``p_i`` は空で、主に ``x_i`` と ``w_i`` から ``h_i^(0)`` を作ります。
``phi_node`` が ``self.node_encoder`` に対応します。

waveform encoder
~~~~~~~~~~~~~~~~

``waveform_x`` は各accepted pulse nodeに付いた4 channel waveformです。
現行schemaは ``rise_aligned_raw_plus_accepted_mask_v1`` です。
内容は次の4 channelです。

- upper raw window。
- lower raw window。
- upper accepted-pulse mask。
- lower accepted-pulse mask。

raw windowは、accepted pulseのriseに合わせたFADC波形です。
accepted maskは、同じ時間軸上でaccepted pulse区間を1、それ以外を0として表すmaskです。
maskそのものはchargeではありません。
model側では、必要に応じて ``raw * mask`` からaccepted pulse部分のshapeやcharge情報を作れます。

``WaveformEncoder`` は、この4 channel時系列を短いベクトルへ圧縮します。
標準の ``cnn-gru`` では、1次元CNNで局所的な波形形状を読み、その後GRUで時間方向の情報をまとめます。
CNNは「近いbin同士の形」を見る処理、GRUは「時間順の変化」をまとめる処理です。

コード上は、まず

.. math::

   u_i(t) = \mathrm{Conv1D}(w_i(t))

で時刻ごとの局所波形表現を作ります。
``cnn-gru`` の場合は、それをGRUへ入れて時間方向に走査します。
最後に

.. code-block:: text

   recurrent last state
   encoded mean over time
   encoded max over time

を連結してMLPへ入れ、nodeごとのwaveform embeddingを得ます。
meanは波形全体の平均的な形、maxは局所的に強いピーク、GRUのlast stateは時系列として見た形を持ちます。

accepted maskは、raw waveformのどの範囲がaccepted pulseとして使われたかを示します。
0埋めされたaccepted waveformだけを渡す方式では、accepted区間外のraw形状が失われます。
現行schemaではraw windowとmaskを別channelで渡すため、modelは「raw全体の形」と「accepted pulse部分」を同時に見られます。
また、``_waveform_shape_summary`` では ``raw * mask`` からaccepted部分のsum、peak、centroid、tail、上下層非対称性などを作り、enhanced classification headへ直接渡します。

detector embedding
~~~~~~~~~~~~~~~~~~

detector embeddingは、detector IDごとに学習される自由なベクトルです。
しかし、現行の標準設定では ``detector_embedding_dim=0`` なので使いません。
detectorごとの差は、IDそのものではなく、位置、局所配置、pedestal、waveform応答、edge関係から学習させます。

time-edge encoder
~~~~~~~~~~~~~~~~~

message passingの前に、``EdgeTimeDeltaEncoder`` がedge特徴量のうち時刻差関連の列を処理します。
現行実装では、edge featureの4列目から3列、すなわち

.. code-block:: text

   dt_usec, abs_dt_usec, dt_per_km

をMLPへ入れ、destination nodeごとに平均してnode表現へ足します。
これは、arrival time differenceを通常のedge messageよりも直接node表現へ入れるためです。
空気シャワーの到来方向は時刻差に強く依存するため、この経路は物理的にも重要です。

この処理は、edge ``i -> j`` ごとの時刻差特徴量

.. math::

   \tau_{ij}
   =
   (\Delta t_{ij},\ |\Delta t_{ij}|,\ \Delta t_{ij}/r_{ij})

をMLPへ入れてmessageにし、destination nodeごとに平均するものです。

.. math::

   h_j
   \leftarrow
   \mathrm{LayerNorm}
   \left[
     h_j
     +
     \frac{1}{|\mathcal{N}(j)|}
     \sum_{i\in\mathcal{N}(j)}
     \phi_t(\tau_{ij})
   \right].

ここで ``N(j)`` はnode ``j`` へ入るedgeのsource node集合です。
通常のmessage passingとは別に時刻差を足すため、時刻差が多数のedge featureの中で埋もれにくくなります。

message passing
~~~~~~~~~~~~~~~

``GatedEdgeMessageLayer`` がGNN本体です。
directed edge ``i -> j`` について、modelは次の3つを見ます。

- source node ``i`` の現在の表現 ``h_i``。
- destination node ``j`` の現在の表現 ``h_j``。
- edge feature ``e_ij``。

実装ではこれらを連結します。

.. code-block:: text

   message_input = [h_i, h_j, e_ij]

標準設定では ``h_i`` と ``h_j`` はそれぞれ192次元、edge featureは13次元です。
したがって、message MLPの入力次元は

.. code-block:: text

   192 + 192 + 13 = 397

です。

messageは次の形で作られます。

.. math::

   m_{i\rightarrow j}
   =
   M_{\mathrm{msg}}([h_i,h_j,e_{ij}])
   \cdot
   \sigma(M_{\mathrm{gate}}([h_i,h_j,e_{ij}])).

``M_msg`` は「何を伝えるか」を作るMLPです。
``M_gate`` は「どれだけ通すか」を作るMLPです。
sigmoidを通すため、gateは0から1に近い値になります。
gateが小さければ、そのedgeから来るmessageは弱くなります。
gateが大きければ、そのedgeのmessageは強く通ります。

destination node ``j`` は、入ってくるmessageを2通りで集約します。

- mean aggregate: 入ってくるmessageの平均。周囲全体の平均的な状況を表す。
- max aggregate: 各成分について最大のmessage。強く特徴的な近傍を拾う。

その後、現在のnode表現、mean aggregate、max aggregateを連結し、MLPで更新量を作ります。
更新は残差接続を持ちます。

.. math::

   h_j' =
   \mathrm{LayerNorm}
   \left(
     h_j +
     M_{\mathrm{update}}([h_j,\overline{m}_j,m_j^{\max}])
   \right).

さらにfeed-forward network（FFN）をもう一度通して、

.. math::

   h_j^{\mathrm{out}} = h_j' + \mathrm{FFN}(h_j')

とします。
残差接続は、元の情報を完全に消さずに新しい情報を足すための構造です。
深いnetworkで勾配を流しやすくする役割もあります。

このlayerを5回繰り返します。
1回のmessage passingでは直接つながったnodeの情報だけが入ります。
5回繰り返すと、edgeで5 step以内の範囲から情報が伝わります。
つまり、各node表現は「そのpulse単体の情報」から「周囲のpulseとの時空間的整合性を含んだ情報」へ変わります。

この実装では、edge ``i -> j`` のmessageにdestination node ``h_j`` も入れます。
これは、同じsource pulseからの情報でも、受け取る側のpulseの状態によって意味が変わるためです。
例えば、同じ時刻差でも、destination側がcore近傍の強信号pulseなのか、遠方の弱信号pulseなのかで、再構成への意味は異なります。
``[h_i, h_j, e_ij]`` を使うことで、その組み合わせをMLPが学習できます。

mean aggregateとmax aggregateを両方使う理由も重要です。
meanだけだと、入ってくるedge全体の平均的傾向は見えますが、少数の強い矛盾や強い手がかりが薄まります。
maxだけだと、event全体の安定した傾向を失いやすくなります。
このため、現在のnode状態、mean message、max messageを同時に使って更新します。

gateは、edgeを固定的に「使う/使わない」と決めるものではありません。
edgeごと、node状態ごとにmessageの通りやすさを連続値で調整します。
距離や時刻差だけでは候補edgeを完全には判断できないため、message内容とgateを同時に学習します。

readoutとhead
~~~~~~~~~~~~~

node表現はgraph単位にpoolingされます。
ここでのpoolingは、node数がeventごとに違っても、固定次元のevent vectorを作るための処理です。

``PhysicsTaleSdGNN`` は次を連結します。

- 全nodeのmean pooling。
- 全nodeのmax pooling。
- 4個のattention readout head。

mean poolingはevent全体の平均的な状態を表します。
max poolingはevent内で最も強い特徴を拾います。
attention readoutは、headごとにnodeへ重みを付けて加重平均します。
どのnodeを重視するかは学習で決まります。

hidden dimensionが192、attention headが4個なので、event vectorの次元は

.. code-block:: text

   192 * (2 + 4) = 1152

です。

この1152次元event vectorをshared MLPへ入れ、最終的なheadへ渡します。

readoutが必要な理由は、最終出力がevent-levelだからです。
message passing後の ``h_i`` はまだnodeごとの表現です。
energyやcoreはevent全体に対して1つの値なので、node表現をgraph表現へ集約しなければなりません。
ここで要求される性質は、nodeの並び替えに対して出力が変わらないことです。
mean、max、attention readoutはいずれもnode順に依存しない形でgraph vectorを作ります。

attention readoutは、headごとに

.. math::

   a_{ih}
   =
   \frac{\exp s_h(h_i)}
        {\sum_{k\in G}\exp s_h(h_k)},
   \qquad
   g_h
   =
   \sum_{i\in G} a_{ih} h_i

を計算します。
``h`` はattention head番号です。
headごとに異なるscore関数を学習するため、あるheadは大信号pulse、別のheadは時間的に重要なpulse、別のheadは遠方のpulseを重視する、といった分担が可能になります。
ただし、これは自動的に物理量名が付くわけではないため、解釈にはfeature importanceやattention分布の確認が必要です。

出力head:

- reconstruction head: ``log10_energy_eV``、``core_x_km``、``core_y_km``、``dir_x``、``dir_y``、``dir_z`` を出す。
- mass classification head: proton/iron分類用のmass logitを出す。
- quality head: reconstruction品質を表すquality logitを出す。
- error head: error prediction用のraw値を出す。現行標準では無効。

どのheadを使うかは ``training_task``、``mass_classification``、``quality_prediction``、``error_prediction`` で決まります。

実コード抜粋: ``PhysicsTaleSdGNN.forward``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``forward`` では、まずnode scalar、detector embedding、waveform embedding、pulse summaryを必要に応じて連結します。
その後、node encoder、time-edge encoder、message passing layers、readout、各headの順で処理します。
classification architectureが ``enhanced`` の場合だけ、最終node表現だけでなく、初期node表現、graph context、waveform direct readoutも分類headへ渡します。

この抜粋だけでは各部品の内部が見えないため、前後に使われる主要moduleも併せて読んでください。
``WaveformEncoder`` は波形をnode embeddingへ圧縮し、``EdgeTimeDeltaEncoder`` は時刻差をnode表現へ直接足します。
``GatedEdgeMessageLayer`` がmessage passing本体、``AttentiveReadout`` がnode表現からevent表現を作る部分です。

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

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/model.py
   :language: python
   :pyobject: PhysicsTaleSdGNN.forward
   :linenos:

GNNがしていること、していないこと
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GNNがしていること:

- accepted pulseごとの入力を192次元の内部表現へ変換する。
- edgeでつながった別detectorのpulseから、距離、時刻差、信号量差を考慮したmessageを受け取る。
- そのmessageを5層分繰り返し、node表現へ周囲の時空間的整合性を埋め込む。
- 全nodeをevent vectorへまとめ、event全体の物理量を予測する。

GNNがしていないこと:

- plane fitやLDF fitを明示的に解いているわけではない。
- detector平面を画像としてCNNに入れているわけではない。
- 同一detector内のpulse node同士をedgeで直接つないでいるわけではない。
- detector ID embeddingを標準で使っているわけではない。

したがって、このモデルは「物理式をそのまま実装した再構成器」ではありません。
物理的に意味のある入力とedgeを与え、message passingによって、event全体の再構成に有用な表現を学習するmodelです。

数学関数の補足
--------------

SmoothL1
~~~~~~~~

``SmoothL1`` は、小さい誤差では二乗誤差、大きい誤差では絶対値誤差に近いlossです。
PyTorchの ``beta`` を使う形では、誤差 ``r`` に対して

.. math::

   \mathrm{SmoothL1}_{\beta}(r)
   =
   \begin{cases}
     \frac{1}{2}r^2/\beta, & |r| < \beta,\\
     |r|-\frac{1}{2}\beta, & |r| \ge \beta.
   \end{cases}

小さい誤差では滑らかな二乗型なので微調整しやすく、大きい外れ値では二乗誤差ほど強くなりすぎません。
energy termでこれを使うのは、外れ値に引っ張られすぎずに ``log10E`` の中心的なずれを減らすためです。

sigmoid, logit, BCEWithLogits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sigmoidは任意の実数 ``z`` を0から1の範囲へ写す関数です。

.. math::

   \sigma(z)=\frac{1}{1+\exp(-z)}.

``z`` はlogitです。
mass headやquality headは確率そのものではなくlogitを出します。
確率として見る時だけ ``sigma(z)`` を使います。

``BCEWithLogits`` は、sigmoidとbinary cross entropyを数値的に安定な形でまとめたlossです。
target ``y`` が0または1の場合だけでなく、quality targetのようなsoft targetにも使えます。
数式では

.. math::

   \mathrm{BCEWithLogits}(z,y)
   =
   \mathrm{softplus}(z) - yz,
   \qquad
   \mathrm{softplus}(z)=\log(1+\exp z).

これは

.. math::

   -y\log\sigma(z)-(1-y)\log(1-\sigma(z))

と同じ意味です。
ただし、実装では ``sigmoid`` を明示的に計算してからlogを取るより、``softplus`` を使う方が大きな正負のlogitで安定です。

loss
----

loss定義は ``train.py`` にまとまっています。
lossは、modelの予測と正解のずれを1つの数値にしたものです。
学習では、このlossを小さくするようにmodelの重みを更新します。

ここで重要なのは、lossは評価指標そのものではないという点です。
例えばcore lossは ``core_loss_scale_km`` で正規化され、direction lossは角度scaleで正規化されます。
これは学習を安定させ、energy、core、directionの寄与を調整するためです。
最終的な性能は、diagnosticsで出すenergy resolution、core resolution、angular resolution、mass accuracyなどで確認します。

reconstruction loss
~~~~~~~~~~~~~~~~~~~

``_reconstruction_loss`` : ``train.py`` が再構成lossです。
``loss_mode=physics`` では、targetを物理量へ戻してから、energy、core、directionを別々に評価します。

現行targetは6成分です。

.. code-block:: text

   y = [log10_energy_eV, core_x_km, core_y_km, dir_x, dir_y, dir_z]

modelは標準化されたtarget空間で値を出しますが、``loss_mode=physics`` では、いったん物理単位へ戻してからlossを計算します。
これは、``log10E``、km単位のcore、unit vectorのdirectionを、それぞれ物理的に意味のある誤差として扱うためです。

physics reconstruction lossは、概念的には次の形です。

.. math::

   L_{\mathrm{reco}}
   =
   w_E L_E
   + w_c L_c
   + w_{\theta} L_{\theta}.

energy termは ``log10E`` の差です。

.. math::

   L_E
   =
   \mathrm{SmoothL1}_{\beta=0.05}
   (\widehat{\log_{10}E}-\log_{10}E).

``log10E`` を使う理由は、宇宙線energyが桁で広がる量だからです。
絶対energy差ではなく対数差を見ることで、低energyと高energyを同じ相対的な意味で扱いやすくなります。

core termは、地表面上のxy core誤差です。

.. math::

   L_c
   =
   \left\langle
     \left(\frac{\hat{x}_c-x_c}{s_c}\right)^2
     +
     \left(\frac{\hat{y}_c-y_c}{s_c}\right)^2
   \right\rangle.

``s_c`` が ``core_loss_scale_km`` です。
現在の標準値は ``0.05 km`` です。
これは「50 m程度のcore誤差をloss上でどれくらい重く見るか」を決めるscaleです。
``s_c`` を大きくすると、同じcore誤差でもlossへの寄与は小さくなります。

direction termは、予測方向ベクトルと正解方向ベクトルのなす角を使います。
到来方向は本質的には角度なので、単純な成分ごとの差より、角度誤差で評価する方が物理的に自然です。

.. math::

   \alpha
   =
   \arccos
   \left(
     \frac{\hat{\mathbf{n}}\cdot\mathbf{n}}
          {|\hat{\mathbf{n}}||\mathbf{n}|}
   \right),
   \qquad
   L_{\theta}
   =
   \mathrm{SmoothL1}_{\beta=1}
   \left(
     \frac{\alpha}{s_{\theta}}
   \right).

``s_theta`` は ``angular_loss_scale_deg`` で、現在の標準値は1度です。

energy bias loss
~~~~~~~~~~~~~~~~

``_energy_bin_bias_loss`` は、true energy binごとの平均energy residualを0へ寄せます。

.. math::

   b_k
   =
   \left\langle
     \widehat{\log_{10}E}-\log_{10}E
   \right\rangle_{\mathrm{bin}\ k},
   \qquad
   L_{\mathrm{energy\ bias}}
   =
   \left\langle b_k^2 \right\rangle_k.

これは、全体のRMSEだけでなく、特定のenergy binで系統的に高く予測する、または低く予測する傾向を抑えるためです。
``ENERGY_BIAS_WEIGHT`` が正の時だけtotal lossへ足されます。

``_energy_particle_bias_loss`` は、同じtrue energy bin内でprotonとironの平均energy residual差を小さくします。

.. math::

   \Delta b_k
   =
   \left\langle r_E \right\rangle_{k,\mathrm{Fe}}
   -
   \left\langle r_E \right\rangle_{k,\mathrm{p}},
   \qquad
   r_E = \widehat{\log_{10}E}-\log_{10}E.

massによってenergy再構成が系統的にずれると、reconstruction結果を物理的に解釈しにくくなります。
このlossは、その粒子種依存のbiasを抑えるための項です。

core scaleを変えると、同じcore誤差でもloss上の重みが変わります。
現在の既定は ``0.05 km`` です。

quality loss
~~~~~~~~~~~~

``_quality_targets_from_reconstruction`` : ``train.py`` が、再構成誤差からsoft quality targetを作ります。
``_quality_prediction_loss`` : ``train.py`` はquality logitとsoft targetのBCEWithLogitsです。

ここでのtargetはmass labelではありません。
再構成がどの程度良いかを表す連続値です。

quality targetは、energy、core、directionの再構成誤差から作ります。

.. math::

   q_{\mathrm{target}}
   =
   \exp
   \left[
     -\frac{1}{3}
     \left(
       s_E + s_c + s_{\theta}
     \right)
   \right].

ここで、``s_E``、``s_c``、``s_theta`` は、それぞれenergy、core、directionの誤差を設定scaleで割ったものです。
誤差が小さいeventでは ``q_target`` は1に近くなります。
誤差が大きいeventでは0に近くなります。

quality headは ``q_target`` を直接出すのではなく、quality logitを出します。
そのlogitとsoft targetに対してBCEWithLogitsを計算します。
quality targetは予測から作られますが、``detach`` されるため、quality lossがreconstruction predictionそのものを直接変えるわけではありません。
quality headに「このeventの再構成は良さそうか」を学習させるための補助lossです。

mass loss
~~~~~~~~~

``_mass_classification_loss`` : ``train.py`` がmass分類lossです。
``bce`` と ``focal`` を扱います。
必要ならranking lossも足します。

mass-onlyではreconstruction targetを答えとして使うのではなく、``particle_label`` だけで分類します。
mass accuracyが頭打ちになる場合は、score separation、energy bin別accuracy、input feature group寄与、classification head容量を確認します。

mass headは1つのlogit ``z`` を出します。
sigmoidを通した

.. math::

   p(\mathrm{Fe}) = \sigma(z)

をironらしさのscoreとして扱います。
labelはprotonが0、ironが1です。

BCE modeでは、各eventについて

.. math::

   L_{\mathrm{BCE}}
   =
   -y\log\sigma(z)
   -(1-y)\log(1-\sigma(z))

を計算し、batch平均を取ります。

focal modeでは、分類しやすいeventの寄与を弱め、分類しにくいeventを相対的に重くします。

.. math::

   L_{\mathrm{focal}}
   =
   (1-p_t)^\gamma L_{\mathrm{BCE}},

where ``p_t`` は正解classに対する予測確率です。
正解に自信を持っているeventでは ``p_t`` が1に近くなり、``(1-p_t)^\gamma`` が小さくなります。

ranking lossは、iron logitがproton logitより十分大きくなるように促します。

.. math::

   L_{\mathrm{rank}}
   =
   \left\langle
     \mathrm{softplus}
     \left(
       m - (z_{\mathrm{Fe}}-z_{\mathrm{p}})
     \right)
   \right\rangle.

``m`` はmarginです。
ironとprotonのscore separationを広げたい時に使います。

実コード抜粋: loss関数
~~~~~~~~~~~~~~~~~~~~~~

以下の関数が、本文で説明したlossの実装です。
``_reconstruction_loss`` は物理単位へ戻したenergy/core/direction lossを作ります。
``_quality_targets_from_reconstruction`` と ``_quality_prediction_loss`` は、再構成誤差からsoft quality targetを作り、quality logitへBCEWithLogitsをかけます。
``_mass_classification_loss`` はBCE、focal、rankingをまとめます。

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/train.py
   :language: python
   :pyobject: _reconstruction_loss
   :linenos:

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/train.py
   :language: python
   :pyobject: _quality_targets_from_reconstruction
   :linenos:

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/train.py
   :language: python
   :pyobject: _quality_prediction_loss
   :linenos:

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/train.py
   :language: python
   :pyobject: _mass_classification_loss
   :linenos:

total loss
~~~~~~~~~~

reconstructionとmassを同時に学習する時は、複数のlossを重み付きで足します。

.. math::

   L_{\mathrm{total}}
   =
   L_{\mathrm{reco}}
   + w_q L_q
   + w_m L_{\mathrm{mass}}
   + w_b L_{\mathrm{energy\ bias}}
   + w_{pb} L_{\mathrm{particle\ bias}}.

有効になる項は設定で変わります。
例えば、``quality_prediction=0`` ならquality lossは入りません。
``mass_classification=0`` ならmass lossは入りません。

lossの重みは、どの物理量をどれだけ重視して学習させるかを決めます。
ただし、重みを変えると最適化問題そのものが変わるため、過去runとの比較では必ず設定を記録します。

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

``save_input_distributions`` : ``feature_analysis.py:139`` はHDF5 graphを読み、node、edge、pulse、targetなどの分布をPDF/JSONへ出します。
featureを削る前に、まずここで分布、外れ値、energy/particleとの関係を見ます。

feature group重要度
~~~~~~~~~~~~~~~~~~~

``save_feature_group_importance`` : ``feature_analysis.py:369`` はcheckpointを読み、validation/test split上でfeature group ablationを行います。

流れ:

1. checkpointからmodel config、scaler、splitを復元する。
2. 通常入力でbaseline metricsを計算する。
3. feature groupごとに値を平均置換またはzero化する。
4. metricsの悪化量を計算する。
5. JSON/PDFへ保存する。

これは再学習を伴わないpost-hoc評価です。
相関の強い入力群がすべて必要かを調べる時は、まずこの結果を見ます。
ただし、ablationは「その学習済みモデルにとっての使用度」を見るもので、特徴量を削って再学習した時の最終性能を直接保証するものではありません。

実コード抜粋: feature group ablation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``save_feature_group_importance`` は、checkpointからmodel、scaler、split indexを復元し、まず通常入力でbaseline metricsを出します。
その後、feature groupごとにdataset wrapperを差し替え、同じmodelで再推論してmetricsの悪化量を記録します。
ここでは再学習しないため、1回の学習が重い状況でも入力群の効き方を調べられます。

.. literalinclude:: ../../../src/talesd_gnn_reconstruction/feature_analysis.py
   :language: python
   :pyobject: save_feature_group_importance
   :linenos:

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
