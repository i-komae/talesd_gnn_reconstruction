Heterogeneous model の詳しい説明
=================================

このページでは、現在の TALE-SD heterogeneous model を、入力配列から最終出力までの値の流れとして説明します。
書き方は、PyTorch 公式の
`Dataset / DataLoader tutorial <https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_、
`training tutorial <https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html>`_、
PyTorch Geometric 公式の
`heterogeneous graph guide <https://pytorch-geometric.readthedocs.io/en/stable/notes/heterogeneous.html>`_
に合わせています。
つまり、まず data object を定義し、次に model、最後に training と inference を説明します。

ここで説明する経路は、実際にこの repository で使っている実装です。

.. list-table::
   :header-rows: 1

   * - 段階
     - 実装
   * - HDF5 sample の読み書き
     - ``talesd_gnn_reconstruction.hetero_graph_io``
   * - tensor / PyG 変換
     - ``talesd_gnn_reconstruction.hetero_data``
   * - model
     - ``talesd_gnn_reconstruction.hetero_model.MinimalHeteroTaleSdGNN``
   * - 学習
     - ``talesd_gnn_reconstruction.hetero_training.train_hetero_model``
   * - DST 直接推論
     - ``talesd_gnn_reconstruction.hetero_predict.reconstruct_dst``

Forward pass の全体図
---------------------

.. figure:: ../fig/hetero_model_forward.svg
   :alt: GraphEvent から出力headまでの TALE-SD heterogeneous model forward pass。
   :width: 100%

   1 event を 1 つの graph として丸ごと使います。detector waveform は detector node に 1 回だけ持たせます。pulse node は ``pulse_detector_index`` と ``pulse_bounds`` で対応する detector waveform の区間を参照します。

1 event を heterogeneous data object として見る
-----------------------------------------------

PyG では heterogeneous graph を node type ごとの store と edge type ごとの store に分けて表します。
TALE-SD graph でも同じ考え方を使います。1 event には 2 種類の node と 3 種類の relation があります。

.. code-block:: text

   node types:
     detector
     pulse

   edge types:
     ("pulse", "interacts", "pulse")
     ("detector", "near", "detector")
     ("detector", "observes", "pulse")

本番学習では、batch 化のために PyG ``HeteroData`` を使います。
HDF5 dataset は 1 event につき 1 つの ``HeteroData`` を返し、
``torch_geometric.loader.DataLoader`` がそれらを batch 化します。
その後、``MinimalHeteroTaleSdGNN.forward`` の中で、batch 化された ``HeteroData`` を
``hetero_data_to_tensors`` によりこの repository の明示的な tensor dict に戻します。
DST 直接推論などの single-event path では、明示的な tensor dict を直接渡すこともできます。
次は batch 化前に ``hetero_data.sample_to_hetero_data`` が作る実際の構造です。

.. code-block:: python

   data["detector"].x = tensors["detector"]["x"]
   data["detector"].context = tensors["detector"]["context"]
   data["detector"].pos = tensors["detector"]["pos"]
   data["detector"].lid = tensors["detector"]["lid"]
   data["detector"].waveform = tensors["detector"]["waveform"]

   data["pulse"].x = tensors["pulse"]["x"]
   data["pulse"].pos = tensors["pulse"]["pos"]
   data["pulse"].lid = tensors["pulse"]["lid"]
   data["pulse"].detector_index = tensors["pulse"]["detector_index"]
   data["pulse"].pulse_bounds = tensors["pulse"]["pulse_bounds"]

   data["pulse", "interacts", "pulse"].edge_index = ...
   data["pulse", "interacts", "pulse"].edge_attr = ...
   data["detector", "near", "detector"].edge_index = ...
   data["detector", "near", "detector"].edge_attr = ...
   data["detector", "observes", "pulse"].edge_index = ...
   data["detector", "observes", "pulse"].edge_attr = ...

これは、全 node を 1 つの同じ feature matrix に押し込む homogeneous graph ではありません。
detector と pulse は物理的に違う対象なので、別の node type として扱います。

入力 field
----------

``detector_features`` は detector ごとの signal、timing、局所 geometry です。
``detector_context_features`` は readout / calibration context です。
これを分けておくことで、shower feature と readout context を無自覚に混ぜず、後で ablation できます。

``detector_waveforms`` は detector ごとの full calibrated VEM waveform です。
pulse node ごとに同じ waveform を複製しません。pulse は ``pulse_detector_index`` で detector を指し、
``pulse_bounds`` で waveform のどの時間区間に対応するかを持ちます。

``pulse_features`` は pulse timing、charge、Ising reference core がある時の core-relative coordinate、
Ising annotation を含みます。Ising rejected pulse candidate も入力に残します。
delayed pulse、waveform tail、multi-pulse structure は mass / energy に物理的に有効な情報なので、
noise として hard drop しません。

``edge_features_by_type`` は edge の連続物理量です。
例えば pulse-pulse edge には timing difference、距離、``dt_per_km``、``ising_weight`` などが入ります。
これは単なる relation label ではなく、attention が使う数値入力です。

変換と scaling
--------------

``hetero_sample_to_tensors`` は HDF5 または DST 直接 graph から来た NumPy 配列を tensor に変換します。
scaler が与えられている場合、detector、context、pulse、edge、target を train split の統計量で標準化します。

.. code-block:: python

   detector_features = _scale_tensor(
       detector_features,
       _scaler_for(scalers, "detector", "detector_features"),
   )
   detector_context = _scale_tensor(
       detector_context,
       _scaler_for(scalers, "detector_context", "detector_context_features"),
   )
   pulse_features = _scale_tensor(
       pulse_features,
       _scaler_for(scalers, "pulse", "pulse_features"),
   )
   edge_features_by_type[relation] = _scale_tensor(
       edge_features,
       _scaler_for(scalers, f"edge:{relation}", relation),
   )
   target_tensor = _scale_tensor(target_tensor, _scaler_for(scalers, "target"))

これは PyTorch の Dataset / DataLoader の分離と同じ考え方です。
dataset は 1 sample を読み、変換層が typed graph object にし、PyG DataLoader がそれを batch 化し、
model 内部では統一された tensor dict として扱います。

detector encoder と pulse encoder
---------------------------------

model はまず detector と pulse を同じ hidden dimension の vector に変換します。

detector node は 3 系統で処理します。

.. code-block:: text

   detector_features         -> detector feature MLP
   detector_context_features -> detector context MLP
   detector_waveforms        -> waveform encoder
                          concat -> detector_node_encoder

pulse node は pulse feature を MLP に通します。

.. code-block:: text

   pulse_features -> pulse_node_encoder

waveform encoder は detector ごとに 1 回だけ適用します。
最初の transformer waveform sweep では submitter で ``WAVEFORM_ENCODER=transformer`` を指定します。
graph 側の architecture は ``hetero_attention`` のままです。

Relation attention
------------------

message passing の中心は ``HeteroAttentionMessageLayer`` です。
relation type ごとに query、key、value を別々に作ります。

.. code-block:: python

   src_input = torch.cat([src_state[src_index], edge_attr], dim=-1)
   query = self.query[relation](dst_state[dst_index]).view(-1, self.heads, self.head_dim)
   key = self.key[relation](src_input).view(-1, self.heads, self.head_dim)
   value = self.value[relation](src_input).view(-1, self.heads, self.head_dim)
   scores = (query * key).sum(dim=-1) * scale
   weights = _scatter_softmax(scores, dst_index, dst_state.shape[0])
   messages = (value * weights[:, :, None]).reshape(-1, self.hidden_dim)

TALE-SD で重要なのは、``key`` と ``value`` に ``edge_attr`` も入れる点です。
これにより、model は ``dt_usec``、``distance_km``、``dt_per_km``、``ising_weight`` などを見ながら、
どの pulse / detector relation を重視するかを決められます。

message を集めた後、detector state と pulse state は residual connection、LayerNorm、feed-forward block で更新されます。

.. code-block:: text

   new_state = LayerNorm(old_state + update(old_state, aggregated_message))
   new_state = LayerNorm(new_state + FFN(new_state))

これは HGT を参考にした実装ですが、HGT そのものではありません。
PyG ``HGTConv`` は使っていません。HGSampling も使いません。
TALE では 1 event = 1 graph なので、event graph を丸ごと読みます。

Readout
-------

再構成では node ごとの出力ではなく、event ごとに 1 つの出力が必要です。
そこで ``HeteroAttentiveReadout`` は detector node と pulse node を別々に集約します。

.. code-block:: text

   detector states -> mean, max, attention-weighted sums
   pulse states    -> mean, max, attention-weighted sums
   concat          -> event vector

detector と pulse を最初から同じ pool に混ぜず、それぞれの情報を event-level summary にしてから結合します。

Output heads
------------

event vector は task ごとの head に入ります。

.. list-table::
   :header-rows: 1

   * - head
     - 出力
     - 意味
   * - reconstruction
     - 6 values
     - ``log10_energy_eV``, ``core_x_km``, ``core_y_km``, ``dir_x``, ``dir_y``, ``dir_z``
   * - mass
     - 1 logit
     - ``sigmoid(logit)`` が iron probability
   * - quality
     - 1 logit
     - auxiliary quality score
   * - predicted error
     - 3 values
     - energy、angular、core error の予想値

今の比較方針では、auxiliary head を全部同時に入れません。
最初の transformer waveform sweep では、3つの dataset size に対して
quality-only と predicted-error-only の reco+mass を投げます。

学習
----

``train_hetero_model`` は標準的な PyTorch training loop と同じ流れです。

.. code-block:: text

   H5HeteroGraphDataset
     -> train / validation / test split
     -> train split で scaler を fit
     -> first sample から model shape を決める
     -> PyG DataLoader が HeteroData sample を batch 化
     -> model が batched HeteroData を明示的な tensor dict に変換
     -> forward
     -> MC target と比較して loss を計算
     -> backward
     -> optimizer step
     -> checkpoint と scaler を保存

checkpoint には model weight と scaler の両方を保存します。
そのため、DST 直接再構成でも学習時と同じ feature normalization を使えます。

DST 直接再構成
--------------

直接推論では HDF5 を必須にしません。学習時と同じ graph schema と変換層を使います。

.. code-block:: text

   DST
     -> dstio.tale.graph.iter_graphs
     -> graph_event_to_sample
     -> hetero_sample_to_tensors または sample_to_hetero_data
     -> hetero_attention checkpoint
     -> reconstruction CSV

これが最終的な DST 再構成経路です。
HDF5 は学習用 cache であり、本番の一回通し再構成で必須の中間ファイルではありません。

最初の transformer sweep の投入コマンド
----------------------------------------

balanced HDF5 が揃った後、最初の transformer 6本は次の入口で投げます。

.. code-block:: bash

   cd /dicos_ui_home/ikomae/work/src/talesd_gnn_reconstruction

   RUN_ID=hetero_balance_20260606_143020 \
   SUBMIT_EXPORTS=0 \
   SUBMIT_TRAINING=1 \
   MODEL_ARCHITECTURE=hetero_attention \
   WAVEFORM_ENCODER=transformer \
   PARTITION=v100-al9_long \
   scripts/submit_server_hetero_dataset_size_sweep.sh

これは 50000、20000、10000 events/bin の HDF5 に対して、
quality-only と predicted-error-only の reco+mass 学習を投げます。
``cnn-gru`` はここでは同時に投げず、transformer の結果を見て採用条件を決めた後に比較します。
