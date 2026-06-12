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
attention の記法は PyTorch 公式
`MultiheadAttention <https://docs.pytorch.org/docs/2.12/generated/torch.nn.MultiheadAttention.html>`_
の Query / Key / Value の言い方に合わせています。
heterogeneous graph の記法は PyG 公式
`HeteroData <https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.data.HeteroData.html>`_
を基準にし、必要な箇所で PyG
`HGTConv <https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.HGTConv.html>`_
との違いも説明します。

ここで説明する経路は、実際にこの repository で使っている実装です。

.. list-table::
   :header-rows: 1

   * - 段階
     - 実装
   * - HDF5 sample の書き込み / 読み込み
     - 書き込みは ``dstio.tale.graph.write_balanced_graph_h5`` または ``write_graph_h5``、読み込みは ``talesd_gnn_reconstruction.hetero_graph_io``
   * - tensor / PyG 変換
     - ``talesd_gnn_reconstruction.hetero_data``
   * - model
     - ``talesd_gnn_reconstruction.hetero_model.MinimalHeteroTaleSdGNN``
   * - 学習
     - ``talesd_gnn_reconstruction.hetero_training.train_hetero_model``
   * - DST 直接推論
     - ``talesd_gnn_reconstruction.hetero_predict.reconstruct_dst``

一言でいうと、現在の model は TALE-SD event 用の
``edge-attribute-conditioned, relation-specific heterogeneous graph attention``
model です。1 event を detector node と pulse node からなる graph として扱い、
relation ごとに別の Q/K/V attention で node を更新し、最後に event-level vector
から energy、core、direction、mass、quality、predicted error を出します。
HGT に近い考え方ですが、PyG ``HGTConv`` や ``HeteroConv`` そのものではありません。
HGSampling も使いません。TALE では 1 event を 1 graph として丸ごと読めるため、
waveform tail、delayed pulse、multi-pulse structure、Ising rejected pulse
candidate を sampling で落とさず入力に残します。

Forward pass の全体図
---------------------

.. figure:: ../fig/hetero_model_forward.svg
   :alt: GraphEvent から出力headまでの TALE-SD heterogeneous model forward pass。
   :width: 100%

   1 event を 1 つの graph として丸ごと使います。detector waveform は detector node に 1 回だけ持たせます。pulse node は ``pulse_detector_index`` と ``pulse_bounds`` で対応する detector waveform の区間を参照します。

1 event を heterogeneous data object として見る
-----------------------------------------------

PyG では heterogeneous graph を node type ごとの store と edge type ごとの store に分けて表します。
TALE-SD graph でも同じ考え方を使います。1 event には 2 種類の node と v3 の 7 種類の relation があります。

.. code-block:: text

   node types:
     detector
     pulse

   edge types:
     ("pulse", "same_detector_next", "pulse")
     ("pulse", "same_detector_prev", "pulse")
     ("pulse", "near_space", "pulse")
     ("pulse", "time_causal", "pulse")
     ("detector", "near", "detector")
     ("detector", "observes", "pulse")
     ("pulse", "observed_by", "detector")

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

   for relation, edge_type in EDGE_TYPE_BY_RELATION.items():
       data[edge_type].edge_index = tensors["edge_index_by_type"][relation]
       data[edge_type].edge_attr = tensors["edge_features_by_type"][relation]

これは、全 node を 1 つの同じ feature matrix に押し込む homogeneous graph ではありません。
detector と pulse は物理的に違う対象なので、別の node type として扱います。

入力 field
----------

``detector_features`` は detector ごとの signal、timing、局所 geometry です。
``detector_context_features`` は readout / calibration context です。
これを分けておくことで、shower feature と readout context を無自覚に混ぜず、後で ablation できます。

v3 schema の ``pulse_arrival_usec_rel`` は pulse onset から作ります。
ここでの onset は上下層 rise の平均ではなく、上下層 rise の早い方です。
0 点は event 内の最初の accepted graph pulse candidate です。
Ising rejected candidate pulse も ML input に残るため、Ising-kept pulse だけを基準に時刻を引き直しません。

``detector_trigger_usec_rel`` は互換性のための名前です。
値は detector node の代表到着時刻として使います。
``cleaning="ising"`` では、その detector に属する最初の ``ising_keep = 1`` pulse onset を、
``pulse_arrival_usec_rel`` と同じ 0 点で表します。
detector waveform の開始時刻としては使いません。
Ising-kept pulse がない detector は ``detector_arrival_time_valid = 0`` で、この列は 0 です。
detector timing を解釈する時は validity flag も必ず見ます。

``detector_waveforms`` は detector ごとの full calibrated VEM waveform です。
pulse node ごとに同じ waveform を複製しません。pulse は ``pulse_detector_index`` で detector を指し、
``pulse_bounds`` で waveform のどの時間区間に対応するかを持ちます。
v3 schema では no-signal live detector の ``detector_waveform_valid`` は 0 です。
その detector の waveform array は ``dstio`` 側で zero-fill され、model は waveform embedding を
detector scalar/context embedding と結合する前に 0 に mask します。

v3 の detector feature には
``detector_has_ising_kept_pulse``, ``detector_ising_kept_pulse_count``,
``detector_ising_removed_pulse_count`` も入ります。
これは、Ising-kept pulse を持つ detector と、Ising rejected candidate pulse だけを持つ detector を
detector-level でも区別するための情報です。

``pulse_features`` は pulse timing、charge、Ising reference core がある時の core-relative coordinate、
Ising annotation を含みます。Ising rejected pulse candidate も入力に残します。
delayed pulse、waveform tail、multi-pulse structure は mass / energy に物理的に有効な情報なので、
noise として hard drop しません。

``edge_features_by_type`` は edge の連続物理量です。
例えば pulse-pulse edge には timing difference、距離、``dt_per_km``、``ising_weight`` などが入ります。
これは単なる relation label ではなく、attention が使う数値入力です。

v3 の pulse-pulse relation は次のように分けます。

``pulse__same_detector_next__pulse`` / ``pulse__same_detector_prev__pulse``
   同じ detector の連続 pulse です。時間順の forward relation と reverse relation を明示的に分けます。

``pulse__near_space__pulse``
   detector 間距離が ``<= 1.5 km`` の別 detector pulse pair です。双方向で、時間 cut はありません。

``pulse__time_causal__pulse``
   near-space のうち、``abs(dt) <= distance / c + 2 FADC bins`` かつ Ising
   ``raw_weight >= 0.2`` を満たす subset です。これは shower front と矛盾しにくい
   compatible pair を表すため双方向です。

これらの relation は graph schema 側の決定です。
GNN model は ``dstio`` が書いた edge set を入力として使い、学習時に edge を切ったり追加したりしません。

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

tensor shape と行列計算
-----------------------

PyG で batch 化した後は、複数 event の node が連結された tensor になります。
どの node がどの event に属するかは ``batch`` vector で保持します。
以下では ``N_d`` を batch 内の detector node 数、``N_p`` を pulse node 数、
``E_r`` を relation ``r`` の edge 数、``H`` を hidden dimension、``B`` を event 数とします。

.. list-table::
   :header-rows: 1

   * - tensor
     - shape
     - 意味
   * - ``detector.x``
     - ``[N_d, F_detector]``
     - detector の signal、timing、local geometry
   * - ``detector.context``
     - ``[N_d, F_context]``
     - readout / calibration context
   * - ``detector.waveform``
     - ``[N_d, C_waveform, T]``
     - detector ごとの waveform channel と時間 bin
   * - ``pulse.x``
     - ``[N_p, F_pulse]``
     - Ising annotation を含む pulse feature
   * - ``edge_index_by_type[r]``
     - ``[2, E_r]``
     - relation ``r`` の source node index と destination node index
   * - ``edge_features_by_type[r]``
     - ``[E_r, F_edge_r]``
     - edge ごとの連続的な物理量
   * - ``detector.batch`` / ``pulse.batch``
     - ``[N_d]`` / ``[N_p]``
     - batch 化後に各 node が属する event index

dense な adjacency matrix は作りません。
``edge_index_by_type[r][0]`` が source row、``edge_index_by_type[r][1]`` が destination row を選びます。
graph propagation は row selection、destination ごとの softmax、``index_add_`` による加算で実行します。

最初の encoding は通常の学習可能な行列積を含む MLP です。

.. code-block:: text

   detector_feature_embedding = MLP_detector(detector.x)        -> [N_d, H]
   detector_context_embedding = MLP_context(detector.context)   -> [N_d, H]
   pulse_hidden_0             = MLP_pulse(pulse.x)              -> [N_p, H]

detector waveform は detector ごとの 1D time series として処理します。

.. code-block:: text

   detector.waveform [N_d, C_waveform, T]
     -> Conv1d layers
     -> encoded waveform time sequence [N_d, T, H_waveform]

``WAVEFORM_ENCODER=transformer`` の場合、Transformer はこの detector ごとの waveform time sequence にだけ作用します。
つまり、同じ detector waveform 内の time bin 同士の self-attention です。
detector node や pulse node の graph relation を全結合で見るものではありません。
Transformer block に入れる前に、encoded time sequence は
``WAVEFORM_TRANSFORMER_MAX_TOKENS`` 以下へ downsample されます。
server 既定は 128 token です。これにより、保存されている detector waveform が長い場合でも、
self-attention の二乗コストを上限付きにします。
1つの detector waveform では、各 time bin が token になります。
time bin ``t`` と同じ waveform 内の別 time bin ``u`` に対して、Transformer は次を作ります。

.. code-block:: text

   q_t = W_Q x_t
   k_u = W_K x_u
   v_u = W_V x_u

   score_tu = dot(q_t, k_u) / sqrt(d_head)
   alpha_tu = softmax(score_tu over waveform time bins u)
   y_t = sum_u alpha_tu * v_u

物理的には、ある waveform time bin が、同じ detector waveform のどの時刻 bin を見るべきかを学習しています。
そのため、立ち上がり、ピーク、tail、二峰構造、delayed component、multi-pulse waveform 構造を、
固定幅の畳み込みだけより柔軟に表現できます。
最後に時間方向を pooling します。

.. code-block:: text

   waveform_embedding =
     Linear(concat(mean_time(encoded), max_time(encoded))) -> [N_d, H_waveform]

detector branch では、3つの detector embedding を結合して graph hidden space へ射影します。

.. code-block:: text

   detector_hidden_0 =
     Linear(concat(detector_feature_embedding,
                   detector_context_embedding,
                   waveform_embedding)) -> [N_d, H]

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
``detector_waveform_valid`` が 0 の detector row は waveform encoder に通しません。
その row の waveform embedding は zero で埋めます。
その detector は live status、geometry、detector scalar feature、detector-detector edge では寄与できますが、
waveform encoder の計算を消費せず、存在しない waveform signal としても寄与しません。

attention という言葉が出てくる場所
-----------------------------------

この repository では ``attention`` という言葉が複数の場所で出てきます。
同じ名前でも、見ている対象が違います。

.. list-table::
   :header-rows: 1

   * - component
     - code path
     - 何が何を見るか
   * - waveform transformer
     - ``model.py`` の ``WaveformEncoder(mode="transformer")``
     - 1 detector waveform の中の time bin 同士
   * - graph relation attention
     - ``hetero_model.py`` の ``HeteroAttentionMessageLayer``
     - typed edge で destination node に入ってくる source node
   * - event readout attention
     - ``hetero_model.py`` の ``HeteroAttentiveReadout``
     - 1 event graph 内の detector nodes または pulse nodes

このうち、PyTorch の ``TransformerEncoder`` block なのは最初の waveform transformer だけです。
graph relation attention は ``torch.nn.MultiheadAttention`` そのものではなく、
event 内の全 object を全結合で見る Transformer block でもありません。
query/key/value と scaled dot-product という考え方は同じですが、attention の候補は物理 graph で制限されます。

.. code-block:: text

   Transformer-style sequence attention:
     token が基本的に他の全 token を見る

   TALE heterogeneous graph attention:
     node は edge_index にある incoming edge だけを見る
     relation type ごとに別の Q/K/V projection を使う
     edge_attr も key/value に入る

graph relation attention の物理的な読み替えは次です。

.. list-table::
   :header-rows: 1

   * - 名前
     - 直感的な意味
     - この model での意味
   * - Query, ``Q``
     - 「私は今、こういう情報がほしい」
     - destination node state から作る
   * - Key, ``K``
     - 「この source はこういう情報源です」
     - source node state と edge attribute から作る
   * - Value, ``V``
     - 「実際に渡す中身はこれです」
     - source node state と edge attribute から作る
   * - score
     - ``Q`` と ``K`` の相性
     - 1 edge / 1 head の softmax 前 logit
   * - attention weight
     - 正規化された edge 重み
     - 同じ relation で同じ destination に入る edge 同士の相対重み
   * - message
     - 実際に流れる情報
     - ``attention weight * V``

destination node ``j`` と incoming source node ``i`` が relation ``r`` でつながっている場合、次の計算です。

.. code-block:: text

   z_ij = concat(hidden_i, edge_attr_ij)

   q_ij = W_Q[r](hidden_j)
   k_ij = W_K[r](z_ij)
   v_ij = W_V[r](z_ij)

   score_ij = dot(q_j, k_ij) / sqrt(head_dim)
   alpha_ij = softmax(score_ij over incoming edges to j in relation r)
   message_j = sum_i alpha_ij * v_ij

そのため、現在の model は plain sequence Transformer ではなく、
relation-specific graph attention と呼ぶ方が正確です。
graph を全結合にしても、それだけで standard Transformer にはなりません。
detector / pulse の node type、relation ごとの projection、edge attribute、detector-level waveform、
type-wise readout を持つ点が残るからです。

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

Query はなぜ destination から作るのか
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

destination node は更新される側です。
同じ source pulse でも、ある destination pulse には時間的に整合的で、別の destination pulse には外れ値かもしれません。
そのため、destination node が「今ほしい情報」を Query として出します。
物理的には、destination pulse が「自分が shower front に乗っているか判断したい」と言い、
source pulse と edge が「私はこの時刻、電荷、距離、時間勾配を持つ」と答え、
Q と K の内積で相性を測る、と読むと分かりやすいです。

Key と Value はなぜ source + edge_attr から作るのか
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

通常の sequence Transformer では、Key と Value は source token の状態だけから作ることが多いです。
この model では ``source state + edge_attr`` から作ります。
TALE-SD では、同じ source pulse でも destination との距離、時間差、``dt_per_km``、
Ising weight によって意味が変わるからです。
Key に edge attribute を入れることで、この edge が destination の要求に合う情報源かを判断できます。
Value に edge attribute を入れることで、source の情報だけでなく、相対時刻、距離、Ising context も
message 本体として渡せます。

multi-head の shape
~~~~~~~~~~~~~~~~~~~

hidden dimension を ``H``、head 数を ``A`` とすると、1 head は
``d_head = H / A`` 次元です。relation ``r`` の edge 数を ``E_r`` とすると、
shape は次のようになります。

.. code-block:: text

   src_input = concat(src_state[src_index], edge_attr) -> [E_r, H + F_edge_r]
   query -> [E_r, A, d_head]
   key   -> [E_r, A, d_head]
   value -> [E_r, A, d_head]
   scores  = sum(query * key, dim=-1) / sqrt(d_head) -> [E_r, A]
   weights = scatter_softmax(scores, dst_index)      -> [E_r, A]
   messages = value * weights[:, :, None]            -> [E_r, A, d_head]

softmax は event 全体では取りません。
relation ごと、destination node ごとに、``dst_index`` が同じ edge の間で取ります。
同じ relation で、ある destination に入る edge が 1本だけなら、その relation 内の重みは実質的に 1 です。
その後、他 relation からの message や古い node state と組み合わされます。

小さい数値例
~~~~~~~~~~~~

1 head、2次元だけの例です。destination pulse ``P0`` に source pulse ``P1`` と ``P2`` が入るとします。

.. code-block:: text

   query(P0) = q = (1, 0)

   key(P1 -> P0) = k1 = (2.0, 0)
   key(P2 -> P0) = k2 = (0.5, 0)

   score1 = dot(q, k1) = 2.0
   score2 = dot(q, k2) = 0.5

   softmax(2.0, 0.5) ~= (0.82, 0.18)

   value(P1 -> P0) = v1 = (10, 1)
   value(P2 -> P0) = v2 = (2, 8)

   message(P0) = 0.82 * v1 + 0.18 * v2
               = (8.56, 2.26)

Key は「どの edge を強く見るか」に効き、Value は「実際に何を渡すか」に効きます。
Key と Value を分けることで、edge を信じる基準と、edge から渡す情報の中身を別々に学習できます。

relation ごとの情報の流れ
~~~~~~~~~~~~~~~~~~~~~~~~~

``pulse__same_detector_next__pulse`` / ``pulse__same_detector_prev__pulse``
   同じ detector 内の連続 pulse の間で情報を渡します。multi-pulse の時間順序を明示的に使えます。

``pulse__near_space__pulse``
   時間 cut なしで近傍 detector 上の pulse をつなぎます。局所的な shower footprint を保持し、
   delayed component や複雑な pulse 構造を hard drop せず入力に残します。

``pulse__time_causal__pulse``
   near-space のうち、時間整合性と Ising raw weight を満たす subset です。
   shower front timing と direction に最も直接関係する relation です。

``detector__near__detector``
   近傍 detector 間で局所 geometry、signal/no-signal context、local density を渡します。

``detector__observes__pulse``
   detector scalar/context/waveform embedding を、その detector が観測した pulse へ渡します。
   waveform を pulse node ごとに複製せずに、pulse が detector 文脈を受け取る経路です。

``pulse__observed_by__detector``
   pulse 情報を detector 側へ戻します。detector は Ising rejected candidate を含む pulse 群を要約し、
   detector-detector propagation や readout に渡せます。

message を集めた後、detector state と pulse state は residual connection、LayerNorm、feed-forward block で更新されます。

.. code-block:: text

   new_state = LayerNorm(old_state + update(old_state, aggregated_message))
   new_state = LayerNorm(new_state + FFN(new_state))

これは HGT を参考にした実装ですが、HGT そのものではありません。
PyG ``HGTConv`` は使っていません。HGSampling も使いません。
TALE では 1 event = 1 graph なので、event graph を丸ごと読みます。

attention weight は診断値として有用ですが、それだけで物理的因果性を証明するものではありません。
大きい attention weight は、その trained model のその layer / relation / head で、その edge の Value を強く使ったという意味です。
物理的な重要性を判断するには、event display、ablation、既存再構成との比較を合わせて確認します。

方向出力の正規化
----------------

reconstruction head は次の 6 成分を出します。

.. code-block:: text

   log10_energy_eV, core_x_km, core_y_km, dir_x, dir_y, dir_z

physics loss、NLL loss、quality target、predicted-error target を計算する前に、
scaled output を物理 target 単位へ戻し、方向 3 成分を正規化します。

.. code-block:: text

   u = (dir_x, dir_y, dir_z)
   n_hat = u / (||u|| + eps)

方向 loss は ``n_hat`` と true unit direction の開き角から計算します。
これにより、3成分出力の形式は保ちつつ、方向 vector の norm 自由度を reconstruction objective から取り除きます。

GAT、HGT、Transformer、Laplacian GCN との関係
----------------------------------------------

現在の実装は、名前を正確に使う必要があります。

.. list-table::
   :header-rows: 1

   * - reference family
     - 現行 model との関係
   * - GAT
     - learned attention で edge message に重みを付ける点は近い
   * - Transformer
     - query/key/value の scaled dot-product attention を使う点は近い
   * - HGT
     - node type と edge type に応じた relation-specific projection を持つ点は近い
   * - PyG ``HGTConv`` / ``HeteroConv``
     - 直接は使っていない。この repository 独自の layer を使う
   * - Laplacian / spectral GCN
     - 使っていない。``L = D - A`` や normalized Laplacian による propagation は出てこない

graph は dense adjacency matrix ではなく、sparse な ``edge_index`` tensor で表します。
message aggregation は ``_scatter_softmax`` と ``index_add_`` で行います。
graph construction には Ising edge feature 用の degree normalization が一部ありますが、
これは入力 feature の正規化であり、Laplacian graph convolution ではありません。

Readout
-------

再構成では node ごとの出力ではなく、event ごとに 1 つの出力が必要です。
そこで ``HeteroAttentiveReadout`` は detector node と pulse node を別々に集約します。
最後の message passing layer の後、model は次の state を持ちます。

.. code-block:: text

   detector states H_d -> [N_d, H]
   pulse states    H_p -> [N_p, H]

event ``b`` に属する detector node 集合を ``D_b``、pulse node 集合を ``P_b`` とします。
readout は node type ごとに3種類のsummaryを作ります。

mean pooling:

.. code-block:: text

   mean_detector_b = mean_{i in D_b} H_d[i]
   mean_pulse_b    = mean_{i in P_b} H_p[i]

max pooling:

.. code-block:: text

   max_detector_b = max_{i in D_b} H_d[i]
   max_pulse_b    = max_{i in P_b} H_p[i]

attention readout:

.. code-block:: text

   score_i = W_read H[i] + b_read
   beta_i  = softmax(score_i over nodes in the same event and node type)
   attn_b  = sum_i beta_i * H[i]

実装では ``readout_heads`` 個の readout attention head を使います。
hidden dimension を ``H``、readout head 数を ``R`` とすると、1つの node type から
``H * (2 + R)`` 次元が出ます。内訳は mean、max、``R`` 個の attention-weighted sum です。
detector と pulse は最後に結合します。

.. code-block:: text

   detector_readout -> [B, H * (2 + R)]
   pulse_readout    -> [B, H * (2 + R)]
   event vector     -> [B, 2 * H * (2 + R)]

detector と pulse を最初から同じ pool に混ぜず、それぞれの情報を event-level summary にしてから結合します。
そのため detector readout は shower footprint、detector geometry、signal/no-signal context、waveform summary を見やすく、
pulse readout は timing、charge、Ising annotation、delayed/multi-pulse structure を見やすくなります。

forward pass 中に変化するのは、event ごとの hidden state と attention weight です。
これらは一時的な値で、同じ trained model でも event が変われば変わります。
学習で更新されるのは、detector/context/pulse MLP、waveform encoder、relation ごとの
``W_Q``, ``W_K``, ``W_V`` と output projection、LayerNorm、readout attention、各 output head の重みです。
feature scaler は train split で fit され、checkpoint に保存される固定変換であり、
backpropagation で更新される neural-network weight ではありません。

Output heads
------------

event vector は task ごとの head に入ります。
各 head は同じ event vector にかかる小さな MLP です。
有効な head の出力は、reconstruction、mass、quality、predicted error の順に連結されます。

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

reconstruction head
~~~~~~~~~~~~~~~~~~~

reconstruction head は scaled target 空間で6成分を出します。

.. code-block:: text

   y_reco_hat = Head_reco(event_vector)
              = (log10_energy_eV, core_x_km, core_y_km, dir_x, dir_y, dir_z)

物理 loss と metrics を計算する前に、この6成分は target scaler で物理単位へ戻されます。
その後、方向3成分を正規化します。

.. code-block:: text

   u_hat = (dir_x, dir_y, dir_z)
   n_hat = u_hat / (||u_hat|| + eps)

これにより、出力は3成分のままですが、方向vectorのnorm自由度はreconstruction objectiveから取り除かれます。
最終評価は raw vector norm ではなく開き角で行います。

mass head
~~~~~~~~~

mass head は1つの logit ``z_mass`` を出します。
これは確率そのものではありません。diagnosticsで使う iron probability は次です。

.. code-block:: text

   P(iron) = sigmoid(z_mass) = 1 / (1 + exp(-z_mass))

``z_mass = 0`` なら iron probability は 50%、``z_mass = 2`` なら約 88%、
``z_mass = -2`` なら約 12% です。

quality head
~~~~~~~~~~~~

quality head は1つの logit を出します。
training では、その時点の reconstruction error から quality target を作ります。
実装では、物理単位へ戻した energy、core、angular error をそれぞれ scale し、
平均した score に対して ``exp(-score)`` を作り、
``binary_cross_entropy_with_logits`` で学習します。
つまり「この event の再構成が信頼できそうか」を event ごとに学習する補助 head です。

predicted-error head
~~~~~~~~~~~~~~~~~~~~

predicted-error head は3つの raw value を出します。
それらは ``softplus`` と物理 scale で変換されます。

.. code-block:: text

   predicted_errors = softplus(error_raw) * (energy_scale,
                                             angular_scale_deg,
                                             core_scale_km)

target は、その event の現在の reconstruction error です。
relative energy error、opening-angle error、xy core displacement をそれぞれscaleし、
``log1p(predicted_error)`` と ``log1p(target_error)`` を SmoothL1 で比較します。

今の比較方針では、auxiliary head を全部同時に入れません。
最初の transformer waveform sweep では、3つの dataset size に対して
quality-only と predicted-error-only の reco+mass を投げます。

forward pass全体を数式で見る
----------------------------

batch に対する処理は、次の順にまとめられます。

入力 tensor:

.. code-block:: text

   X_d = detector.x        -> [N_d, F_d]
   C_d = detector.context  -> [N_d, F_c]
   W_d = detector.waveform -> [N_d, C_waveform, T]
   X_p = pulse.x           -> [N_p, F_p]
   E_r = relation r ごとの (src, dst, edge_attr)

scaling:

.. code-block:: text

   X_d, C_d, X_p, edge_attr, target
     -> train split scaler で標準化

初期 encoder:

.. code-block:: text

   a_d = MLP_detector(X_d)
   c_d = MLP_context(C_d)
   w_d = WaveformEncoder(W_d)

   h_d^0 = Linear(concat(a_d, c_d, w_d))
   h_p^0 = MLP_pulse(X_p)

message passing layer ``l`` の relation ``r``:

.. code-block:: text

   z_ij^(r,l) = concat(h_src_i^l, edge_attr_ij^r)

   q_ij^(r,l) = W_Q^r h_dst_j^l + b_Q^r
   k_ij^(r,l) = W_K^r z_ij^(r,l) + b_K^r
   v_ij^(r,l) = W_V^r z_ij^(r,l) + b_V^r

   score_ij = dot(q_ij, k_ij) / sqrt(d_head)
   alpha_ij = softmax(score_ij over incoming edges to j in relation r)
   m_j^r = sum_i alpha_ij * v_ij

node update:

.. code-block:: text

   h_j <- LayerNorm(h_j + Update(h_j, m_j))
   h_j <- LayerNorm(h_j + FFN(h_j))

readout と head:

.. code-block:: text

   event_vector = Readout({h_d^L}, {h_p^L})

   y_reco_hat = Head_reco(event_vector)
   z_mass     = Head_mass(event_vector)
   z_quality  = Head_quality(event_vector)
   error_raw  = Head_error(event_vector)

学習時に変わるもの
------------------

forward pass 中に変わる一時的な値と、optimizer step で更新される重みは別です。

一時的な値:

- 各 layer の detector hidden state と pulse hidden state
- event、layer、relation、head、edge ごとの graph-relation attention weight
- detector node / pulse node の readout attention weight
- その batch の output prediction

これらは event ごとに再計算されます。
同じ trained checkpoint でも、event が変われば attention map は変わります。

学習で更新される重み:

- detector feature MLP
- detector context MLP
- pulse MLP
- waveform Conv1d と optional Transformer
- relation ごとの ``W_Q``, ``W_K``, ``W_V`` と relation output projection
- node update block、FFN、LayerNorm の scale/bias
- readout attention
- reconstruction、mass、quality、predicted-error head

training loop では、概念的に次の loss を計算します。

.. code-block:: text

   L = L_reco
       + lambda_mass    * L_mass
       + lambda_quality * L_quality
       + lambda_error   * L_error
       + optional bias penalties

そして backpropagation で learnable parameter を更新します。
scaler は別物です。scaler は train split で一度 fit され、checkpoint に保存される固定変換であり、
neural-network weight として更新されません。

出力ごとの物理的な効き方
------------------------

Energy
~~~~~~

energy には shower size が直接効きます。
detector signal sum、pulse charge、waveform integral/shape、hit/pulse multiplicity、
core位置との関係が主な情報です。
この model では、``detector.x``、``detector.waveform``、``pulse.x``、
pulse-pulse relation、detector-detector relation、readout を通して event vector へ入ります。
waveform encoder は detector ごとの calibrated waveform 全体を要約し、
pulse node は local pulse candidate と Ising annotation を保持します。

Core
~~~~

core には signal footprint、近傍 detector 配置、arrival-time pattern、signal/no-signal context が効きます。
直接関係する relation は ``detector__near__detector``、
``pulse__near_space__pulse``、``pulse__time_causal__pulse`` です。
near-space と time-causal を両方持つことで、幾何的には近いが時間的には怪しい pulse と、
幾何的にも時間的にも shower front と整合的な pulse を区別できます。

Direction
~~~~~~~~~

direction には pulse arrival time と detector position の関係が最も直接効きます。
pulse-pulse edge attribute には ``dt_usec``、``distance_km``、``dt_per_km`` が入ります。
edge attribute は key と value の両方に入るため、どの pulse pair を信じるか、
どの timing message を渡すかの両方に時間勾配情報を使えます。

Mass
~~~~

mass には waveform tail、delayed component、multi-pulse structure、
muon-richness に関係する情報が効きます。
そのため、この model は Ising rejected pulse candidate を ML graph から hard drop しません。
rejected candidate は Ising annotation 付きの pulse node として残り、detector waveform も detector node に保持されます。
これにより、通常の cleaning で落ちがちな delayed pulse や tail structure が mass head へ届く経路を残します。

Quality / predicted error
~~~~~~~~~~~~~~~~~~~~~~~~~

quality と predicted error は event-wise consistency を表す補助出力です。
pulse timing の不整合、geometry と signal の不一致、rejected candidate の多さ、
複雑な waveform、分散した attention などを持つ event では、quality が低く、
predicted error が大きくなる方向に学習されます。
これらは補助 head なので、次の比較では quality-only と predicted-error-only を
同じ reco+mass task の下で別々に試します。

短い説明
--------

論文や口頭説明では、次の言い方が最も正確です。

   この model は、TALE-SD event を detector node と pulse node からなる heterogeneous graph として表す。
   detector waveform は detector node に1回だけ保持し、pulse node は対応する detector と waveform window を参照する。
   graph では、同一 detector 内の連続 pulse、近傍 pulse pair、時間因果的に整合する pulse pair、
   detector-detector 近傍、detector-pulse 観測関係を別 relation として扱う。
   各 relation では、destination node から query を作り、source node と edge attribute から key/value を作る
   relation-specific multi-head attention で message passing を行う。
   最後に detector state と pulse state を別々に readout し、event vector から energy、core、arrival direction、
   mass、quality、predicted error を出力する。

さらに短く言うなら、

   TALE-SD の detector、waveform、pulse candidate、時空間 relation を、物理的な型と relation を保ったまま学習する、
   edge attribute 入り heterogeneous graph attention model です。

最重要ポイント
--------------

- plain Transformer ではありません。
- PyG ``HGTConv`` や ``HeteroConv`` をそのまま使う model でもありません。
- HGSampling は使わず、1 TALE event を 1つの full graph として保持します。
- waveform Transformer attention は、各 detector waveform 内の time bin 同士を見ます。
- graph relation attention は、``dstio`` が定義した graph edge に沿ってだけ message passing します。
- graph relation attention では、Query は destination node から、Key/Value は source node + edge_attr から作ります。
- readout attention は、event summary に効く detector node / pulse node を重く見ます。
- attention map は診断値であって、それだけで物理的説明にはなりません。ablation、event display、既存再構成との比較も必要です。

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

balanced HDF5 が揃い、split/input 分布を確認した後、最初の transformer 6本は次の入口で投げます。

.. code-block:: bash

   cd /dicos_ui_home/ikomae/work/src/talesd_gnn_reconstruction

   RUN_ID=<exportで使った同じRUN_ID> \
   SUBMIT_EXPORTS=0 \
   SUBMIT_TRAINING=1 \
   MODEL_ARCHITECTURE=hetero_attention \
   WAVEFORM_ENCODER=transformer \
   PARTITION=v100-al9_long \
   scripts/submit_server_hetero_dataset_size_sweep.sh

これは 50000、20000、10000 events/bin の HDF5 に対して、
quality-only と predicted-error-only の reco+mass 学習を投げます。
``cnn-gru`` はここでは同時に投げず、transformer の結果を見て採用条件を決めた後に比較します。
