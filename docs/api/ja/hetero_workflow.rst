Heterogeneous DST 再構成 workflow
=================================

heterogeneous path は、学習後に DST を直接再構成するための現行経路です。
graph の物理的な定義は ``dstio.tale.graph`` に寄せ、この repository では HDF5 学習 cache、model input 変換、学習、診断、直接推論を扱います。

モデル内部の詳しい説明は :doc:`hetero_model` にあります。
そのページでは、PyTorch / PyG 公式ドキュメントの書き方に合わせて、この repository で実際に使う tensor、encoder、relation attention、readout、loss head、直接推論の流れを説明しています。

最初の図は workflow 図です。HDF5 学習 cache 経路と DST 直接再構成経路を分けて示します。

.. figure:: ../fig/hetero_dst_workflow.svg
   :alt: DSTからheterogeneous graph、学習cache、直接再構成へ進むTALE-SD workflow。
   :width: 100%

   detector waveform は detector node に 1 回だけ保存します。pulse node は ``pulse_detector_index`` と ``pulse_bounds`` を持ち、同じ waveform を pulse node ごとに複製せずに参照します。

Graph schema
------------

次の図は graph schema 図です。1つの ``GraphEvent`` の中で使う node type と edge type を graph として示します。
detector node と pulse node は別の node type です。
detector waveform は detector node 側に 1 回だけ保存し、pulse node は ``pulse_detector_index`` と ``pulse_bounds`` で waveform の一部を参照します。
Ising-kept pulse candidate と Ising-rejected pulse candidate はどちらも ML graph に残します。

.. figure:: ../fig/hetero_graph_schema.svg
   :alt: detector node、pulse node、typed edge、detector waveform、pulse bounds、Ising kept/rejected pulse candidates を示す heterogeneous TALE-SD graph schema。
   :width: 100%

   detector-detector、pulse-pulse、detector-pulse relation は別 edge type です。Ising rejected pulse candidate は hard drop せず、annotation付きで入力に残します。

``dstio.tale.graph.iter_graphs`` は
``tale_sd_hetero_ising_pulse_detector_graph_v1`` の ``GraphEvent`` を返します。
ML graph の標準は ``node_policy="all_candidates_with_ising"`` です。
Ising で rejected になった pulse candidate も graph から消さず、Ising annotation feature を持たせます。
``node_policy="ising_kept"`` は reconstruction-cleaned subset が必要な時だけ使います。

node と relation は次の通りです。

.. list-table::
   :header-rows: 1

   * - 種類
     - 保存する field
   * - detector node
     - ``detector_features``, ``detector_context_features``, ``detector_positions_km``, ``detector_lids``, ``detector_waveforms``
   * - pulse node
     - ``pulse_features``, ``pulse_positions_km``, ``pulse_lids``, ``pulse_detector_index``, ``pulse_bounds``
   * - relation
     - ``pulse__same_detector_next__pulse``, ``pulse__same_detector_prev__pulse``, ``pulse__near_space__pulse``, ``pulse__time_causal__pulse``, ``detector__near__detector``, ``detector__observes__pulse``, ``pulse__observed_by__detector``

core-relative pulse feature は Ising reference core がある時だけ有効です。
そのため、学習用 export では通常 ``--require-reference-core`` を使います。

Training cache path
-------------------

``export-hetero`` は繰り返し学習で読むための HDF5 graph shard を書きます。
この HDF5 は cache であり、最終的な再構成 interface ではありません。

.. code-block:: text

   DST
     -> talesd-gnn export-hetero
       -> dstio.tale.graph.iter_graphs
         -> hetero_graph_io.py
           -> heterogeneous HDF5 shards

``train-hetero`` はこの shard を読み、train split で scaler を fit し、heterogeneous model を学習して checkpoint を保存します。
既定 architecture は ``hetero_attention`` です。detector / pulse relation ごとの multi-head attention と、detector / pulse type 別 attention readout を使います。
HGSampling は使いません。TALE では 1 event を 1 graph として丸ごと扱い、detector、pulse、waveform、Ising rejected pulse の情報を sampling で落としません。
最初の waveform encoder 比較では、6本の reco+mass size-sweep job を ``WAVEFORM_ENCODER=transformer`` で投げます。``cnn-gru`` は、その結果を見て採用条件を決めた後に比較します。

Direct reconstruction path
--------------------------

``reconstruct-dst`` は DST を直接読みます。
``train-hetero`` と同じ graph schema と checkpoint scaler を使い、中間 HDF5 graph を書きません。

.. code-block:: text

   DST
     -> talesd-gnn reconstruct-dst
       -> dstio.tale.graph.iter_graphs
         -> hetero_data.sample_to_hetero_data
           -> hetero_attention checkpoint
             -> reconstruction CSV

heterogeneous model を学習した後の大量 data / MC 一回通し再構成では、この direct path を使います。
