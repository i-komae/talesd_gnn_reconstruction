Heterogeneous DST 再構成 workflow
=================================

heterogeneous path は、学習後に DST を直接再構成するための現行経路です。
graph の物理的な定義は ``dstio.tale.graph`` に寄せ、この repository では HDF5 学習 cache、model input 変換、学習、診断、直接推論を扱います。

.. figure:: ../fig/hetero_dst_workflow.svg
   :alt: DSTからheterogeneous graph、学習cache、直接再構成へ進むTALE-SD workflow。
   :width: 100%

   detector waveform は detector node に 1 回だけ保存します。pulse node は ``pulse_detector_index`` と ``pulse_bounds`` を持ち、同じ waveform を pulse node ごとに複製せずに参照します。

Graph schema
------------

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
     - ``pulse__interacts__pulse``, ``detector__near__detector``, ``detector__observes__pulse``

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

Direct reconstruction path
--------------------------

``reconstruct-dst`` は DST を直接読みます。
``train-hetero`` と同じ graph schema と checkpoint scaler を使い、中間 HDF5 graph を書きません。

.. code-block:: text

   DST
     -> talesd-gnn reconstruct-dst
       -> dstio.tale.graph.iter_graphs
         -> hetero_data.sample_to_hetero_data
           -> MinimalHeteroTaleSdGNN checkpoint
             -> reconstruction CSV

heterogeneous model を学習した後の大量 data / MC 一回通し再構成では、この direct path を使います。
