サーバー実行フロー
==================

Slurmでの本番実行では、直接 ``talesd-gnn`` を打つよりもsubmitterを使います。submitterがresource、run directory、local cache、runtime copy、環境変数を管理します。

標準submitter
-------------

.. list-table::
   :header-rows: 1

   * - 用途
     - script
   * - HDF5 graph export
     - ``scripts/submit_server_graph_export.sh``
   * - 小規模HDF5作成
     - ``scripts/submit_server_small_graph_dataset.sh``
   * - 再構成 quality-only
     - ``scripts/submit_server_waveform_full_training.sh``
   * - mass-only
     - ``scripts/submit_server_mass_only_training.sh``
   * - reco+mass
     - ``scripts/submit_server_reco_mass_training.sh``

再構成quality-onlyの例
----------------------

.. code-block:: bash

   GRAPH_INPUT=/dicos_ui_home/ikomae/work/gnn/graphs/flat50000 \
   PARTITION=v100-al9_long \
   RUN_NAME=flat50000_reco_v100_128epoch_$(date +%Y%m%d_%H%M%S) \
   scripts/submit_server_waveform_full_training.sh

mass focalの例
--------------

.. code-block:: bash

   GRAPH_INPUT=/dicos_ui_home/ikomae/work/gnn/graphs/flat50000 \
   PARTITION=v100-al9_long \
   RUN_NAME=flat50000_mass_focal_v100_128epoch_$(date +%Y%m%d_%H%M%S) \
   MASS_LOSS_MODE=focal \
   MASS_FOCAL_GAMMA=2.0 \
   MASS_RANKING_WEIGHT=0.5 \
   scripts/submit_server_mass_only_training.sh

注意点
------

- full trainingではgraph、task、loss、epoch、splitを勝手に変えない。
- failed jobの投げ直しと改善実験を混ぜない。
- 同じ学習を複数resourceへ重複投入しない。
- B6000はCUDA/cuBLAS preflightが通るまで本番投入しない。
- A100はdriver/CUDA互換性を確認してから使う。
- local graph cacheはSlurm allocation後に作る。submit前にrsyncしない。

