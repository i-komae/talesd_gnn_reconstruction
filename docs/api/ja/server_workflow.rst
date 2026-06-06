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
   * - heterogeneous reco+mass
     - ``scripts/submit_server_hetero_training.sh``
   * - heterogeneous reco+mass, quality-only補助head
     - ``scripts/submit_server_hetero_reco_mass_quality_training.sh``
   * - heterogeneous reco+mass, predicted-error-only補助head
     - ``scripts/submit_server_hetero_reco_mass_error_training.sh``

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

heterogeneous reco+mass比較
---------------------------

これらの submitter は task を reco+mass のままにし、``LOSS_MODE=physics`` を保ちます。
1本目は quality head だけを有効にします。
2本目は predicted-error head だけを有効にします。
同じ heterogeneous graph input に対する別 run として比較します。

.. code-block:: bash

   GRAPH_INPUT=/dicos_ui_home/ikomae/work/gnn/graphs/<hetero_graph_dir> \
   PARTITION=v100-al9_long \
   RUN_NAME=hetero_reco_mass_quality_v100_128epoch_$(date +%Y%m%d_%H%M%S) \
   scripts/submit_server_hetero_reco_mass_quality_training.sh

.. code-block:: bash

   GRAPH_INPUT=/dicos_ui_home/ikomae/work/gnn/graphs/<hetero_graph_dir> \
   PARTITION=v100-al9_long \
   RUN_NAME=hetero_reco_mass_error_v100_128epoch_$(date +%Y%m%d_%H%M%S) \
   scripts/submit_server_hetero_reco_mass_error_training.sh

``scripts/submit_server_hetero_training.sh`` は既定では ``FEATURE_IMPORTANCE=0`` です。
同じSlurm job内で学習後のgroup ablationも走らせる場合だけ ``FEATURE_IMPORTANCE=1`` を指定します。

注意点
------

- full trainingではgraph、task、loss、epoch、splitを勝手に変えない。
- failed jobの投げ直しと改善実験を混ぜない。
- 同じ学習を複数resourceへ重複投入しない。
- B6000はCUDA/cuBLAS preflightが通るまで本番投入しない。
- A100はdriver/CUDA互換性を確認してから使う。
- local graph cacheはSlurm allocation後に作る。submit前にrsyncしない。
