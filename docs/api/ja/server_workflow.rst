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
   * - balanced heterogeneous HDF5 export とサイズ比較
     - ``scripts/submit_server_hetero_dataset_size_sweep.sh``

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
明示的に上書きしない限り、heterogeneous model architecture は ``hetero_attention`` です。
event graph は丸ごと使い、HGSampling は使いません。
最初の waveform encoder 比較では ``WAVEFORM_ENCODER=transformer`` を使います。
transformer の結果で dataset size と auxiliary head 条件を決めるまで、対応する ``cnn-gru`` sweep は投げません。
transformer waveform run では、submitter の既定を GPU micro-batch
``BATCH_SIZE=32``、``GRADIENT_ACCUMULATION_STEPS=4`` にします。
effective batch size は 128 のまま保ち、``BATCH_SIZE=128`` をそのまま
waveform Transformer に入れた時の大きな activation memory を避けます。
log に出る ``hetero_loader_memory`` は CPU/DataLoader prefetch の見積もりであり、
GPU activation memory の保証ではありません。

.. code-block:: bash

   GRAPH_INPUT=/dicos_ui_home/ikomae/work/gnn/graphs/<hetero_graph_dir> \
   PARTITION=v100-al9_long \
   WAVEFORM_ENCODER=transformer \
   RUN_NAME=hetero_reco_mass_quality_v100_128epoch_$(date +%Y%m%d_%H%M%S) \
   scripts/submit_server_hetero_reco_mass_quality_training.sh

.. code-block:: bash

   GRAPH_INPUT=/dicos_ui_home/ikomae/work/gnn/graphs/<hetero_graph_dir> \
   PARTITION=v100-al9_long \
   WAVEFORM_ENCODER=transformer \
   RUN_NAME=hetero_reco_mass_error_v100_128epoch_$(date +%Y%m%d_%H%M%S) \
   scripts/submit_server_hetero_reco_mass_error_training.sh

``scripts/submit_server_hetero_training.sh`` は既定では ``FEATURE_IMPORTANCE=0`` です。
同じSlurm job内で学習後のgroup ablationも走らせる場合だけ ``FEATURE_IMPORTANCE=1`` を指定します。
一方で ``ATTENTION_MAPS=1`` は既定です。checkpoint 作成後に、validation の少数eventについて
``<checkpoint>.diagnostics/attention_maps/validation/`` へ attention map を保存します。
既定の保存数は ``ATTENTION_MAPS_MAX_GRAPHS=16`` なので、これは全dataset dumpではなく軽量診断です。

balanced heterogeneous HDF5 サイズ比較
---------------------------------------

次の balanced dataset 作成には ``scripts/submit_server_hetero_dataset_size_sweep.sh`` を使います。
最終的な既定サイズは true energy / particle bin ごとに ``50000``, ``20000``, ``10000`` event を選ぶ3種類のHDF5です。
この script はまず overdraw した pool dataset を1つ作り、その pool から reshard/subsample して3種類の最終HDF5を作ります。
balanced export は ``DAT??????`` source group、zenith bin、azimuth bin、core位置bin、event時刻binで候補を分散させ、selection summary と train/validation/test split distribution summary を出します。

この sweep の既定 split は、train/validation/test の source group が約 ``45/10/45`` です。
同じ ``DAT??????`` source group は split 間で共有しません。
validation は early stopping と model selection 用の独立 source-group holdout とし、test は最終比較用として触らない split にします。

まず HDF5 export stage を投げます。既定では pool export 1本と、それに依存する reshard 3本が投入されます。

.. code-block:: bash

   RUN_ID=hetero_balance_$(date +%Y%m%d_%H%M%S) \
   scripts/submit_server_hetero_dataset_size_sweep.sh

3種類の最終HDF5とsummaryが完了したら、以下を確認します。

.. code-block:: text

   /dicos_ui_home/ikomae/work/gnn/graphs/hetero_balanced_flat*/summaries/hetero_selection_summary.json
   /dicos_ui_home/ikomae/work/gnn/graphs/hetero_balanced_flat*/summaries/split_distribution_summary.json
   /dicos_ui_home/ikomae/work/gnn/graphs/hetero_balanced_flat*/summaries/split_distributions/split_distribution_plot_data.json
   /dicos_ui_home/ikomae/work/gnn/graphs/hetero_balanced_flat*/summaries/split_distributions/

``split_distribution_plot_data.json`` には PDF に使った histogram bins、
counts/densities、energy-bin count curve が入ります。
見た目だけを後から調整する場合は、HDF5 を再 scan せずにこの JSON から再描画できます。

その後、同じHDF5に対して6本の reco+mass 比較学習を投げます。

.. code-block:: bash

   RUN_ID=<exportで使った同じRUN_ID> \
   SUBMIT_EXPORTS=0 \
   SUBMIT_TRAINING=1 \
   WAVEFORM_ENCODER=transformer \
   scripts/submit_server_hetero_dataset_size_sweep.sh

6本の内訳は、``50000``, ``20000``, ``10000`` events/bin それぞれに対する ``quality-only`` と ``predicted-error-only`` です。
これは最初の transformer waveform sweep です。``cnn-gru`` はこの結果を見て条件を選んだ後に比較します。

注意点
------

- full trainingではgraph、task、loss、epoch、splitを勝手に変えない。
- failed jobの投げ直しと改善実験を混ぜない。
- 同じ学習を複数resourceへ重複投入しない。
- B6000はCUDA/cuBLAS preflightが通るまで本番投入しない。
- A100はdriver/CUDA互換性を確認してから使う。
- local graph cacheはSlurm allocation後に作る。submit前にrsyncしない。
