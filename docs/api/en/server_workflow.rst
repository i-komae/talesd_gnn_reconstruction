Server Workflow
===============

For Slurm production runs, use the submitter scripts instead of invoking ``talesd-gnn`` directly. The submitters manage resources, run directories, local graph cache, runtime copy, and environment variables.

Standard submitters
-------------------

.. list-table::
   :header-rows: 1

   * - Purpose
     - Script
   * - HDF5 graph export
     - ``scripts/submit_server_graph_export.sh``
   * - Small HDF5 dataset creation
     - ``scripts/submit_server_small_graph_dataset.sh``
   * - Reconstruction quality-only
     - ``scripts/submit_server_waveform_full_training.sh``
   * - Mass-only
     - ``scripts/submit_server_mass_only_training.sh``
   * - Reco+mass
     - ``scripts/submit_server_reco_mass_training.sh``
   * - Heterogeneous reco+mass
     - ``scripts/submit_server_hetero_training.sh``
   * - Heterogeneous reco+mass, quality-only auxiliary head
     - ``scripts/submit_server_hetero_reco_mass_quality_training.sh``
   * - Heterogeneous reco+mass, predicted-error-only auxiliary head
     - ``scripts/submit_server_hetero_reco_mass_error_training.sh``
   * - Balanced heterogeneous HDF5 export and size sweep
     - ``scripts/submit_server_hetero_dataset_size_sweep.sh``

Reconstruction quality-only example
-----------------------------------

.. code-block:: bash

   GRAPH_INPUT=/dicos_ui_home/ikomae/work/gnn/graphs/flat50000 \
   PARTITION=v100-al9_long \
   RUN_NAME=flat50000_reco_v100_128epoch_$(date +%Y%m%d_%H%M%S) \
   scripts/submit_server_waveform_full_training.sh

Mass focal example
------------------

.. code-block:: bash

   GRAPH_INPUT=/dicos_ui_home/ikomae/work/gnn/graphs/flat50000 \
   PARTITION=v100-al9_long \
   RUN_NAME=flat50000_mass_focal_v100_128epoch_$(date +%Y%m%d_%H%M%S) \
   MASS_LOSS_MODE=focal \
   MASS_FOCAL_GAMMA=2.0 \
   MASS_RANKING_WEIGHT=0.5 \
   scripts/submit_server_mass_only_training.sh

Heterogeneous reco+mass comparison
----------------------------------

These submitters keep the task as reco+mass and keep ``LOSS_MODE=physics``.
The first run enables the quality head only.
The second run enables the predicted-error head only.
They should be compared as separate runs on the same heterogeneous graph input.
Unless explicitly overridden, the heterogeneous model architecture is ``hetero_attention``.
It uses full event graphs and does not use HGSampling.
The first waveform-encoder sweep should use ``WAVEFORM_ENCODER=transformer``.
Do not launch the matching ``cnn-gru`` sweep until the transformer results determine the dataset size and auxiliary-head condition.
For transformer waveform runs, the submitter defaults to GPU micro-batch
``BATCH_SIZE=32`` with ``GRADIENT_ACCUMULATION_STEPS=4``. This keeps the
effective batch size at 128 while avoiding the larger activation memory of
``BATCH_SIZE=128`` inside the waveform Transformer. The printed
``hetero_loader_memory`` line is a CPU/DataLoader prefetch estimate; it is not a
GPU activation-memory guarantee.

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

``scripts/submit_server_hetero_training.sh`` sets ``FEATURE_IMPORTANCE=0`` by default.
Set ``FEATURE_IMPORTANCE=1`` only when post-training group ablation should run in the same Slurm job.
It sets ``ATTENTION_MAPS=1`` by default. After the checkpoint is written, the
runner saves validation attention maps for a small event sample under
``<checkpoint>.diagnostics/attention_maps/validation/``. The default sample size
is ``ATTENTION_MAPS_MAX_GRAPHS=16`` so this is a lightweight diagnostic, not a
full-dataset attention dump.

Balanced heterogeneous HDF5 size sweep
--------------------------------------

Use ``scripts/submit_server_hetero_dataset_size_sweep.sh`` to make the next balanced datasets.
The default sizes are ``50000``, ``20000``, and ``10000`` selected events per true-energy/particle bin.
The balanced export preselects candidates by ``DAT??????`` source group, zenith bin, azimuth bin, core-position bin, and event-time bin, then writes a selection summary and a train/validation/test split distribution summary.

The default split for this sweep is source-group based ``45/10/45`` for train/validation/test.
The same ``DAT??????`` source group is not shared across splits.
Validation is kept as an independent source-group holdout for early stopping and model selection; the test split is left for final comparison.

First submit the three export jobs:

.. code-block:: bash

   RUN_ID=hetero_balance_$(date +%Y%m%d_%H%M%S) \
   scripts/submit_server_hetero_dataset_size_sweep.sh

After the three HDF5 exports and their summaries are complete, inspect:

.. code-block:: text

   /dicos_ui_home/ikomae/work/gnn/graphs/hetero_balanced_flat*/summaries/hetero_selection_summary.json
   /dicos_ui_home/ikomae/work/gnn/graphs/hetero_balanced_flat*/summaries/split_distribution_summary.json
   /dicos_ui_home/ikomae/work/gnn/graphs/hetero_balanced_flat*/summaries/split_distributions/split_distribution_plot_data.json
   /dicos_ui_home/ikomae/work/gnn/graphs/hetero_balanced_flat*/summaries/split_distributions/

The split plot-data JSON stores the histogram bins, histogram counts/densities,
and energy-bin count curves used by the PDFs, so cosmetic changes can be made
later without rescanning the HDF5 dataset.

Then submit the six reco+mass comparison runs on those HDF5 files:

.. code-block:: bash

   RUN_ID=<same RUN_ID used for export> \
   SUBMIT_EXPORTS=0 \
   SUBMIT_TRAINING=1 \
   WAVEFORM_ENCODER=transformer \
   scripts/submit_server_hetero_dataset_size_sweep.sh

The six training jobs are ``quality-only`` and ``predicted-error-only`` for each of ``50000``, ``20000``, and ``10000`` events per bin.
This is the first transformer waveform sweep. The ``cnn-gru`` comparison should be run later only for the selected condition.

Rules
-----

- Do not silently change graph, task, loss, epoch count, or split for full training.
- Do not mix a failed-job rerun with an improvement experiment.
- Do not submit the same training to multiple resources unless explicitly requested.
- Do not use B6000 for production until the CUDA/cuBLAS preflight passes.
- Check driver/CUDA compatibility before using A100.
- Create local graph cache after Slurm allocation. Do not rsync before submission.
