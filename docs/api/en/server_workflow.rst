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

Rules
-----

- Do not silently change graph, task, loss, epoch count, or split for full training.
- Do not mix a failed-job rerun with an improvement experiment.
- Do not submit the same training to multiple resources unless explicitly requested.
- Do not use B6000 for production until the CUDA/cuBLAS preflight passes.
- Check driver/CUDA compatibility before using A100.
- Create local graph cache after Slurm allocation. Do not rsync before submission.

