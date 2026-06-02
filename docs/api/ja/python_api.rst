Python API
==========

CLIではなくPythonから直接呼ぶ場合の主要入口です。大量学習ではSlurm submitterを使い、notebookや小規模検証ではここに示すAPIを使います。

学習
----

.. code-block:: python

   from talesd_gnn_reconstruction.train import train_model

   result = train_model(
       graphs_path="/path/to/graphs/flat50000",
       output_path="/path/to/checkpoints/reconstruction.pt",
       epochs=128,
       batch_size=256,
       training_task="reconstruction",
       model_architecture="physics",
       waveform_encoder="cnn-gru",
       loss_mode="physics",
       quality_prediction=True,
       split_mode="source-stratified",
       val_fraction=0.05,
       test_fraction=0.10,
       device="cuda",
   )

``train_model`` はdataset初期化、split、scaler fitting、DataLoader作成、学習、validation、best checkpoint保存、test評価、diagnostics作成までをまとめて実行します。

HDF5 datasetの読み込み
----------------------

.. code-block:: python

   from talesd_gnn_reconstruction.dataset import H5GraphDataset

   dataset = H5GraphDataset(
       "/path/to/graphs/flat50000",
       require_target=True,
       load_particle_label=True,
   )

   sample = dataset[0]
   dataset.close()

``sample`` には ``node_features``、``edge_features``、``waveform_features``、``target``、metadataが入ります。

推論
----

.. code-block:: python

   from talesd_gnn_reconstruction.predict import predict_graphs

   csv_path = predict_graphs(
       graphs_path="/path/to/graphs/data_graphs.h5",
       checkpoint_path="/path/to/checkpoints/reconstruction.pt",
       output_csv="/path/to/predictions.csv",
       batch_size=256,
       device="cuda",
   )

入力分布
--------

.. code-block:: python

   from talesd_gnn_reconstruction.feature_analysis import save_input_distributions

   summary = save_input_distributions(
       graphs_path="/path/to/graphs/flat50000",
       output_dir="/path/to/input_distributions",
       max_graphs=100000,
       show_progress=True,
   )

特徴量group重要度
-----------------

.. code-block:: python

   from talesd_gnn_reconstruction.feature_analysis import save_feature_group_importance

   result = save_feature_group_importance(
       graphs_path="/path/to/graphs/flat50000",
       checkpoint_path="/path/to/checkpoints/reconstruction.pt",
       output_dir="/path/to/feature_importance",
       split="validation",
       max_graphs=50000,
       batch_size=256,
       device="cuda",
   )

HDF5 graphの手動作成
--------------------

通常は ``talesd-gnn export`` を使います。手動でgraphを書き込む場合の低レベルAPIは以下です。

.. code-block:: python

   from talesd_gnn_reconstruction.graph_io import create_graph_file, write_graph

   with create_graph_file("/path/to/graphs.h5", config={"kind": "mc"}) as handle:
       write_graph(handle, 0, graph_event)

``graph_event`` は ``event_graph.build_graph_event`` が返す ``GraphEvent`` です。

