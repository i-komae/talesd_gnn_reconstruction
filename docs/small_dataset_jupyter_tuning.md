# 小規模データセットと Jupyter チューニング

大規模 HDF5 は最終学習・性能評価には必要ですが、パラメーターの試行錯誤には重すぎます。
チューニング用には、既存の大規模 HDF5 shard から小規模 HDF5 を切り出して使います。

## DiCOS App の起動

公式マニュアルでは、DiCOS App は DiCOS web にログインして次のページから起動します。

```
https://dicos.grid.sinica.edu.tw/dockerapps/
```

使う App は `JupyterLab(GPU)` を基本にします。
公式表では `JupyterLab(GPU)` は CPU 12 core、GPU 1、Memory 128GB、SSD ありです。
大規模学習用の `b6000-al9_long` や `a100_long-al9` と同じ資源ではないので、ここでは小規模データセットでのパラメーター確認に限定します。

手順:

1. DiCOS web にログインする。
2. `https://dicos.grid.sinica.edu.tw/dockerapps/` を開く。
3. `JupyterLab(GPU)` を選ぶ。
4. `launch` を押す。
5. lifetime を選ぶ画面が出る場合は、作業時間に合わせて選ぶ。
6. 起動した JupyterLab で Terminal を開く。

起動後、Terminal で確認します。

```
hostname
nvidia-smi
df -hT /tmp "$HOME" /dicos_ui_home/ikomae /ceph/sharedfs/work/SATORI/ikomae 2>/dev/null
```

`nvidia-smi` が使えない場合は、GPU App ではなく CPU App に入っている可能性があります。
また、A100 App では driver が `12090` 程度で、`torch + cu130` では CUDA 初期化に失敗することがあります。
このリポジトリでは、Linux/Windows の `torch` を `pyproject.toml` と `uv.lock` で `cu128` に固定します。

## DiCOS 上の置き場所

公式マニュアルでは `/dicos_ui_home/<user_name>` は NFS の一時領域で、性能が限られ、定期的に消去される可能性があると説明されています。
また、Ceph の group working directory は job の working directory と保存用に使う領域とされています。

このため、基本は次の使い分けにします。

- リポジトリ: `/ceph/sharedfs/work/SATORI/ikomae/src/talesd_gnn_reconstruction`
- 大規模 HDF5: `/dicos_ui_home/ikomae/work/gnn/graphs/...` に既にあるものを入力として使う。ただし長期保存の前提にはしない。
- 新しい小規模 HDF5 と結果: 可能なら Ceph 側に置く。既存運用に合わせて `/dicos_ui_home/ikomae/work/gnn/...` を使う場合も、重要な結果は後で退避する。
- App 内の一時高速領域: `/tmp`。その App instance が消えると失われる前提で使う。

JupyterLab の Terminal で作業ディレクトリへ移動します。

```
cd /ceph/sharedfs/work/SATORI/ikomae/src/talesd_gnn_reconstruction
git pull
uv sync
```

Linux では `uv sync` により `torch==2.11.0+cu128` が入ります。
Jupyter からも script からも、この同じ `.venv` を使います。

## 小規模 HDF5 の作成

DST を読み直さず、作成済み graph HDF5 から event group をコピーします。
energy bin と粒子種で層化するので、少数サンプルでも proton/iron と energy の偏りを抑えられます。
大規模 HDF5 を全 scan するため、この処理は Jupyter や login shell ではなく Slurm に投げます。

```
RUN_NAME=small_graph_energyflat2000_20260526_141104 \
PER_BIN=2000 \
GRAPH_INPUT=/dicos_ui_home/ikomae/work/gnn/graphs/server_graph_export_energyflat200000_20260524_075508 \
scripts/submit_server_small_graph_dataset.sh
```

出力:

```
/dicos_ui_home/ikomae/work/gnn/graphs/small_graph_energyflat2000_20260526_141104/small_graph_energyflat2000_20260526_141104.h5
/dicos_ui_home/ikomae/work/gnn/graphs/small_graph_energyflat2000_20260526_141104/small_graph_energyflat2000_20260526_141104-perbin500.h5
/dicos_ui_home/ikomae/work/gnn/graphs/small_graph_energyflat2000_20260526_141104/small_graph_energyflat2000_20260526_141104-perbin200.h5
/dicos_ui_home/ikomae/work/gnn/outputs/talesd_gnn_reconstruction/runs/small_graph_energyflat2000_20260526_141104/
```

`--per-bin 2000 --stratify-particle` は、proton/iron それぞれの log10(E/eV) 0.1 bin ごとに最大 2000 event を残します。
既定で `EXTRA_PER_BINS=500,200` が使われるため、同じ scan 結果から `*-perbin500.h5` と `*-perbin200.h5` の shard も同時に作ります。
既に `PER_BIN=2000` の small がある場合は、2000 を作り直さずに派生サイズだけ作れます。

```
RUN_NAME=small_graph_energyflat2000_20260526_141104 \
DERIVE_ONLY=1 \
GRAPH_INPUT=/dicos_ui_home/ikomae/work/gnn/graphs/small_graph_energyflat2000_20260526_141104 \
scripts/submit_server_small_graph_dataset.sh
```

`OUTPUT_SHARDS=auto` では、最大サイズの出力は元の巨大HDF5を並列に読むため入力shard単位で分割し、小さい派生データは event 数から分割数を決めます。既定の `TARGET_EVENTS_PER_SHARD=20000` なら、`PER_BIN=500` や `200` は1個または少数shardにまとまります。
より軽くしたい場合は `PER_BIN=200` や `MAX_TOTAL=50000` を使います。
作成された HDF5 path は run directory の `config/graph_inputs.txt` にも保存されます。
既定では 30 秒ごとに `scan small graph files` と `write small graph shards` の進捗が出ます。
間隔を変える場合は `PROGRESS_INTERVAL=10` のように指定します。
scan は HDF5 shard 単位で並列化されます。
CPU 要求数は既存の DST export と同じ `AUTO_RESOURCES=1` の資源選択で決めます。
`SCAN_WORKERS=auto` では、選ばれた CPU 数と入力 HDF5 shard 数から worker 数を決めます。

## Script として実行

Notebook と同じ JSON 設定を script から読みます。

```
.venv/bin/python scripts/run_small_tuning.py \
  --config configs/small_tuning_example.json \
  --dry-run
```

実際に学習する場合:

```
.venv/bin/python scripts/run_small_tuning.py \
  --config configs/small_tuning_example.json \
  --set run_name=small_reco_quality_trial01 \
  --set config_name=small_reco_quality_trial01 \
  --set train.epochs=8 \
  --set train.device=cuda
```

`--set` は JSON の dotted key を上書きします。
例えば `train.hidden_dim=192`、`train.num_layers=5`、`train.learning_rate=3e-4` のように指定できます。

## Jupyter で実行

DiCOS App の JupyterLab で、このリポジトリを開きます。
Notebook は次を使います。

```
notebooks/small_dataset_tuning.ipynb
```

Kernel が見えない場合は、同じ `.venv` を Jupyter kernel として登録します。

```
.venv/bin/python -m ipykernel install --user --name talesd-gnn --display-name "TALE GNN"
```

JupyterLab では kernel を `TALE GNN` に切り替えます。
これは `.venv` の中身を別環境に作り直す操作ではありません。

Jupyter 上で GPU が見えているかは notebook か Terminal で確認します。

```
.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## 両立の仕組み

共通入口は `src/talesd_gnn_reconstruction/tuning.py` です。

- `scripts/run_small_tuning.py` は JSON を読み、`tuning.run_training_from_config()` を呼びます。
- `notebooks/small_dataset_tuning.ipynb` も同じ関数を呼びます。
- そのため、script と Jupyter で学習処理本体は分岐しません。

小規模 HDF5 は `metadata/source_path` を保持するので、既定の `split_mode=source-stratified` が使えます。
ただし、極端に小さいデータでは source 単位の分割が不安定になる場合があります。
その場合だけ `--set train.split_mode=event` で動作確認用に切り替えます。

## 注意点

小規模データセットはパラメーター探索と実装確認用です。
最終的な精度、mass accuracy、reconstruction resolution の判断には大規模 HDF5 を使います。

参照した公式マニュアル:

- DiCOSApps: https://dicos.grid.sinica.edu.tw/static/docs/dicosapp.html
- Storage and Data Transfer: https://dicos.grid.sinica.edu.tw/static/docs/data_storage_transfer.html
