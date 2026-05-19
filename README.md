# TALE-SD GNN Reconstruction

TALE-SD の不規則な検出器配置をグラフとして扱い、GNNでエネルギー、コア位置、到来方向を推定するための新規 `uv` プロジェクトです。

この初期版は次の流れを実装しています。

1. TALE-SD data DST (`talesdcalibev`) または MC DST (`rusdraw` + `rusdmc`) を読む
2. waveform から上下層 coincidence pulse を抽出する
3. `coincidence_analysis` の Ising graph と同じ条件で、上下層coincident pulse候補をノードにしたHDF5グラフへ変換する
4. MC truth付きグラフでGNNを学習する
5. 学習済みモデルで data / MC を再構成しCSVへ出力する

## セットアップ

```bash
cd /Users/ikomae/TALE/talesd_gnn_reconstruction
export DSTDIR=/path/to/dst2k-ta
uv sync
```

`dstio` は親ディレクトリの `../dstio` を editable path dependency として参照します。SD座標はテキスト表ではなく `talesdconst_pass2.dst` または `talesdcalibev` bank内のcalibration情報から読みます。

C++/pybind11拡張を明示的に再ビルドする場合:

```bash
./build_extensions.sh
```

## 基本コマンド

MC DSTをグラフへ変換:

```bash
uv run talesd-gnn export /path/to/mc.dst.gz \
  --kind mc \
  --const-dst $TADIR/data/SD/talesdconst_pass2.dst \
  --workers 4 \
  --shard-size 100000 \
  -o outputs/mc_graphs.h5
```

大量MCでは、ディレクトリ指定とenergy-flat reservoir samplingを使います。

```bash
uv run talesd-gnn export \
  --input-dir /path/to/tale_proton5.5yr_16-16.9 \
  --input-dir /path/to/tale_proton5.5yr_17-17.9 \
  --input-dir /path/to/tale_proton5.5yr_18-18.9 \
  --kind mc \
  --const-dst $TADIR/data/SD/talesdconst_pass2.dst \
  --energy-sample-per-bin 10000 \
  --energy-bin-width 0.1 \
  --energy-oversample-factor 2 \
  --workers 8 \
  --worker-max-files 200 \
  --shard-size 50000 \
  --skip-errors \
  -o outputs/graphs/talesd_mc_energy_flat.h5
```

通常の大量MC exportでは `--workers` はファイル単位の並列数です。energy sampling時は、先にmetadata scanで候補event indexだけを選び、その候補だけgraph化します。progress barは `scan files`, `export files`, `write graphs` の3段階で表示します。`dstio` の内部unit上限を避けるため、file workerは既定で200ファイルごとに再起動します。

GNNを学習:

```bash
uv run talesd-gnn train \
  --graphs outputs/mc_graphs_*.h5 \
  -o outputs/talesd_gnn.pt \
  --epochs 100 \
  --batch-size 128
```

`export --shard-size` で分割した場合もlist fileは不要です。`--graphs outputs/graphs/talesd_mc_energy_flat.h5` のようにexport時のbase pathを渡すと、`talesd_mc_energy_flat_*.h5` を自動検出します。

学習時は `fit scalers`、epochごとのtrain/validation、最後のvalidation/test推論にprogress barを表示します。`--num-workers` はHDF5 graph読み込みとbatch構築に使うDataLoader worker数で、初期値の `-1` は小規模入力では0、大規模入力では最大4 workerを自動選択します。0にすると単一processへ戻ります。`--collate-backend auto` が既定で、小規模入力ではPython/NumPy collate、大規模入力やworker利用時は必須依存のpybind11 C++ collateを使います。

C++ collateの内部並列数は `--collate-threads` または環境変数 `TALESD_GNN_COLLATE_THREADS` で指定できます。既定値は実測最速の1です。0なら自動選択です。

核種別に再構成モデルを比較する場合は、DSTを読み直さず既存HDF5 graphから `rusdmc.parttype` 由来のラベルで絞ります。

```bash
uv run talesd-gnn train \
  --graphs /Users/ikomae/TALE/gnn/outputs/graphs/mass_12h_64perfile_6epoch.h5 \
  -o /Users/ikomae/TALE/gnn/outputs/talesd_gnn_reconstruction/models/reco_proton_only.pt \
  --particle-filter proton
```

proton/ironを続けて学習する比較用スクリプトもあります。

```bash
scripts/run_species_existing_graphs.sh
```

baselineのハイパーパラメーターを同時比較する場合は、既存graphだけを使うsweepを実行します。既定では4設定を2並列で走らせ、結果をCSVにまとめます。

```bash
scripts/run_hparam_sweep_existing_graphs.sh
```

新しい実行では、run単位でファイルをまとめる版を使います。

```bash
scripts/run_hparam_sweep_run_dir.sh
```

出力は `~/TALE/gnn/outputs/talesd_gnn_reconstruction/runs/<run_name>/` の下にまとまります。既存のflat layoutは移動せず、`scripts/index_legacy_outputs.py` でCSV indexを作れます。

数日規模のlarge runでは、DST読み込みと学習を分けます。まずlocal HDF5 graphを作成します。

```bash
MAX_EVENTS_PER_FILE=256 RUN_NAME=large256_export_$(date +%Y%m%d_%H%M%S) \
  scripts/export_large_graphs.sh
```

export完了時に `DST FILE READING COMPLETE` と表示されます。この表示以降は `/Volumes/TALE` のDSTを読まず、local HDF5 shardだけで学習できます。次にmassなし再構成モデルを学習します。

```bash
EXPORT_RUN_DIR=~/TALE/gnn/outputs/talesd_gnn_reconstruction/runs/<export_run_name> \
RUN_NAME=large256_baseline_$(date +%Y%m%d_%H%M%S) \
TRAIN_EPOCHS=12 TRAIN_WORKERS=4 scripts/train_large_existing_graphs.sh
```

large trainingの出力も `~/TALE/gnn/outputs/talesd_gnn_reconstruction/runs/<run_name>/` にまとまります。`summaries/<config>_precision_targets.txt` には、energy bias 5%、energy spread 25%、角度1度、core 50 mの最低ラインに対するPASS/FAILを保存します。

大規模HDF5 shardでは、DataLoaderはbatch単位でshuffleしつつbatch内indexを昇順に保ち、HDF5のランダムアクセスを抑えます。M2 Pro上の `talesd_mc_energy_flat_0000.h5` では `--batch-size 128`、auto worker、C++ collateで1 file / 1 epochが約161秒です。

学習後は `outputs/talesd_gnn.pt.metrics.json` に加えて、学習曲線とvalidation/test再構成診断のPDFを自動保存します。energy相対誤差のGaussianはSciPy `curve_fit` でfitします。

data DSTをグラフへ変換して再構成:

```bash
uv run talesd-gnn export /path/to/talesdcalibev_pass2_XXXXXX.dst \
  --kind data \
  -o outputs/data_graphs.h5

uv run talesd-gnn predict \
  --graphs outputs/data_graphs.h5 \
  --checkpoint outputs/talesd_gnn.pt \
  -o outputs/data_reconstruction.csv
```

GNN入力グラフを可視化:

```bash
uv run talesd-gnn visualize \
  --graphs outputs/mc_graphs_0000.h5 \
  --index 0 \
  --const-dst $TADIR/data/SD/talesdconst_pass2.dst \
  --annotate-lids \
  -o outputs/graph_0000.pdf
```

グラフ条件は `coincidence_analysis` の Ising filter に揃えています。ノードはpulse候補、edgeは `dr <= 1.5 km` かつ `|dt| <= 8 us` のpulse候補間で、Ising support weightも同じ式を使います。これらはコマンドラインで調整しません。

詳しい使い方と出力列は [docs/usage_ja.md](docs/usage_ja.md) を参照してください。
