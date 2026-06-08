# TALE-SD GNN Reconstruction

TALE-SD の不規則な検出器配置をグラフとして扱い、GNNでエネルギー、コア位置、到来方向を推定するための新規 `uv` プロジェクトです。

現行コードには、従来の homogeneous HDF5 graph と、新しい
`dstio.tale.graph` 由来の heterogeneous graph の2系統があります。
最終的な data / MC 再構成では、中間 HDF5 を必須にせず、学習済み
heterogeneous checkpoint で DST を直接読む経路を使います。

従来系の流れ:

1. TALE-SD data DST (`talesdcalibev`) または MC DST (`rusdraw` + `rusdmc`) を読む
2. waveform から上下層 coincidence pulse を抽出する
3. `coincidence_analysis` の Ising graph と同じ条件で、上下層coincident pulse候補をノードにしたHDF5グラフへ変換する
4. MC truth付きグラフでGNNを学習する
5. 学習済みモデルで data / MC を再構成しCSVへ出力する

新しい heterogeneous graph 系の流れ:

1. `dstio.tale.graph.iter_graphs` で DST から `GraphEvent` を作る
2. detector node と pulse node、3種類の edge relation を持つ HDF5 graph を学習用cacheとして保存する
3. `talesd-gnn train-hetero` で detector branch、pulse branch、detector-level waveform encoder、relation-specific attention を持つ heterogeneous model を学習する
4. 学習済みheterogeneous checkpointを使い、`talesd-gnn reconstruct-dst` で DST から直接再構成CSVを作る

## コード構成と実行フロー

このプロジェクトは、実行入口、CLI、実処理の3層で見ると追いやすくなります。

- `scripts/`: Slurm投入、resource指定、run名、graph path、環境変数を決める
- `src/talesd_gnn_reconstruction/cli.py`: `talesd-gnn export/train/predict/...` の入口
- `src/talesd_gnn_reconstruction/`: HDF5作成、dataset読み込み、モデル、loss、評価、診断図の実装

`talesd-gnn` は `pyproject.toml` の console script で、`talesd_gnn_reconstruction.cli:main` に対応します。

### HDF5グラフ作成

サーバー上の標準入口は `scripts/submit_server_graph_export.sh` です。DSTを読み、GNN用HDF5 graph shardを書き出します。

```text
scripts/submit_server_graph_export.sh
  -> .venv/bin/talesd-gnn export
    -> cli._cmd_export()
      -> dst_reader.py        DST bankを読む
      -> event_graph.py       eventをgraphへ変換する
      -> graph_io.py          HDF5へ書く
```

主な実装箇所:

- `src/talesd_gnn_reconstruction/cli.py`: `export` commandの引数処理
- `src/talesd_gnn_reconstruction/dst_reader.py`: DST入力の読み出し
- `src/talesd_gnn_reconstruction/event_graph.py`: node/edge/waveform/target特徴量の構築
- `src/talesd_gnn_reconstruction/graph_io.py`: HDF5 schemaと書き込み
- `scripts/summarize_graph_shards.py`: graph数、metadata、particle/source分布などの集計

新しい heterogeneous graph では、DST の読み出しと graph 定義は
`dstio.tale.graph` に寄せます。GNN repo 側は HDF5 cache、dataset、
model input、training、direct inference を担当します。

```text
.venv/bin/talesd-gnn export-hetero
  -> cli._cmd_export_hetero()
    -> dstio.tale.graph.iter_graphs()
    -> hetero_graph_io.py
```

`export-hetero` の HDF5 は学習用cacheです。最終的な一回通し再構成では、
HDF5を作らず `reconstruct-dst` が同じ graph schema と checkpoint scaler を使って
DSTから直接推論します。

既存の大規模HDF5から小規模HDF5を切り出す場合は、DSTを読み直しません。

```text
scripts/submit_server_small_graph_dataset.sh
  -> scripts/make_small_graph_dataset.py
    -> 既存HDF5 shardを読み、層化samplingして小規模HDF5を書く
```

### 学習

学習の標準入口はtaskごとに分かれています。

- 再構成 quality-only: `scripts/submit_server_waveform_full_training.sh`
- mass-only: `scripts/submit_server_mass_only_training.sh`
- reco+mass: `scripts/submit_server_reco_mass_training.sh`

`mass-only` と `reco+mass` のsubmitterも、最終的には `submit_server_waveform_full_training.sh` に渡ります。実際の学習本体は `scripts/train_large_existing_graphs.sh` から `talesd-gnn train` を呼びます。

```text
scripts/submit_server_*.sh
  -> scripts/train_large_existing_graphs.sh
    -> .venv/bin/talesd-gnn train
      -> cli._cmd_train()
        -> train.train_model()
          -> dataset.H5GraphDataset
          -> DataLoader / collate
          -> model.PhysicsTaleSdGNN
          -> loss計算
          -> checkpoint / metrics / diagnostics
```

主な実装箇所:

- `src/talesd_gnn_reconstruction/cli.py`: `train` commandの引数処理
- `src/talesd_gnn_reconstruction/train.py`: split、scaler、DataLoader、training loop、loss、best checkpoint、最終評価。再構成lossには、true energy binごとのlogE bias penaltyとproton/iron bias差 penaltyを指定できます。
- `src/talesd_gnn_reconstruction/dataset.py`: HDF5 graphの読み込み、metadata参照、collate
- `src/talesd_gnn_reconstruction/model.py`: GNN本体、waveform encoder、readout head
- `src/talesd_gnn_reconstruction/metrics.py`: 評価指標
- `src/talesd_gnn_reconstruction/diagnostics.py`: 学習曲線、energy依存性、quality cutなどの診断図

heterogeneous graph の標準入口は次です。

- 基本reco+mass: `scripts/submit_server_hetero_training.sh`
- reco+mass + quality-only比較: `scripts/submit_server_hetero_reco_mass_quality_training.sh`
- reco+mass + predicted-error-only比較: `scripts/submit_server_hetero_reco_mass_error_training.sh`

どちらの比較 wrapper も `LOSS_MODE=physics`、`MASS_CLASSIFICATION=1` を保ちます。
quality-only は `QUALITY_PREDICTION=1`, `ERROR_PREDICTION=0`、
predicted-error-only は `QUALITY_PREDICTION=0`, `ERROR_PREDICTION=1` です。
`physics-nll` へは自動では切り替えません。

```text
scripts/submit_server_hetero_*.sh
  -> scripts/train_hetero_existing_graphs.sh
    -> .venv/bin/talesd-gnn train-hetero
      -> cli._cmd_train_hetero()
        -> hetero_training.train_hetero_model()
          -> hetero_graph_io.H5HeteroGraphDataset
          -> hetero_data.sample_to_hetero_data()
          -> hetero_model.MinimalHeteroTaleSdGNN (`hetero_attention` architecture)
          -> loss / metrics / diagnostics
```

heterogeneous training の既定 architecture は `hetero_attention` です。
これは detector/pulse relation ごとの multi-head attention と detector/pulse type 別 attention readout を使います。
HGSampling は使いません。TALE では 1 event を 1 graph として丸ごと扱い、detector/pulse/waveform/Ising rejected pulse を sampling で落とさない方針です。

### テスト・評価・推論

学習時のvalidationと、学習後のtest評価は `train.train_model()` の中で実行されます。学習後はbest validation checkpointを読み戻して、validation/testに対するprediction、metrics、diagnosticsを作ります。

```text
train.train_model()
  -> train/validation/test split
  -> epochごとのtrain/validation loss
  -> best validation checkpoint保存
  -> best checkpoint読み戻し
  -> validation predict
  -> test predict
  -> metrics JSON
  -> diagnostics PDF/JSON
```

学習済みcheckpointを使って別途CSVを作る場合は `talesd-gnn predict` です。

```text
.venv/bin/talesd-gnn predict
  -> cli._cmd_predict()
    -> predict.predict_graphs()
```

heterogeneous checkpoint を使って DST を直接再構成する場合は
`talesd-gnn reconstruct-dst` です。これは HDF5 graph を入力にしません。

```text
.venv/bin/talesd-gnn reconstruct-dst
  -> cli._cmd_reconstruct_dst()
    -> hetero_predict.reconstruct_dst()
      -> dstio.tale.graph.iter_graphs()
      -> hetero_data.sample_to_hetero_data()
      -> hetero_model.MinimalHeteroTaleSdGNN
```

入力分布と特徴量重要度は、学習本体とは別の診断入口です。
通常のサーバーH5作成では、`scripts/submit_server_graph_export.sh` がH5作成後に入力分布とsplit分布をサーバー側で作り、H5ディレクトリ配下の `summaries/` に保存します。大きなH5をローカルへ同期せず、必要なPDF/JSONだけを確認します。
split診断の `sources` は学習で使う source group 数です。通常は `source_path` 単位ですが、`DAT??????_gea_trg_XXX.dst.gz` のように同じ CORSIKA shower を分割したDSTでは、`XXX` が異なっても共通の `DAT??????` を同じ source group として扱います。
通常の学習では、`scripts/train_large_existing_graphs.sh` が学習完了後にbest checkpointを使ってfeature importanceを自動実行し、checkpoint diagnostics配下へ保存します。

```text
.venv/bin/talesd-gnn input-distributions
  -> feature_analysis.py

.venv/bin/talesd-gnn feature-importance
  -> feature_analysis.py
```

### 目的別に見る場所

| 目的 | 主に見るファイル |
| --- | --- |
| Slurm投入条件、run名、graph path | `scripts/submit_*.sh` |
| CLI引数とcommand入口 | `src/talesd_gnn_reconstruction/cli.py` |
| DST読み込み | `src/talesd_gnn_reconstruction/dst_reader.py` |
| graph特徴量の定義 | `src/talesd_gnn_reconstruction/event_graph.py` |
| HDF5書き込み | `src/talesd_gnn_reconstruction/graph_io.py` |
| HDF5読み込み、collate | `src/talesd_gnn_reconstruction/dataset.py` |
| train/val/test split | `src/talesd_gnn_reconstruction/train.py` |
| loss計算 | `src/talesd_gnn_reconstruction/train.py` |
| モデル構造 | `src/talesd_gnn_reconstruction/model.py` |
| metrics、診断図 | `src/talesd_gnn_reconstruction/metrics.py`, `src/talesd_gnn_reconstruction/diagnostics.py` |
| 入力分布、特徴量重要度 | `src/talesd_gnn_reconstruction/feature_analysis.py` |
| hetero HDF5書き込み | `src/talesd_gnn_reconstruction/hetero_graph_io.py` |
| hetero graph変換 | `src/talesd_gnn_reconstruction/hetero_data.py` |
| hetero model/training | `src/talesd_gnn_reconstruction/hetero_model.py`, `src/talesd_gnn_reconstruction/hetero_training.py` |
| DST直接再構成 | `src/talesd_gnn_reconstruction/hetero_predict.py` |
| hetero特徴量重要度 | `src/talesd_gnn_reconstruction/hetero_feature_analysis.py` |

現在のモデル入力では、accepted-pulse node scalar は28列です。列名は `pulse_*`、`detector_*`、事象文脈量に分け、パルス自身の信号量、同一検出器内のaccepted pulse合計信号量、accepted pulse数、時間幅を区別します。`pulse_features` は対応nodeを示す `node_index` だけを保持し、pulse scalar encoderは使いません。旧HDF5の `log10_total_rho`、`sqrt_total_rho`、`local_detector_density_1p5km2` などを現行schemaへ黙って読み替えることはしません。特徴量の物理定義が変わったため、現行コードで学習するHDF5は再exportが必要です。

## APIドキュメント

CLI API、Python API、サーバー実行フローのSphinxドキュメントを `docs/api/` に置いています。日本語版と英語版を同じSphinxプロジェクト内で管理します。

```bash
uv sync --dev
.venv/bin/sphinx-build -b html docs/api docs/api/_build/html
```

ビルド後は `docs/api/_build/html/index.html` を開きます。

## セットアップ

```bash
cd /Users/ikomae/TALE/talesd_gnn_reconstruction
export DSTDIR=/path/to/dst2k-ta
uv sync
```

`dstio` は親ディレクトリの `../dstio` を editable path dependency として参照します。SD座標はテキスト表ではなく、MCでは `talesdconst_pass2.dst` の `talesdconst` bank、dataでは `talesdcalibev` bank内の位置情報から読みます。MCのpedestalとMIP換算係数は、Java解析の `RUDSTBankUtil.convertRuSDRaw2TASDCalibev2` と同じく外部校正を使います。

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
  --mc-calib-dir /path/to/tale_mc_calib \
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
  --mc-calib-dir /path/to/tale_mc_calib \
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

heterogeneous HDF5 を作成:

```bash
uv run talesd-gnn export-hetero \
  --input-dir /path/to/tale_mc_dst \
  --kind mc \
  --const-dst $TADIR/data/SD/talesdconst_pass2.dst \
  --mc-calib-dir /path/to/tale_mc_calib \
  --require-reference-core \
  --skip-errors \
  --skip-missing-mc-calibration \
  --shard-size 50000 \
  -o outputs/hetero_graphs/tale_hetero.h5
```

heterogeneous reco+mass を学習:

```bash
GRAPH_INPUT=/dicos_ui_home/ikomae/work/gnn/graphs/<hetero_graph_dir> \
PARTITION=v100-al9_long \
scripts/submit_server_hetero_training.sh
```

source group / geometry / time balance を取った heterogeneous HDF5 を3サイズで作る:

```bash
RUN_ID=hetero_balance_$(date +%Y%m%d_%H%M%S) \
scripts/submit_server_hetero_dataset_size_sweep.sh
```

これは既定で各 energy/particle bin あたり `50000`, `20000`, `10000` event の3種類を作ります。
selection summary と split distribution summary を確認してから、同じ `RUN_ID` で6本の比較学習を投げます。
既定 split は train/validation/test の source group を約 `45/10/45` にし、同じ `DAT??????` source group を split 間で共有しません。

```bash
RUN_ID=<exportで使ったRUN_ID> \
SUBMIT_EXPORTS=0 \
SUBMIT_TRAINING=1 \
scripts/submit_server_hetero_dataset_size_sweep.sh
```

次の比較では、同じ reco+mass 条件で quality-only と predicted-error-only を別々に投げます。

```bash
GRAPH_INPUT=/dicos_ui_home/ikomae/work/gnn/graphs/<hetero_graph_dir> \
PARTITION=v100-al9_long \
scripts/submit_server_hetero_reco_mass_quality_training.sh

GRAPH_INPUT=/dicos_ui_home/ikomae/work/gnn/graphs/<hetero_graph_dir> \
PARTITION=v100-al9_long \
scripts/submit_server_hetero_reco_mass_error_training.sh
```

heterogeneous checkpoint で DST を直接再構成:

```bash
uv run talesd-gnn reconstruct-dst \
  --input-dir /path/to/data_or_mc_dst \
  --kind auto \
  --checkpoint outputs/hetero_reco_mass.pt \
  --const-dst $TADIR/data/SD/talesdconst_pass2.dst \
  --mc-calib-dir /path/to/tale_mc_calib \
  --batch-size 256 \
  --skip-errors \
  -o outputs/reconstruction.csv
```

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
小規模HDF5を使ったJupyter/script共通のチューニング手順は [docs/small_dataset_jupyter_tuning.md](docs/small_dataset_jupyter_tuning.md) を参照してください。
