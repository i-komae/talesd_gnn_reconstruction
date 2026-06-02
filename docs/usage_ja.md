# TALE-SD GNN Reconstruction 使用方法

## 目的

このプロジェクトは、TALE-SDのいびつな検出器配置を2次元画像ではなくグラフとして表現し、GNNで以下を直接回帰するための土台です。

- `log10_energy_eV`
- `core_x_km`, `core_y_km`, `core_z_km`
- 到来方向ベクトル `dir_x`, `dir_y`, `dir_z`

推論CSVでは方向ベクトルを `zenith_deg`, `azimuth_deg` に戻して出力します。

## 入力DST

対応している入力は2種類です。

| 種類 | bank | 用途 |
| --- | --- | --- |
| MC | `rusdraw`, `rusdmc` | 学習用。`rusdmc` から energy/core/direction のtruthを作る。 |
| data | `talesdcalibev` | 推論用。truthなしでグラフ化する。 |

SD座標はテキスト配置表からは読みません。dataの `talesdcalibev` では各hit sub bank内の `posX/posY/posZ` を使い、MCの `rusdraw` では `talesdconst_pass2.dst` の `talesdconst` bankから `lid -> posX/posY/posZ` を作って補います。MCのpedestalとMIP換算係数は `rusdraw` の値を主入力にせず、Java解析の `RUDSTBankUtil.convertRuSDRaw2TASDCalibev2` と同じ考え方で外部校正から取得します。

## セットアップ

```bash
cd /Users/ikomae/TALE/talesd_gnn_reconstruction
export DSTDIR=/path/to/dst2k-ta
uv sync
```

`dstio` は `../dstio` を参照します。`DSTDIR` は `dstio` のビルド時に必要です。

C++/pybind11拡張を明示的に再ビルドする場合:

```bash
./build_extensions.sh
```

## 1. MC DSTをグラフへ変換

```bash
uv run talesd-gnn export /path/to/DATXXXXXX_tale.dst.gz \
  --kind mc \
  --const-dst $TADIR/data/SD/talesdconst_pass2.dst \
  --mc-calib-dir /path/to/tale_mc_calib \
  --workers 4 \
  --shard-size 100000 \
  -o outputs/mc_graphs.h5
```

複数ファイルを指定できます。

```bash
uv run talesd-gnn export /path/to/mc/*.dst.gz --kind mc -o outputs/mc_graphs.h5
```

多数のMCを読む場合は、shell globで全ファイルを展開せず、ディレクトリ指定または入力リストを使います。

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
  --seed 12345 \
  --workers 8 \
  --worker-max-files 200 \
  --shard-size 50000 \
  --skip-errors \
  -o outputs/graphs/talesd_mc_energy_flat.h5
```

`--energy-sample-per-bin` を指定すると、まず軽いmetadata scanで `log10(E/eV)` binごとの候補event indexを抽出し、その候補だけgraph化します。各ファイル内のイベントは時刻順に近いため、学習用のランダム抽出では `--max-events` で先頭から切らず、この sampling を使います。sampling keyはイベントIDとseedから決めるので、ファイル単位workerの完了順には依存しません。`zenith` は再重み付けせず、元のMC分布のまま残します。

`--energy-oversample-factor` は、metadata scanでgraph化前に余分に残す倍率です。例えば `--energy-sample-per-bin 50000 --energy-oversample-factor 2` なら、各energy binから最大100000候補をgraph化し、graph化後に最終50000件へ落とします。graph化に失敗する候補があるため、1より大きい値を使います。

`--max-events` を指定しない通常の大量MC exportでは、`--workers` はファイル単位の並列数です。各workerがDST読み込み、waveform処理、graph構築まで担当し、親processは完了したファイルのgraphを受け取ってsamplingとHDF5書き込みを行います。progress barは `scan files`, `export files`, `write graphs` を表示します。DSTライブラリには1 processあたりの内部unit数に上限があるため、file workerは既定で200ファイルごとに再起動します。

主なオプション:

| オプション | 意味 |
| --- | --- |
| `--kind auto|mc|data` | 入力bankの種類。MC学習では `mc`、実データでは `data` を推奨。 |
| `--max-events N` | 動作確認用に最初のNイベントで止める。 |
| `--input-dir PATH` | ディレクトリ内の `*.dst.gz` を再帰的に入力する。多数MCではこれを推奨。 |
| `--input-list PATH` | 入力DSTパスを1行1ファイルで書いたリスト。 |
| `--const-dst PATH` | MC `rusdraw` の `lid` からSD座標を引くための `talesdconst_pass2.dst`。未指定なら `TALESD_CONST_DST` または `$TADIR/data/SD/talesdconst_pass2.dst` を探す。 |
| `--mc-calib-dir PATH` | MC `rusdraw` を `talesdcalibev` 相当へ変換するための校正ディレクトリ。Java解析と同じ `talesdcalib_pass2_*.dst(.gz)` を読む。MCでは必須。 |
| `--energy-sample-per-bin N` | `log10(E/eV)` binごとに最大N graphをランダム抽出する。 |
| `--energy-bin-width W` | energy samplingのbin幅。初期値は0.1。 |
| `--energy-oversample-factor F` | graph化前のmetadata scanで各binから余分に残す倍率。初期値は2。 |
| `--seed N` | reservoir samplingの乱数seed。 |
| `--workers N` | 大量MCではDST読み込みからgraph構築までをファイル単位でN並列にする。 |
| `--worker-max-files N` | file workerをNファイルごとに再起動する。初期値は200。DST unit上限対策なので、通常は0にしない。 |
| `--chunk-size N` | `--max-events` 指定時だけ、workerへ渡すイベント数。初期値は128。 |
| `--shard-size N` | Nグラフごとに `mc_graphs_0000.h5`, `mc_graphs_0001.h5` のように分割する。 |
| `--keep-non-mode0` | `trgMode != 0` も残す。通常は使わない。 |
| `--skip-errors` | 読めないDSTを警告してスキップする。大量MCでは推奨。 |

graph条件は `coincidence_analysis` の Ising filter と同じで、コマンドラインから変更しません。

- ノードはSDではなく、上下層 coincidence pulse 候補です。
- ノード採用条件は `rho >= 0.3` かつ `fadc_peak < 4095` です。
- イベント採用条件は、有効pulse nodeが4個以上、かつ有効検出器が4台以上です。
- edgeは同一検出器内には張らず、`dr <= 1.5 km` かつ `|dt| <= 8 us` のpulse候補間だけに張ります。
- edge weightは `coincidence_analysis/cpp/src/reconstruction_ising.cpp` と同じ Ising support 式を使い、graph-degree補正も同じ `p_J=0.70` で行います。

出力HDF5は `events/00000000/...` のようなグループを持ち、各イベントに以下を保存します。

- `node_features`: pulse候補ごとの位置、到来時刻、coincidence積分信号量、対応する検出器のpulse情報、pedestal情報
- `node_positions_km`: pulse候補の属するSD位置
- `edge_index`: `shape=(2, n_edges)` の有向エッジ
- `edge_features`: pulse候補間の距離、時刻差、信号比、Ising edge weightなど
- `pulse_features`: 対応nodeを示す `node_index` だけを保持します。現在のgraphではpulse候補そのものがnodeなので、追加のpulse scalar encoderは使いません。
- `target`: MC truthがある場合のみ保存

複数pulseの扱い:

- グラフのノードは `coincidence_analysis` の Ising graph と同じくpulse候補です。同一SD内に複数の上下層 coincidence 候補がある場合、それぞれを別nodeとして保存します。
- DST内で同じ `lid` が複数sub entryとして現れた場合は、`wfId` と `clock/maxClock` から128-bin segmentを先に長い波形へ連結し、その連結波形に対してpulse searchします。segmentごとに先にpulseを拾ってからmergeする処理ではありません。
- 各pulse nodeの `rho` は候補単体の局所電荷ではなく、その候補のonset以降に成立した上下層coincidence pulseの積分電荷です。
- detector-levelの代表時刻へ潰さず、複数pulse候補をgraph nodeとしてGNNへ渡します。
- coincidence積分には固定の5 us gateを使いません。Ising filter側と同じく、候補onset以降に保存波形内で成立した上下層coincidence pulseだけを積分し、単層だけの後続pulseは含めません。

## 2. GNNを学習

```bash
uv run talesd-gnn train \
  --graphs outputs/mc_graphs_*.h5 \
  -o outputs/talesd_gnn.pt \
  --epochs 100 \
  --batch-size 128 \
  --hidden-dim 128 \
  --layers 4
```

核種別の再構成性能を比較する場合は、既存HDF5 graphを `rusdmc.parttype` 由来のラベルで絞ります。`proton` は `parttype=14`、`iron` は `parttype=5626` です。DSTは読み直しません。

```bash
uv run talesd-gnn train \
  --graphs /Users/ikomae/TALE/gnn/outputs/graphs/mass_12h_64perfile_6epoch.h5 \
  -o /Users/ikomae/TALE/gnn/outputs/talesd_gnn_reconstruction/models/reco_proton_only.pt \
  --particle-filter proton \
  --split-mode source-stratified
```

proton/ironの両方を同じ条件で順番に学習する場合:

```bash
scripts/run_species_existing_graphs.sh
```

baselineのハイパーパラメーターを同時比較する場合:

```bash
scripts/run_hparam_sweep_existing_graphs.sh
```

既定では `hidden_dim/layers/lr/dropout/weight_decay/LR scheduler` の4設定を2並列で走らせ、`~/TALE/gnn/outputs/talesd_gnn_reconstruction/sweeps/` にsummary CSVを書きます。DSTは読み直しません。

次回以降は、1実行を1ディレクトリに閉じ込める版を推奨します。

```bash
scripts/run_hparam_sweep_run_dir.sh
```

出力構成:

```text
~/TALE/gnn/outputs/talesd_gnn_reconstruction/runs/<run_name>/
  README.txt
  config/run.env
  config/configs.txt
  summaries/metrics_summary.csv
  logs/<config>.log
  checkpoints/<config>.pt
  checkpoints/<config>.pt.metrics.json
  checkpoints/<config>.pt.diagnostics/
```

既存の `models/`, `logs/`, `sweeps/` に散らばった結果は移動せず、以下でCSV indexを作れます。

```bash
scripts/index_legacy_outputs.py
```

## 数日規模のlarge run

現在の `64 events/file` graphで精度が頭打ちの場合は、まずDSTをlocal HDF5 graphへ増量exportし、その後はDSTを読まずに学習します。

```bash
MAX_EVENTS_PER_FILE=256 RUN_NAME=large256_export_$(date +%Y%m%d_%H%M%S) \
  scripts/export_large_graphs.sh
```

このscriptはproton/ironの6入力ディレクトリを読み、`~/TALE/gnn/outputs/talesd_gnn_reconstruction/runs/<run_name>/graphs/` にshardを作成します。export完了時に以下のmarkerを強調表示します。

```text
DST FILE READING COMPLETE
```

この行以降は `/Volumes/TALE` のDST入力を使いません。ネットワークを切るならこの後です。作成されたgraphは `config/graph_input.txt` に保存されます。

massなしのlarge trainingは以下です。

```bash
EXPORT_RUN_DIR=~/TALE/gnn/outputs/talesd_gnn_reconstruction/runs/<export_run_name> \
RUN_NAME=large256_baseline_$(date +%Y%m%d_%H%M%S) \
TRAIN_EPOCHS=12 TRAIN_WORKERS=4 scripts/train_large_existing_graphs.sh
```

既定では `source-stratified` split、`test_fraction=0.20`、C++ collate、mass分類headなしで学習します。出力は以下にまとまります。

```text
~/TALE/gnn/outputs/talesd_gnn_reconstruction/runs/<train_run_name>/
  README.txt
  config/train.env
  logs/<config>.log
  checkpoints/<config>.pt
  checkpoints/<config>.pt.metrics.json
  checkpoints/<config>.pt.diagnostics/
  summaries/metrics_summary.csv
  summaries/<config>_precision_targets.txt
```

`summaries/<config>_precision_targets.txt` は、energy bias 5%、energy Gaussian sigma 20%（15%は参考基準）、opening angle 1度、core xy 50 mを最低ラインとしてPASS/FAILを出します。

標準splitは `train:validation:test = 8:1:1` です。`validation` はbest checkpointの選択に使い、`test` は最後に一度だけ評価します。比率を変える場合は以下を指定します。

```bash
uv run talesd-gnn train \
  --graphs outputs/mc_graphs_*.h5 \
  -o outputs/talesd_gnn.pt \
  --val-fraction 0.1 \
  --test-fraction 0.1
```

出力:

- `outputs/talesd_gnn.pt`: モデル、feature scaler、target scalerを含むcheckpoint
- `outputs/talesd_gnn.pt.metrics.json`: epochごとのloss、validation metric、test metric、split件数
- `outputs/talesd_gnn.pt.learning_curve.pdf`: 学習曲線
- `outputs/talesd_gnn.pt.validation_diagnostics.pdf`: validation setの再構成診断図
- `outputs/talesd_gnn.pt.test_diagnostics.pdf`: test setの再構成診断図
- `outputs/talesd_gnn.pt.diagnostics.json`: 診断図に使った68%値、energy binごとの `mu`, `sigma` など

validation/test metricには以下を出します。

- `rmse_log10_energy`
- `core_rmse_km`
- `angular_median_deg`
- `angular_68_deg`

診断PDFには以下を自動保存します。

- `train_loss` と `validation_loss` の学習曲線
- opening angleのヒストグラムと68% containment
- core `x`, `y` 残差のヒストグラム
- core位置ずれ `sqrt(dx^2 + dy^2)` のヒストグラムと68% containment
- energy相対誤差 `(E_rec - E_true) / E_true` のヒストグラムとSciPy `curve_fit` によるGaussian fit
- true `log10(E/eV)` binごとのenergy相対誤差ヒストグラム、SciPy fitのGaussian `mu`, `sigma`
- energy binごとのcentral 68%幅と `mu +/- sigma` のenergy依存性

大規模MCでは以下を推奨します。

- `export --shard-size` でHDF5を分割する
- `train --graphs outputs/mc_graphs.h5` のようにexport時のbase pathを渡す。shard出力なら `outputs/mc_graphs_*.h5` を自動検出するためlist fileは不要です。
- `--batch-size 128` から始める。この環境の `talesd_mc_energy_flat_0000.h5` では32/256より速い。
- `--num-workers` は初期値の `-1` に任せる。小規模入力では0、大規模入力では最大4 workerを自動選択します。
- `--collate-backend auto` は大規模入力でC++/pybind11 collateを使う。C++ collateの内部thread数は `--collate-threads` または環境変数 `TALESD_GNN_COLLATE_THREADS` で指定できます。既定値は実測最速の1です。
- `--sample-cache-size 0` が既定です。HDF5をほぼ一方向に読むため、通常の1 epoch学習ではcacheしません。
- DataLoaderはbatch単位でshuffleしつつbatch内indexを昇順に保ち、HDF5のランダムアクセスを抑えます。
- feature scalerは全イベントをメモリへ連結せず、online統計で計算します。`node_features`, `edge_features`, `pulse_features`, `target` を別々に標準化します。

学習中は `fit scalers`、epochごとのtrain/validation、最後のvalidation/test推論にprogress barを表示します。不要な場合は `--no-progress` を指定します。

## 3. data DSTを推論

まず data DST を同じ形式のグラフへ変換します。

```bash
uv run talesd-gnn export /path/to/talesdcalibev_pass2_XXXXXX.dst \
  --kind data \
  -o outputs/data_graphs.h5
```

次に学習済みモデルで推論します。

```bash
uv run talesd-gnn predict \
  --graphs outputs/data_graphs.h5 \
  --checkpoint outputs/talesd_gnn.pt \
  -o outputs/data_reconstruction.csv
```

推論CSVの主な列:

| 列 | 意味 |
| --- | --- |
| `event_id` | `date_time_usec_index` 形式のイベントID |
| `log10_energy_eV`, `energy_eV` | 推定エネルギー |
| `core_x_km`, `core_y_km`, `core_z_km` | 推定コア位置 |
| `zenith_deg`, `azimuth_deg` | 推定到来方向 |
| `n_nodes`, `n_edges` | GNN入力に使ったグラフサイズ |

MCグラフを推論した場合はtruth列と誤差列も出力します。

## 4. グラフを可視化

HDF5に保存したGNN入力グラフをPDFとして描画できます。既存のevent displayの描画作法を参考に、白抜き四角で全TALE-SD配置、薄い線でGNN edge、塗りつぶし丸で各SDノード、白抜きリングで同じSD内の追加pulseを示します。色は相対到来時刻、丸の大きさは信号量を表します。MC truthがある場合はtrue coreと方向投影も描きます。

```bash
uv run talesd-gnn visualize \
  --graphs outputs/mc_graphs_0000.h5 \
  --index 0 \
  --const-dst $TADIR/data/SD/talesdconst_pass2.dst \
  --annotate-lids \
  -o outputs/graph_0000.pdf
```

複数shardから `event_id` で選ぶ場合:

```bash
uv run talesd-gnn visualize \
  --graphs outputs/mc_graphs_*.h5 \
  --event-id MC123456_240315_123014_421927_000000 \
  -o outputs/selected_graph.pdf
```

連続イベントをディレクトリに出す場合:

```bash
uv run talesd-gnn visualize \
  --graphs outputs/mc_graphs_*.h5 \
  --index 0 \
  --count 20 \
  -o outputs/graph_preview
```

主なオプション:

| オプション | 意味 |
| --- | --- |
| `--index N` | HDF5内のN番目のグラフを描画する。 |
| `--event-id ID` | `event_id` でグラフを選ぶ。 |
| `--const-dst PATH` | 背景の全TALE-SD配置を描くための `talesdconst_pass2.dst`。未指定時は環境変数か `$TADIR` から探す。 |
| `--count N` | 連続Nイベントを描画する。 |
| `--no-edges` | GNN edgeを描画しない。 |
| `--annotate-lids` | SD lidを表示する。 |
| `--max-edges N` | 描画するedge数の上限。 |

## 実装上の注意

- waveform処理は `JAIDATALESDAnalyzer.java` と既存 `coincidence_analysis/talecoin/reconstruction.py` を参考に、`SDPreAnalysis` 相当のpulse searchと上下層coincidenceをPythonで実装しています。
- MCの `rusdraw` は `talesdcalibev` に似た擬似bankへ変換します。検出器位置は `talesdconst_pass2.dst` から補い、pedestal/MIP換算係数は `--mc-calib-dir` の外部校正から取得します。`rusdraw` 内の値は校正recordがない場合の旧式の補助経路であり、通常のMC exportでは使いません。
- dataの `talesdcalibev` はbank内の `posX/posY/posZ`、pedestal、MIP変換係数をそのまま使います。
- GNNは `torch-geometric` を使わず、PyTorchだけでpulse集合poolingとedge message passingを実装しています。依存関係を単純にし、TALE-SD固有のpulse/edge featureを直接扱うためです。
- この初期版は教師あり回帰です。実データで性能評価するには、まずMCで十分に学習し、既存のTALE-SD再構成結果やハイブリッド再構成との比較が必要です。
