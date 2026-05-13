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

SD座標はテキスト配置表からは読みません。dataの `talesdcalibev` では各hit sub bank内の `posX/posY/posZ` を使い、MCの `rusdraw` では `talesdconst_pass2.dst` の `talesdconst` bankから `lid -> posX/posY/posZ` を作って補います。

## セットアップ

```bash
cd /Users/ikomae/TALE/talesd_gnn_reconstruction
export DSTDIR=/path/to/dst2k-ta
uv sync
```

`dstio` は `../dstio` を参照します。`DSTDIR` は `dstio` のビルド時に必要です。

## 1. MC DSTをグラフへ変換

```bash
uv run talesd-gnn export /path/to/DATXXXXXX_tale.dst.gz \
  --kind mc \
  --const-dst $TADIR/data/SD/talesdconst_pass2.dst \
  --min-nodes 3 \
  --edge-radius-km 1.5 \
  --edge-k 6 \
  --workers 4 \
  --shard-size 100000 \
  -o outputs/mc_graphs.h5
```

複数ファイルを指定できます。

```bash
uv run talesd-gnn export /path/to/mc/*.dst.gz --kind mc -o outputs/mc_graphs.h5
```

主なオプション:

| オプション | 意味 |
| --- | --- |
| `--kind auto|mc|data` | 入力bankの種類。MC学習では `mc`、実データでは `data` を推奨。 |
| `--max-events N` | 動作確認用に最初のNイベントで止める。 |
| `--const-dst PATH` | MC `rusdraw` の `lid` からSD座標を引くための `talesdconst_pass2.dst`。未指定なら `TALESD_CONST_DST` または `$TADIR/data/SD/talesdconst_pass2.dst` を探す。 |
| `--min-nodes N` | 採用する最小ヒットSD数。初期値は3。 |
| `--edge-radius-km R` | この距離以内のSD間にエッジを張る。 |
| `--edge-k K` | 各SDに最低K近傍のエッジを張る。 |
| `--workers N` | waveform処理後のグラフ構築をN workerで並列化する。 |
| `--chunk-size N` | workerへ渡すイベント数。初期値は128。 |
| `--shard-size N` | Nグラフごとに `mc_graphs_0000.h5`, `mc_graphs_0001.h5` のように分割する。 |
| `--keep-non-mode0` | `trgMode != 0` も残す。通常は使わない。 |

出力HDF5は `events/00000000/...` のようなグループを持ち、各イベントに以下を保存します。

- `node_features`: SDごとの位置、代表時刻、集約信号量、pedestal情報
- `node_positions_km`: SD位置
- `edge_index`: `shape=(2, n_edges)` の有向エッジ
- `edge_features`: SD間距離、時刻差、信号比など
- `pulse_features`: 各SDで見つかった全coincident pulse候補。`node_index`, `arrival_usec_rel`, `dt_from_first_usec`, `log10_rho`, `sqrt_rho`, `pulse_order`, `is_first_pulse` を保存する。ここでの `rho` は候補単体の局所電荷ではなく、その候補のonset以降に成立した上下層coincidence pulseの積分電荷です。
- `target`: MC truthがある場合のみ保存

複数pulseの扱い:

- グラフのノードはpulseではなくSDです。TALE-SDの再構成で自然な単位は、配置と幾何を持つ検出器なので、同一SD内のpulseを別ノードに分裂させません。
- DST内で同じ `lid` が複数sub entryとして現れた場合は、`wfId` と `clock/maxClock` から128-bin segmentを先に長い波形へ連結し、その連結波形に対してpulse searchします。segmentごとに先にpulseを拾ってからmergeする処理ではありません。
- 各SDノードの `node_features` には、最初のcoincident pulseを時刻アンカーとして使い、`log10_total_rho`, `sqrt_total_rho`, `log10_max_rho`, `n_pulses`, `pulse_time_span_usec`, `n_wf_segments`, `wf_length_usec` も入れます。`total_rho` は全候補の単純和ではなく、先頭候補のonset以降のcoincidence積分電荷です。
- 同じSD内の全coincident pulseは `pulse_features` に保存します。学習時は `pulse_features[:, 0]` の `node_index` でSDノードに対応付け、残りのpulse特徴をMLPで埋め込んでmean/max poolし、SDノード特徴に結合します。
- つまり、最初のpulseだけで再構成する設計ではありません。最初のpulseは時刻基準であり、複数pulseの情報はpulse集合encoder経由でGNNに入ります。
- coincidence積分には固定の5 us gateを使いません。Ising filter側と同じく、候補onset以降に保存波形内で成立した上下層coincidence pulseだけを積分し、単層だけの後続pulseは含めません。

## 2. GNNを学習

```bash
uv run talesd-gnn train \
  --graphs outputs/mc_graphs_*.h5 \
  -o outputs/talesd_gnn.pt \
  --epochs 100 \
  --batch-size 32 \
  --hidden-dim 128 \
  --layers 4
```

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

validation/test metricには以下を出します。

- `rmse_log10_energy`
- `core_rmse_km`
- `angular_median_deg`
- `angular_68_deg`

大規模MCでは以下を推奨します。

- `export --shard-size` でHDF5を分割する
- `train --graphs outputs/mc_graphs_*.h5` のようにshardをまとめて指定する
- `train --sample-cache-size 0` でメモリ使用を最小化するか、十分なメモリがある場合は初期値のLRU cacheを使う
- feature scalerは全イベントをメモリへ連結せず、online統計で計算します。`node_features`, `edge_features`, `pulse_features`, `target` を別々に標準化します。

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
- MCの `rusdraw` は `talesdcalibev` に似た擬似bankへ変換します。検出器位置は `talesdconst_pass2.dst`、pedestal/MIP変換係数は `rusdraw` 内の値から作ります。
- dataの `talesdcalibev` はbank内の `posX/posY/posZ`、pedestal、MIP変換係数をそのまま使います。
- GNNは `torch-geometric` を使わず、PyTorchだけでpulse集合poolingとedge message passingを実装しています。依存関係を単純にし、TALE-SD固有のpulse/edge featureを直接扱うためです。
- この初期版は教師あり回帰です。実データで性能評価するには、まずMCで十分に学習し、既存のTALE-SD再構成結果やハイブリッド再構成との比較が必要です。
