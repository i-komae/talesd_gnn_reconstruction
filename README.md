# TALE-SD GNN Reconstruction

TALE-SD の不規則な検出器配置をグラフとして扱い、GNNでエネルギー、コア位置、到来方向を推定するための新規 `uv` プロジェクトです。

この初期版は次の流れを実装しています。

1. TALE-SD data DST (`talesdcalibev`) または MC DST (`rusdraw` + `rusdmc`) を読む
2. waveform から上下層 coincidence pulse を抽出する
3. ヒットSDをノード、同一SD内の全coincident pulseをノード付随のpulse集合にしたHDF5グラフへ変換する
4. MC truth付きグラフでGNNを学習する
5. 学習済みモデルで data / MC を再構成しCSVへ出力する

## セットアップ

```bash
cd /Users/ikomae/TALE/talesd_gnn_reconstruction
export DSTDIR=/path/to/dst2k-ta
uv sync
```

`dstio` は親ディレクトリの `../dstio` を editable path dependency として参照します。SD座標はテキスト表ではなく `talesdconst_pass2.dst` または `talesdcalibev` bank内のcalibration情報から読みます。

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

GNNを学習:

```bash
uv run talesd-gnn train \
  --graphs outputs/mc_graphs_*.h5 \
  -o outputs/talesd_gnn.pt \
  --epochs 100
```

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

グラフはSD単位です。複数pulseがあるSDをpulseごとの別ノードにはせず、`pulse_features` として保存し、学習時はpulse集合encoderで各SDノードへ集約します。

詳しい使い方と出力列は [docs/usage_ja.md](docs/usage_ja.md) を参照してください。
