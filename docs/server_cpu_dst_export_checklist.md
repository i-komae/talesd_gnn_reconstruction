# Server CPU DST Export Checklist

DST の読み込みと HDF5 グラフ作成をサーバー上の CPU ジョブで行う前に確認する項目をまとめる。
ここでの目的は、GPU 学習ジョブを待たせず、DST 読み込みとグラフ作成を CPU partition で先に完了させることである。

## 入力 DST

標準入力ディレクトリは次の 6 ディレクトリである。

```text
/dicos_ui_home/ikomae/work/taleMC/proton/sel/tale_proton5.5yr_16-16.9_v260313
/dicos_ui_home/ikomae/work/taleMC/proton/sel/tale_proton5.5yr_17-17.9_v260313
/dicos_ui_home/ikomae/work/taleMC/proton/sel/tale_proton5.5yr_18-18.9_v260313
/dicos_ui_home/ikomae/work/taleMC/iron/sel/tale_iron4yr_16-16.9_v260316
/dicos_ui_home/ikomae/work/taleMC/iron/sel/tale_iron4yr_17-17.9_v260316
/dicos_ui_home/ikomae/work/taleMC/iron/sel/tale_iron4yr_18-18.9_v260316
```

サーバー上で、各ディレクトリが見えること、`*.dst.gz` が入っていること、読み取り権限があることを確認する。

```bash
for d in \
  /dicos_ui_home/ikomae/work/taleMC/proton/sel/tale_proton5.5yr_16-16.9_v260313 \
  /dicos_ui_home/ikomae/work/taleMC/proton/sel/tale_proton5.5yr_17-17.9_v260313 \
  /dicos_ui_home/ikomae/work/taleMC/proton/sel/tale_proton5.5yr_18-18.9_v260313 \
  /dicos_ui_home/ikomae/work/taleMC/iron/sel/tale_iron4yr_16-16.9_v260316 \
  /dicos_ui_home/ikomae/work/taleMC/iron/sel/tale_iron4yr_17-17.9_v260316 \
  /dicos_ui_home/ikomae/work/taleMC/iron/sel/tale_iron4yr_18-18.9_v260316
do
  printf "%s " "$d"
  find "$d" -type f -name '*.dst.gz' | wc -l
done
```

## const DST

MC のグラフ作成では TALE-SD の検出器配置を読むため、`talesdconst_pass2.dst` が必要である。
次のいずれかを設定する。

```bash
export CONST_DST=/path/to/talesdconst_pass2.dst
# or
export TALESD_CONST_DST=/path/to/talesdconst_pass2.dst
# or
export TADIR=/path/to/TALE
```

`CONST_DST` を使う場合は、ジョブ投入前に次を確認する。

```bash
test -r "$CONST_DST" && ls -lh "$CONST_DST"
```

## Slurm リソース

CPU export の submit script は `scripts/submit_server_graph_export.sh` である。
`PARTITION` は標準値を持たせず、投入時に明示する。
これは、その時点の空き CPU、空きメモリ、待機列、利用可能 account を確認せずに投入先を固定しないためである。
標準的な初期設定は次の通り。

```text
partition        explicit PARTITION required
cpus-per-task    64
memory           256G
time-limit       2-00:00:00
export workers   64
summary workers  64
```

投入前に、partition の時間制限、空き CPU、空きメモリ、自分の account から投入できるかを確認する。

```bash
show_slurm_summary -c
sinfo -p edr1-al9_large,edr2-al9_large -o "%P %a %l %D %t %c %m %N"
sacctmgr show assoc user=$USER format=User,Account,Partition,QOS
squeue -u "$USER"
```

CPU を増やす場合は、`CPUS_PER_TASK` と `EXPORT_WORKERS` を同じ値にする。
OpenMP 系の内部並列化で worker 数が実質的に増えすぎないように、submit script は `OMP_NUM_THREADS=1` を設定する。

## 容量と出力先

標準出力先は次の 2 か所に分かれる。

```text
/dicos_ui_home/ikomae/work/gnn/graphs/<run_name>/
/dicos_ui_home/ikomae/work/gnn/outputs/talesd_gnn_reconstruction/runs/<run_name>/
```

前者に HDF5 graph shard、後者に Slurm script、ログ、設定、summary を置く。
投入前に空き容量を確認する。

```bash
df -h /dicos_ui_home/ikomae/work/gnn
```

古い試験では 50,000 graph で約 1.4 GiB だった。
波形付きで event 数を大きく増やす場合、数百 GiB 以上になる可能性がある。

## 実行環境

submit script はジョブ内で次の module を読む。

```bash
module purge
module load gcc/13.1.0 cmake/3.28 hdf5/2.0.0 mkl/latest tbb/latest
```

`RUN_BUILD=1` が標準であり、ジョブ内で `uv sync` と `./build_extensions.sh` を実行する。
既にサーバー側で build 済みで、同じ commit のまま再投入する場合だけ `RUN_BUILD=0` にしてよい。

## smoke test

最初は 1 ファイルあたりの読み込み数を小さくして、DST 読み込み、const DST、HDF5 出力、summary 作成が通ることを確認する。

```bash
CONST_DST=/path/to/talesdconst_pass2.dst \
PARTITION=edr1-al9_large \
RUN_NAME=server_graph_export_smoke_$(date +%Y%m%d_%H%M%S) \
MAX_EVENTS_PER_FILE=2 \
ENERGY_SAMPLE_PER_BIN=0 \
CPUS_PER_TASK=8 \
EXPORT_WORKERS=8 \
SUMMARY_WORKERS=8 \
MEM=32G \
TIME_LIMIT=01:00:00 \
scripts/submit_server_graph_export.sh
```

## full export

smoke test 後に、標準の energy-flat sampling で本番 export を投入する。

```bash
CONST_DST=/path/to/talesdconst_pass2.dst \
PARTITION=edr1-al9_large \
RUN_NAME=server_graph_export_energyflat200k_$(date +%Y%m%d_%H%M%S) \
scripts/submit_server_graph_export.sh
```

完了後、ログに `DST FILE READING COMPLETE` が出ていることを確認する。
その後は `config/graph_input.txt` に書かれた HDF5 graph shard を training script の `GRAPH_INPUT` に渡す。

## export 後の確認

`summaries/graph_summary.json` で次を確認する。

- graph 数
- proton と iron の数
- unknown particle が 0 であること
- source path 数
- shard 数と各 shard の graph 数

さらに、学習前に `talesd-gnn train` の source-stratified split summary で、train、validation、test の graph 数が意図した比率に近いことを確認する。
