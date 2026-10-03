[戻る](../README.md)
---

# 音声強調モデルの学習

学習入口は `src/se/mp_senet/train.py` / `src/se/se_mamba/train.py` / `src/se/se_mamba_pp/train.py`。GPU 1 枚でも `torchrun` で起動する。

```bash
source .venvs/<host>/bin/activate
export PYTHONPATH="$(pwd)"
export SPEECH_DB_ROOT_DIR=...      # metadata.duckdb のあるディレクトリ（.env.example 参照）
export SPEECH_CORPORA_ROOT_DIR=... # LibriSpeech などのコーパスルート。DB の audio_path は

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m src.se.se_mamba_pp.train \
    --run_dir data/checkpoints/se_mamba_pp/$(date +%Y%m%d_%H%M%S) \
    --config src/se/se_mamba_pp/configs/default.json
```

- `--config` を省くと各モデルの `configs/default.json` を使う。SEMamba++ で ASR 特徴量を入れないときは `--config src/se/se_mamba_pp/configs/no_asr.json`（`asr_guidance_dim` が 0）。
- `--run_dir` に `config.json` がある場合は再開とみなし、その config と `latest.pt` から続ける（`--config` を併用するとエラー）。
- `train.batch_size` は GPU あたり。全体のバッチは `nproc_per_node` 倍になる。
- 出力: `run_dir/config.json`、`latest.pt`（毎 epoch）、`best.pt`（検証 PESQ 更新時）、`logs/`（TensorBoard）。

## error-aware SEMamba++

`src/se/error_aware_se_mamba_pp` は SEMamba++ の複製に、凍結 ASR の clean / noisy 認識差（置換・脱落）でスペクトル損失と mel 損失を時間重みする経路を足したもの。`src.se.se_mamba_pp` は import しない。augmentation は `data.variants_per_utterance`（K）で決まり、epoch `e` は variant `e % K` を再生する。検証は variant 0 の1回。

`configs/default.json` は重み付けあり（`error_aware.enabled=true`、`alpha=1.0`）。`configs/baseline.json` は同じ K のスケジュールで重み付けなし。キャッシュのパスはホスト依存なので config に入れず、`--alignment-cache` で渡す。enabled が true のときだけ必須。

キャッシュは学習前に作る。シャードに分けてから結合する。

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.se.error_aware_se_mamba_pp.build_cache \
    --alignment-cache data/error_aware_se_mamba_pp/alignment.shard0.sqlite \
    --config src/se/error_aware_se_mamba_pp/configs/default.json \
    --num-shards 2 --shard 0

python -m src.se.error_aware_se_mamba_pp.build_cache \
    --alignment-cache data/error_aware_se_mamba_pp/alignment.sqlite \
    --merge-shards 2
```

学習:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m src.se.error_aware_se_mamba_pp.train \
    --run_dir data/checkpoints/error_aware_se_mamba_pp/baseline \
    --config src/se/error_aware_se_mamba_pp/configs/baseline.json

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m src.se.error_aware_se_mamba_pp.train \
    --run_dir data/checkpoints/error_aware_se_mamba_pp/proposed \
    --config src/se/error_aware_se_mamba_pp/configs/default.json \
    --alignment-cache data/error_aware_se_mamba_pp/alignment.sqlite
```

`--max-steps N` は optimizer step が N に達した時点で検証と保存をせずに止める。`run_dir/config.json` がある再開では `--config` と `--max-steps` はエラー。
