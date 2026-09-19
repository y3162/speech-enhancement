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

- `--config` を省くと各モデルの `configs/default.json` を使う。
- `--run_dir` に `config.json` がある場合は再開とみなし、その config と `latest.pt` から続ける（`--config` を併用するとエラー）。
- `train.batch_size` は GPU あたり。全体のバッチは `nproc_per_node` 倍になる。
- 出力: `run_dir/config.json`、`latest.pt`（毎 epoch）、`best.pt`（検証 PESQ 更新時）、`logs/`（TensorBoard）。
