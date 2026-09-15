# Quadro RTX 5000

| package | version |
|---------|:-------:|
| Driver version | 535.274.02 |
| CUDA version | 12.2 |

## 仮想環境作成

```bash
export UV_CACHE_DIR=./.venvs/.uv
uv venv ./.venvs/turing --python 3.10
source ./.venvs/turing/bin/activate
```

## 依存パッケージインストール

```bash
uv pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    "nemo_toolkit[asr]" \
    evaluate \
    jiwer \
    num2words \
    cmudict \
    g2p_en \
    duckdb \
    duckdb-cli \
    "setuptools<82" \
    "numba==0.60.0" \
    "llvmlite==0.43.0" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple
```
