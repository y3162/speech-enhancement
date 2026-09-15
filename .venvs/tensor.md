[戻る](../docs/environment.md)
---

# NVIDIA A100-PCIE-40GB / NVIDIA A100 80GB PCIe

| package | version |
|---------|:-------:|
| Driver version | 535.183.01 |
| CUDA version | 12.2 |
| CUDA toolkit (nvcc) | 11.0.221 |

## 仮想環境作成

```bash
export UV_CACHE_DIR=./.venvs/.uv
uv venv ./.venvs/tensor --python 3.10
source ./.venvs/tensor/bin/activate
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
    "numpy<1.25" \
    "numba==0.57.1" \
    "llvmlite==0.40.1" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple
```
