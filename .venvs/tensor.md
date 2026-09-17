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
    pesq \
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

## mamba_ssm インストール

`/net/spring/work/seki/exp/master/speech-enhancement` と同じ `mamba-ssm` 2.2.4 の事前ビルド wheel を入れる。`nemo_toolkit[asr]` が入れる transformers 5.x では `GreedySearchDecoderOnlyOutput` が無く `mamba_ssm` の import が失敗するため、`.venvs/sitecustomize.py` を site-packages に置く。

```bash
uv pip install \
    triton==3.2.0 \
    packaging \
    ninja \
    wheel
uv pip install --no-deps \
    "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
cp .venvs/sitecustomize.py .venvs/tensor/lib/python3.10/site-packages/sitecustomize.py
```
