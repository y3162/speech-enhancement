[戻る](../docs/environment.md)
---

# NVIDIA H100-96GB

| package | version |
|---------|:-------:|
| Python | 3.10.20 |
| NVIDIA driver | 580.105.08 (構築時の実行ノードで確認) |
| CUDA runtime | 12.4 (PyTorch cu124 wheel) |
| CUDA toolkit / ptxas | 12.4.131 (`nvidia-cuda-nvcc-cu12`) |
| NeMo | 2.4.0 |
| Numba | 0.57.1 |
| llvmlite | 0.40.1 |
| CUDA Python | 11.8.5 |
| NumPy | 1.24.4 |
| transformers | 4.51.3 |
| mamba_ssm | 2.2.4 |
| nnAudio | 0.3.4 |

## 仮想環境作成

```bash
export UV_CACHE_DIR=./.venvs/.uv
uv venv ./.venvs/tsubame --python 3.10
source ./.venvs/tsubame/bin/activate
```

## 依存パッケージインストール

```bash
uv pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    "nemo_toolkit[asr]==2.4.0" \
    evaluate \
    jiwer \
    pesq \
    num2words \
    cmudict \
    g2p_en \
    duckdb \
    duckdb-cli \
    "setuptools<82" \
    "numba==0.57.1" \
    "llvmlite==0.40.1" \
    "numpy<1.25" \
    "cuda-python==11.8.5" \
    "nvidia-cuda-nvcc-cu12==12.4.131" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple
```

`cuda-python==11.8.5` は yanked の警告が出るが、NeMo/Numba の旧 TDT 経路を再現するため固定する。

CUDA 12.4 の toolkit は pip パッケージの標準配置に合わせるため、Numba が参照する `lib64` に PyTorch の CUDA runtime をリンクする。

```bash
CUDA_HOME="$PWD/.venvs/tsubame/lib/python3.10/site-packages/nvidia/cuda_nvcc"
CUDA_RUNTIME="$PWD/.venvs/tsubame/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
mkdir -p "$CUDA_HOME/lib64"
ln -sfn "$CUDA_RUNTIME/libcudart.so.12" "$CUDA_HOME/lib64/libcudart.so.12"
```

## mamba_ssm インストール

`/net/spring/work/seki/exp/master/speech-enhancement` と同じ `mamba-ssm` 2.2.4 の事前ビルド wheel を入れる。NeMo 2.4.0 が使用する transformers 4.51.3 では、現行環境で必要だった `sitecustomize.py` の互換パッチは不要。

```bash
uv pip install \
    triton==3.2.0 \
    packaging \
    ninja \
    wheel
uv pip install --no-deps \
    "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
```

## nnAudio インストール

SEMamba++ の判別器（`src/se/se_mamba_pp/discriminator.py` の `DiscriminatorCQT`）が CQT 変換に使う。

```bash
uv pip install nnAudio==0.3.4
```

## TSUBAME での実行環境

```bash
source .venvs/tsubame/bin/activate
export PYTHON="$PWD/.venvs/tsubame/bin/python"
export CUDA_HOME="$PWD/.venvs/tsubame/lib/python3.10/site-packages/nvidia/cuda_nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/nvvm/lib64:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export STRICT_NUMBA_COMPAT_CHECK=0
export NUMBA_CUDA_DEFAULT_PTX_CC=9.0
```

TDT loss を含む学習では、`src/asr/parakeet_tdt_0_6b_v2.py` が NeMo の `normalize_batch` にある autograd 非対応の in-place 更新を out-of-place 更新へ置き換える。
