[戻る](../README.md)
---

## Turing

[.venvs/turing.md](../.venvs/turing.md)を参照。

## Tensor

[.venvs/tensor.md](../.venvs/tensor.md)を参照。

## Shannon

[.venvs/shannon.md](../.venvs/shannon.md)を参照。

## Tsubame

[.venvs/tsubame.md](../.venvs/tsubame.md)を参照。

## テスト

リポジトリルートで実行する。`PYTHON` でインタプリタを差し替えられる。

普段の回帰（CPU、合成データ、temp DuckDB。`src/data` の integration を含む）:

```bash
source .venvs/<host>/bin/activate
PYTHON="$(pwd)/.venvs/<host>/bin/python" bash tests/run.sh
```

GPU / 実モデル（CUDA 必須。無ければ失敗する。NeMo の Parakeet と `mamba_ssm`）:

```bash
PYTHON="$(pwd)/.venvs/<host>/bin/python" bash tests/run_gpu.sh
```

認識文と close/far NLL まで見るときは、実発話を環境変数で渡す（テストコードにコーパスパスは書かない）:

```bash
export PARAKEET_TEST_AUDIO=...
export PARAKEET_TEST_TRANSCRIPT=...
export PARAKEET_TEST_UTTERANCE_ID=...
```

`nnAudio` や `pesq` が無い環境では、該当クラスだけ skip する。全テストは `bash tests/run.sh && bash tests/run_gpu.sh`。

## 開発ツール（Ruff / Pyright）

`src/` の formatter / linter / 型チェックと、`tests/` の formatter / linter。学習の実行には不要。設定はリポジトリルートの `pyproject.toml`。Pyright は `src/` のみ（テスト専用の型注釈を増やさない）。

```bash
source .venvs/<host>/bin/activate
uv pip install ruff pyright
```

```bash
ruff check src tests
ruff check --fix src tests
ruff format src tests
ruff format --check src tests
pyright
```
