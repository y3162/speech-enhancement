[戻る](../README.md)
---

## Turing

[.venvs/turing.md](../.venvs/turing.md)を参照。

## Tensor

[.venvs/tensor.md](../.venvs/tensor.md)を参照。

## Shannon

[.venvs/shannon.md](../.venvs/shannon.md)を参照。

## 開発ツール（Ruff / Pyright）

`src/`（`src/se`・`src/asr`・`src/utils`）の formatter / linter / 型チェック。学習の実行には不要。設定はリポジトリルートの `pyproject.toml`。

```bash
source .venvs/<host>/bin/activate
uv pip install ruff pyright
```

```bash
ruff check src
ruff check --fix src
ruff format src
ruff format --check src
pyright
```
