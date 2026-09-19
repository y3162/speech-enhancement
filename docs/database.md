[戻る](../README.md)
---

## シンボリックリンクの作成

全コーパスを `./corpora` にシンボリックリンクする。

[`src/data/db.py`](../src/data/db.py) にあるように、全コーパスは同一親ディレクトリに配置されていることを仮定している。

```bash
mkdir -p ./corpora
mkdir -p ./data/db

ln -sfn /path/to/LibriSpeech ./corpora/LibriSpeech
ln -sfn /path/to/LibriTTS ./corpora/LibriTTS
ln -sfn /path/to/VCTK ./corpora/VCTK
ln -sfn /path/to/DEMAND ./corpora/DEMAND
```

## データベースの作成


```bash
mkdir -p ./data/db
```

## 環境変数の設定

```bash
export SPEECH_CORPORA_ROOT_DIR=./corpora
export SPEECH_DB_ROOT_DIR=./data/db
```

## メタデータのインポート

同じ DuckDB ファイルへ続けて投入できる。テーブルを作り直すときは `--replace` を付ける。

```bash
python -m src.data.utterances --corpus librispeech
python -m src.data.utterances --corpus libritts
python -m src.data.utterances --corpus vctk
python -m src.data.noises --corpus demand
python -m src.data.noise_configs
python -m src.asr.timestamp_cache --splits train-clean-100 train-clean-360 dev-clean test-clean
python -m src.asr.timestamp_cache --splits train-clean-100 train-clean-360 dev-clean test-clean --check
```
