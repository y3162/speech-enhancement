[戻る](../README.md)
---

## シンボリックリンクの作成

全コーパスを `./corpora` にシンボリックリンクする。

[config.py](./src/utils/database/config.py)にあるように、全コーパスは同一親ディレクトリに配置されていることを仮定している。

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
export SPEECH_UTILS_CORPORA_ROOT_DIR=./corpora
export SPEECH_UTILS_DB_ROOT_DIR=./data/db
```

## メタデータのインポート

```bash
python -m src.utils.database.utterances --corpus librispeech --force
python -m src.utils.database.utterances --corpus libritts --force
python -m src.utils.database.utterances --corpus vctk --force
python -m src.utils.database.utterances --corpus demand --force
```
