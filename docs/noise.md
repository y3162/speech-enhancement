[戻る](../README.md)
---

## JSON 形式 (v1.0)

`target_range` は clean のどこに加算するか。`noise_valid_range` はノイズファイルの使用区間（ファイル長に対する比率）。フレームは `[floor(start_ratio * N), floor(end_ratio * N))`。区間が clean より短いときはその範囲内で周回する。

```json
{
    "version": "1.0",
    "pipeline": [
        {
            "method": "additive",
            "params": {
                "seed": 1234, // for noise start frame randomization
                "noise_id": "1234", // noises-table id
                "target_range": {
                    "type": "all"
                },
                "noise_valid_range": {
                    "start_ratio": 0.3,
                    "end_ratio": 0.5
                },
                "snr_db": 5.0
            }
        },
        ...
    ]
}
```

## DEMAND の制約

`noises` には全チャンネルを入れる。`noise_configs` は各環境の `ch01.wav` のみを使う。

1 ファイルを時間で分割し、同じ環境が train / dev / test に出るようにする。SNR は -10〜10（1 刻み）を各区間について作る。

- train: `[0.0, 0.8)` を 0.1 刻みで 8 区間
- dev: `[0.8, 0.9)`
- test: `[0.9, 1.0)`
