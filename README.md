# moshi-ja

[LLM-jp-Moshi-v1](https://huggingface.co/llm-jp/llm-jp-moshi-v1)（国立情報学研究所が公開した世界初の商用利用可能な日本語全二重音声対話モデル）を、macOS Apple Silicon 上で動かすためのランチャーです。

## 動作環境

- macOS（Apple Silicon: M1/M2/M3/M4）
- Python 3.12
- RAM: 16GB 以上（q8 量子化の場合）、8GB 以上（q4 量子化の場合）
- ディスク空き容量: 25GB 以上（モデルのダウンロード・変換後）

## セットアップ

[uv](https://docs.astral.sh/uv/) を使います。

```bash
# 依存パッケージをインストール
uv sync
```

## 起動

```bash
uv run python main.py
```

初回起動時は以下の処理が自動で行われます（数分〜十数分かかります）：

1. HuggingFace から `llm-jp/llm-jp-moshi-v1` の重みをダウンロード（約 16GB）
2. PyTorch 形式から MLX 形式に変換・量子化（デフォルト: 8bit）
3. ブラウザで `http://localhost:8998` を自動で開く

2回目以降は変換済みモデルをそのまま使用するため、すぐに起動します。

## オプション

| オプション | 説明 | デフォルト |
| --- | --- | --- |
| `-q 8` | 8bit 量子化（品質と速度のバランス） | ✓ |
| `-q 4` | 4bit 量子化（メモリ節約、RAM 8GB 以上で動作） | |
| `--no-browser` | ブラウザを自動で開かない | |
| `--reconvert` | 変換済みモデルが存在しても強制的に再変換する | |
| `--port PORT` | サーバーのポート番号 | `8998` |
| `--host HOST` | サーバーのホスト | `localhost` |

```bash
# 4bit 量子化で起動（メモリ節約）
uv run python main.py -q 4

# ブラウザを自動で開かずに起動
uv run python main.py --no-browser
```

## 使い方

1. ブラウザで `http://localhost:8998` を開く
2. **ヘッドフォンを着用**（スピーカーだとエコーが発生します）
3. マイクを有効にして日本語で話しかける
4. 停止するには `Ctrl+C`

## モデルについて

- **モデル名:** LLM-jp-Moshi-v1
- **開発元:** 国立情報学研究所（NII）・LLM-jp 研究会・早稲田大学・慶應義塾大学
- **パラメータ数:** 約 70 億
- **ライセンス:** Apache 2.0（商用利用可能）
- **HuggingFace:** [llm-jp/llm-jp-moshi-v1](https://huggingface.co/llm-jp/llm-jp-moshi-v1)
- **発表:** [NII プレスリリース（2026年2月25日）](https://www.nii.ac.jp/news/release/2026/0225.html)
