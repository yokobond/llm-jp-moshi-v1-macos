"""
PyTorch → MLX 変換スクリプト
llm-jp/llm-jp-moshi-v1 の重みを moshi_mlx 用に変換する。

使用例:
    python convert.py --output-dir models/llm-jp-moshi-mlx-q8 -q 8
    python convert.py --output-dir models/llm-jp-moshi-mlx-q4 -q 4
    python convert.py --output-dir models/llm-jp-moshi-mlx-bf16
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from huggingface_hub import hf_hub_download
from moshi_mlx import models

HF_REPO = "llm-jp/llm-jp-moshi-v1"


def convert(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] モデル設定をダウンロード中: {HF_REPO}/moshi_lm_kwargs.json")
    config_file = hf_hub_download(HF_REPO, "moshi_lm_kwargs.json")
    with open(config_file) as f:
        config_dict = json.load(f)

    print(f"[2/5] モデル重みをダウンロード中（キャッシュ済みの場合はスキップ）: model.safetensors")
    model_file = hf_hub_download(HF_REPO, "model.safetensors")

    print("[3/5] トークナイザーファイルをダウンロード中")
    tokenizer_file = hf_hub_download(HF_REPO, "tokenizer_spm_32k_3.model")
    mimi_file = hf_hub_download(
        HF_REPO, "tokenizer-e351c8d8-checkpoint125.safetensors"
    )

    print("[4/5] MLX モデルを構築中")
    lm_config = models.LmConfig.from_config_dict(config_dict)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)

    if args.quantize == 8:
        weight_name = "model.q8.safetensors"
    elif args.quantize == 4:
        weight_name = "model.q4.safetensors"
    else:
        weight_name = "model.safetensors"

    # PyTorch重みを先にロードしてからキーマッピング → その後に量子化
    print(f"     PyTorch 重みをロードして MLX 形式にキーマッピング中...")
    model.load_pytorch_weights(model_file, lm_config, strict=True)
    mx.eval(model.parameters())

    if args.quantize == 8:
        print("     量子化: 8bit (group_size=64)")
        nn.quantize(model, bits=8, group_size=64)
    elif args.quantize == 4:
        print("     量子化: 4bit (group_size=32)")
        nn.quantize(model, bits=4, group_size=32)
    else:
        print("     量子化なし (bfloat16)")
    mx.eval(model.parameters())

    print(f"[5/5] MLX 重みを保存中: {output_dir / weight_name}")
    flat_weights = dict(mlx.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(output_dir / weight_name), flat_weights)

    # config.json: moshi_lm_kwargs.json の内容 + ファイル名情報
    config_for_mlx = dict(config_dict)
    config_for_mlx["moshi_name"] = weight_name
    config_for_mlx["tokenizer_name"] = "tokenizer_spm_32k_3.model"
    config_for_mlx["mimi_name"] = "tokenizer-e351c8d8-checkpoint125.safetensors"
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_for_mlx, f, indent=2)

    shutil.copy2(tokenizer_file, output_dir / "tokenizer_spm_32k_3.model")
    shutil.copy2(
        mimi_file,
        output_dir / "tokenizer-e351c8d8-checkpoint125.safetensors",
    )

    print(f"\n変換完了！モデルを保存しました: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="llm-jp/llm-jp-moshi-v1 PyTorch → MLX 変換"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="変換後のモデルを保存するディレクトリ",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        type=int,
        choices=[4, 8],
        default=None,
        help="量子化ビット数 (4 or 8)。省略時は bfloat16",
    )
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
