"""
LLM-jp-Moshi-v1 ランチャー（macOS Apple Silicon 用）

使用例:
    uv run python main.py              # デフォルト: q8, ポート 8998
    uv run python main.py -q 4        # 4bit 量子化（メモリ節約）
    uv run python main.py --no-browser # ブラウザ自動起動なし
    uv run python main.py --reconvert  # 強制再変換
"""

import argparse
import subprocess
import sys
from pathlib import Path

HF_REPO = "llm-jp/llm-jp-moshi-v1"
MODELS_DIR = Path(__file__).parent / "models"


def get_model_dir(quantize: int | None) -> Path:
    if quantize == 8:
        return MODELS_DIR / "llm-jp-moshi-mlx-q8"
    elif quantize == 4:
        return MODELS_DIR / "llm-jp-moshi-mlx-q4"
    else:
        return MODELS_DIR / "llm-jp-moshi-mlx-bf16"


def get_weight_name(quantize: int | None) -> str:
    if quantize == 8:
        return "model.q8.safetensors"
    elif quantize == 4:
        return "model.q4.safetensors"
    else:
        return "model.safetensors"


def needs_conversion(model_dir: Path, weight_name: str) -> bool:
    required = [
        model_dir / weight_name,
        model_dir / "config.json",
        model_dir / "tokenizer_spm_32k_3.model",
        model_dir / "tokenizer-e351c8d8-checkpoint125.safetensors",
    ]
    return not all(f.exists() for f in required)


def run_conversion(quantize: int | None, model_dir: Path) -> None:
    print(f"\nMLX 形式への変換を開始します: {model_dir}")
    print("（初回は 15GB のダウンロードと変換のため数分かかります）\n")
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "convert.py"),
        "--output-dir",
        str(model_dir),
    ]
    if quantize is not None:
        cmd += ["-q", str(quantize)]
    subprocess.run(cmd, check=True)


def launch_server(
    model_dir: Path,
    weight_name: str,
    quantize: int | None,
    port: int,
    host: str,
    no_browser: bool,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "moshi_mlx.local_web",
        "--moshi-weight",
        str(model_dir / weight_name),
        "--mimi-weight",
        str(model_dir / "tokenizer-e351c8d8-checkpoint125.safetensors"),
        "--tokenizer",
        str(model_dir / "tokenizer_spm_32k_3.model"),
        "--lm-config",
        str(model_dir / "config.json"),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if quantize is not None:
        cmd += ["-q", str(quantize)]
    if no_browser:
        cmd.append("--no-browser")

    print(f"\nMoshi サーバーを起動中: http://{host}:{port}")
    print("ヘッドフォン推奨（エコー防止）。停止するには Ctrl+C\n")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-jp-Moshi-v1 ランチャー（macOS Apple Silicon）"
    )
    parser.add_argument(
        "-q",
        "--quantize",
        type=int,
        choices=[4, 8],
        default=8,
        help="量子化レベル（デフォルト: 8）",
    )
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    parser.add_argument(
        "--reconvert",
        action="store_true",
        help="変換済みモデルが存在しても強制再変換する",
    )
    args = parser.parse_args()

    model_dir = get_model_dir(args.quantize)
    weight_name = get_weight_name(args.quantize)

    if args.reconvert or needs_conversion(model_dir, weight_name):
        run_conversion(args.quantize, model_dir)
    else:
        print(f"変換済みモデルを使用します: {model_dir}")

    launch_server(
        model_dir,
        weight_name,
        args.quantize,
        args.port,
        args.host,
        args.no_browser,
    )


if __name__ == "__main__":
    main()
