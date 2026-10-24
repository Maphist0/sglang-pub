#!/usr/bin/env python3
"""Launch the reviewed MiniCPM runtime using explicitly supplied local paths."""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "python",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--served-model-name", default="minicpm-v4.7-35b-int4")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    work = args.work_dir.resolve()
    model = args.model_path.resolve()
    cuda = args.cuda_home.resolve()
    cache = args.cache_dir.resolve()
    # Keep a virtual environment executable symlink intact.
    python = args.python.absolute()

    overrides = {
        "PYTHONPATH": str(args.source_dir.resolve()),
        "CUDA_HOME": str(cuda),
        "PATH": str(cuda / "bin") + os.pathsep + os.environ.get("PATH", ""),
        "TMPDIR": str(args.tmp_dir.resolve()),
        "XDG_CACHE_HOME": str(cache),
        "HF_HOME": str(cache / "huggingface"),
        "MODELSCOPE_CACHE": str(cache / "modelscope"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch-extensions"),
        "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": str(config.get("threads", 8)),
        "SGLANG_RUN_ID": "minicpm47-" + config["name"],
        "MINICPM47_AUDIT_PATH": str(
            work / "results" / (config["name"] + "-runtime-audit.json")
        ),
    }
    overrides.update(config.get("env", {}))
    command = [
        str(python),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model),
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--context-length",
        str(config.get("context", 4096)),
        "--max-total-tokens",
        str(config.get("context", 4096)),
        "--max-running-requests",
        "1",
        "--chunked-prefill-size",
        str(config.get("chunk", 256)),
        "--mem-fraction-static",
        str(config.get("mem_fraction", 0.82)),
        "--disable-shared-experts-fusion",
        "--attention-backend",
        config.get("attention", "triton"),
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": False}),
    ]
    if not config.get("prefix_cache", False):
        command.append("--disable-radix-cache")
    flags = config.get("flags", [])
    if "--kt-method" in flags:
        command.extend(["--kt-weight-path", str(model)])
    command.extend(flags)
    if args.dry_run:
        # Do not log inherited environment variables or filesystem locations.
        path_values = {
            str(python): "<python>",
            str(model): "<model>",
        }
        print(
            json.dumps(
                {"command": [path_values.get(value, value) for value in command]},
                indent=2,
            )
        )
        return
    os.chdir(work)
    os.execve(str(python), command, dict(os.environ, **overrides))


if __name__ == "__main__":
    main()
