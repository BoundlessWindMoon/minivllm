#!/usr/bin/env python3
"""Start the mini-vllm HTTP server.

Usage:
    python scripts/start_server.py --config configs/default.yaml
    python scripts/start_server.py --config configs/default.yaml --host 0.0.0.0 --port 8000
"""

import argparse
import sys
import os

# Ensure project root is on the path when run from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.distributed as dist
import uvicorn

from utils.config import GlobalConfig, print_runtime_config
import server as _server_module


def main():
    parser = argparse.ArgumentParser(description="mini-vllm server")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    cfg = GlobalConfig.from_yaml(args.config)
    _server_module._cfg = cfg

    torch.set_default_dtype(cfg.env.get_torch_dtype())
    torch.set_default_device(cfg.env.device)

    dist.init_process_group(
        backend=cfg.env.distributed.backend if torch.cuda.is_available() else "gloo",
        init_method=cfg.env.distributed.init_method,
        world_size=cfg.env.distributed.world_size,
        rank=cfg.env.distributed.rank,
    )

    print_runtime_config(cfg)
    uvicorn.run(_server_module.app, host=args.host, port=args.port)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
