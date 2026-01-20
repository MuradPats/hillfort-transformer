#!/usr/bin/env python3
"""GPU and PyTorch sanity-check script.

Prints PyTorch and CUDA versions, availability, device count and device names.

Usage:
  python scripts/test_torch_gpu.py        # basic info
  python scripts/test_torch_gpu.py --verbose
"""
from __future__ import annotations

import argparse
import sys
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test PyTorch and GPU availability")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed device properties")
    args = parser.parse_args(argv)

    try:
        import torch
    except Exception as e:  # pragma: no cover - environment dependent
        print("Failed to import torch:", e, file=sys.stderr)
        return 2

    print("Torch:", getattr(torch, "__version__", "<unknown>"))
    print("CUDA version baked in:", getattr(torch.version, "cuda", "<unknown>"))

    cuda_available = torch.cuda.is_available()
    print("GPU available:", cuda_available)

    try:
        device_count = torch.cuda.device_count()
    except Exception:
        device_count = 0
    print("CUDA device count:", device_count)

    if cuda_available and device_count > 0:
        for i in range(device_count):
            try:
                name = torch.cuda.get_device_name(i)
            except Exception:
                name = "<unknown>"
            print(f"GPU {i}: {name}")

            if args.verbose:
                try:
                    props = torch.cuda.get_device_properties(i)
                    print("  - major/minor:", props.major, props.minor)
                    print("  - total_memory (GB):", round(props.total_memory / (1024 ** 3), 2))
                except Exception as e:
                    print("  - failed to read properties:", e)

    # Also print simple torch.cuda.current_device() if available
    try:
        if cuda_available:
            cur = torch.cuda.current_device()
            print("Current device id:", cur)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
