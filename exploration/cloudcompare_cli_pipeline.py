"""
CloudCompare CLI pipeline: ICP alignment + C2C distance + export.

This script assembles a robust CloudCompare command-line workflow to avoid
any GUI interaction. It relies on CloudCompare's headless mode and processes
two point clouds (moving vs reference):

- Load moving (T2) and reference (T1)
- Run ICP to align moving to reference
- Compute C2C distances (moving vs reference)
- Save the aligned moving cloud (with scalar field) to the requested output

Requirements
------------
- CloudCompare installed with CLI support (Windows/macOS/Linux)
- Optional LAZ support for output (LASzip)

Usage (PowerShell examples)
---------------------------
  # Using default synthetic dataset paths
  uv run scripts/cloudcompare_cli_pipeline.py \
    --out data/synthetic/synthetic_area/outputs/2020_aligned_with_c2c.laz

  # Explicit inputs and custom ICP/C2C tuning
  uv run scripts/cloudcompare_cli_pipeline.py \
    --ref data/synthetic/synthetic_area/2015/data/synthetic_tile_01.laz \
    --mov data/synthetic/synthetic_area/2020/data/synthetic_tile_01.laz \
    --out data/synthetic/synthetic_area/outputs/2020_aligned_with_c2c.laz \
    --icp-iter 60 --icp-overlap 80 --icp-sample 60000 --c2c-max-dist 5

Notes
-----
- If CloudCompare is not on PATH, set --cc-bin or the CLOUDCOMPARE_BIN env var.
- For LAZ exports, CloudCompare must be built with LASzip/LAZ support. If not,
  use a .las or .ply output extension.
- This script generates a temporary command file and calls CloudCompare with
  -SILENT and -AUTO_SAVE OFF to keep control of saving.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _default_repo_base() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def _guess_cc_bin(user_path: str | None) -> str:
    # Priority: explicit arg, env var, PATH, common Windows install paths
    if user_path:
        return str(Path(user_path).expanduser().resolve())
    env = os.environ.get("CLOUDCOMPARE_BIN")
    if env:
        return str(Path(env).expanduser().resolve())
    which = shutil.which("CloudCompare") or shutil.which("CloudCompare.exe")
    if which:
        return which
    # Common Windows locations
    for p in (
        r"C:\\Program Files\\CloudCompare\\CloudCompare.exe",
        r"C:\\Program Files (x86)\\CloudCompare\\CloudCompare.exe",
    ):
        if Path(p).exists():
            return p
    raise FileNotFoundError(
        "CloudCompare binary not found. Provide --cc-bin or set CLOUDCOMPARE_BIN."
    )


def _build_commands(
    mov: Path,
    ref: Path,
    out_path: Path,
    icp_iter: int,
    icp_overlap: int,
    icp_sample: int,
    c2c_max_dist: float | None,
    verbosity: int,
) -> list[str]:
    """Build command file lines for CloudCompare CLI.

    Load order matters:
      - First: moving (T2) as 'data' (will be moved)
      - Second: reference (T1) as 'model'
    Then: -ICP, then -C2C_DIST (compared=first, reference=second)
    """
    lines: list[str] = []
    # Reduce noise in logs (still printed to stdout by CC)
    lines.append(f"-VERBOSITY {max(0, min(4, verbosity))}")
    lines.append("-AUTO_SAVE OFF")

    # Load clouds (ensure absolute paths for CC)
    lines.append(f"-O \"{mov}\"")
    lines.append(f"-O \"{ref}\"")

    # ICP alignment (moving -> reference)
    # Defaults intentionally conservative; let caller tune via args
    icp = ["-ICP"]
    if icp_iter > 0:
        icp += ["-ITER", str(int(icp_iter))]
    if icp_overlap:
        icp += ["-OVERLAP", str(int(icp_overlap))]
    if icp_sample:
        icp += ["-RANDOM_SAMPLING_LIMIT", str(int(icp_sample))]
    # Keep a small minimum error difference for convergence
    icp += ["-MIN_ERROR_DIFF", "1e-6"]
    lines.append(" ".join(icp))

    # C2C distances (first=compared, second=reference)
    c2c = ["-C2C_DIST"]
    if c2c_max_dist is not None and c2c_max_dist > 0:
        c2c += ["-MAX_DIST", str(float(c2c_max_dist))]
    # A local model can improve robustness; keep default NN unless user requests
    lines.append(" ".join(c2c))

    # Export format from output suffix
    suffix = out_path.suffix.lower().lstrip(".")
    if suffix in {"laz", "las"}:
        # LAS format, custom extension if needed (LAZ support requires LASzip)
        lines.append("-C_EXPORT_FMT LAS")
        lines.append(f"-EXT {suffix}")
    elif suffix in {"ply"}:
        lines.append("-C_EXPORT_FMT PLY")
    elif suffix in {"e57"}:
        lines.append("-C_EXPORT_FMT E57")
    else:
        # Default to PLY if unknown
        lines.append("-C_EXPORT_FMT PLY")

    # Save only the first cloud (moving) with the new scalar field
    lines.append("-SELECT_ENTITIES -FIRST 1 -CLOUD")
    # -SAVE_CLOUDS FILE accepts one or multiple names (in order of selection)
    lines.append(f"-SAVE_CLOUDS FILE \"{out_path}\"")

    # Clear memory (optional)
    lines.append("-CLEAR")

    return lines


def run():
    parser = argparse.ArgumentParser(description="CloudCompare CLI pipeline (ICP + C2C + export)")
    parser.add_argument("--cc-bin", type=str, default=None, help="Path to CloudCompare executable")
    parser.add_argument("--ref", type=str, default=None, help="Reference cloud (e.g., 2015)")
    parser.add_argument("--mov", type=str, default=None, help="Moving cloud (e.g., 2020)")
    parser.add_argument(
        "--out", type=str, default=None, help="Output file for aligned moving cloud with C2C SF"
    )
    parser.add_argument("--icp-iter", type=int, default=60, help="ICP max iterations")
    parser.add_argument("--icp-overlap", type=int, default=80, help="ICP expected overlap [10..100]")
    parser.add_argument("--icp-sample", type=int, default=60000, help="ICP random sampling limit per iter")
    parser.add_argument("--c2c-max-dist", type=float, default=None, help="C2C max NN distance (units)")
    parser.add_argument("--verbosity", type=int, default=2, help="CloudCompare verbosity 0..4")

    args = parser.parse_args()

    repo = _default_repo_base()
    # Default synthetic dataset if not provided
    default_ref = repo / "data" / "synthetic" / "synthetic_area" / "2015" / "data" / "synthetic_tile_01.laz"
    default_mov = repo / "data" / "synthetic" / "synthetic_area" / "2020" / "data" / "synthetic_tile_01.laz"
    default_out = repo / "data" / "synthetic" / "synthetic_area" / "outputs" / "2020_aligned_with_c2c.laz"

    ref = Path(args.ref) if args.ref else default_ref
    mov = Path(args.mov) if args.mov else default_mov
    out_path = Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for p in (ref, mov):
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")

    cc_bin = _guess_cc_bin(args.cc_bin)

    commands = _build_commands(
        mov=mov.resolve(),
        ref=ref.resolve(),
        out_path=out_path.resolve(),
        icp_iter=args.icp_iter,
        icp_overlap=args.icp_overlap,
        icp_sample=args.icp_sample,
        c2c_max_dist=args.c2c_max_dist,
        verbosity=args.verbosity,
    )

    # Write a temporary command file and call CloudCompare
    with tempfile.TemporaryDirectory() as td:
        cmd_file = Path(td) / "cc_commands.txt"
        cmd_file.write_text("\n".join(commands) + "\n", encoding="utf-8")

        proc_args = [cc_bin, "-SILENT", "-COMMAND_FILE", str(cmd_file)]
        print("[cloudcompare-cli] Executing:")
        print("  ", " ".join(f'"{a}"' if " " in a else a for a in proc_args))

        try:
            result = subprocess.run(proc_args, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"CloudCompare not found: {cc_bin}")

        # Stream logs for visibility
        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(f"CloudCompare failed with exit code {result.returncode}")

        if not out_path.exists():
            raise RuntimeError(f"Expected output file not found: {out_path}")

        print(f"[cloudcompare-cli] Done. Output saved to: {out_path}")


if __name__ == "__main__":
    run()

