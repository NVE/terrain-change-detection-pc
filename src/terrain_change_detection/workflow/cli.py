"""
CLI argument parser and override builder for the terrain change detection workflow.

This module owns the ``argparse`` definition and the translation of dedicated
CLI flags into config-style dot-path overrides.  It is imported by the
compatibility shim in ``scripts/run_workflow.py`` and by ``workflow.runner``.
"""

from __future__ import annotations

import argparse
import sys

from terrain_change_detection.utils.config import AppConfig, load_config


def build_cli_overrides(args: argparse.Namespace) -> list[str]:
    """Translate dedicated CLI flags into config-style dot-path overrides."""
    overrides = list(args.set_overrides or [])

    if args.base_dir:
        overrides.append(f"paths.base_dir={args.base_dir}")
    if args.seed is not None:
        overrides.append(f"alignment.random_seed={args.seed}")
    if args.reference is not None:
        overrides.append(f"alignment.reference={args.reference}")

    if args.m3c2_radius is not None:
        overrides.extend(
            [
                "detection.m3c2.use_autotune=false",
                f"detection.m3c2.fixed.radius={args.m3c2_radius}",
            ]
        )
        if args.m3c2_normal_scale is None:
            overrides.append("detection.m3c2.fixed.normal_scale=null")
        if args.m3c2_depth_factor is None:
            overrides.append("detection.m3c2.fixed.depth_factor=null")

    if args.m3c2_normal_scale is not None:
        overrides.extend(
            [
                "detection.m3c2.use_autotune=false",
                f"detection.m3c2.fixed.normal_scale={args.m3c2_normal_scale}",
            ]
        )
    if args.m3c2_depth_factor is not None:
        overrides.extend(
            [
                "detection.m3c2.use_autotune=false",
                f"detection.m3c2.fixed.depth_factor={args.m3c2_depth_factor}",
            ]
        )

    return overrides


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser with all current flags."""
    parser = argparse.ArgumentParser(description="Terrain Change Detection Workflow")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory containing area folders (e.g., data/raw or data/synthetic)",
    )
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        default=None,
        help=(
            "Repeatable override YAML layered on top of config/default.yaml. "
            "Example: --config config/profiles/large_scale.yaml"
        ),
    )
    parser.add_argument(
        "--set",
        "-s",
        action="append",
        dest="set_overrides",
        default=None,
        help=(
            "Override any config value with KEY=VALUE dot notation. "
            "Example: --set discovery.source_type=drone"
        ),
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the resolved configuration YAML and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for NumPy RNG to make subsampling/core selection reproducible.",
    )
    parser.add_argument(
        "--cores-file",
        type=str,
        default=None,
        help="Path to a .npy file with core points to LOAD if it exists; otherwise SAVE selected cores to this path.",
    )
    parser.add_argument(
        "--m3c2-radius",
        type=float,
        default=None,
        help="Override M3C2 radius (meters). Sets projection_scale=cylinder_radius=radius for both modes.",
    )
    parser.add_argument(
        "--m3c2-normal-scale",
        type=float,
        default=None,
        help="Override M3C2 normal_scale (meters). Defaults to radius when not set.",
    )
    parser.add_argument(
        "--m3c2-depth-factor",
        type=float,
        default=None,
        help="Override max_depth factor so that max_depth = depth_factor * radius (default from config).",
    )
    parser.add_argument(
        "--debug-m3c2-compare",
        action="store_true",
        help="Run both streaming and in-memory M3C2 on the same core points and print sign/correlation diagnostics.",
    )
    parser.add_argument(
        "--area-name",
        type=str,
        default=None,
        help="Specify the area name to process.",
    )
    parser.add_argument(
        "--show-plots",
        type=bool,
        default=False,
        help="If True, show plots interactively instead of saving to files.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="List of years to select for processing (e.g., --years 2020 2021).",
    )
    parser.add_argument(
        "--save-dems",
        type=bool,
        default=False,
        help="If True, generate DEMs after ICP and saves them to disk.",
    )
    parser.add_argument(
        "--reference",
        choices=["t1", "t2"],
        default=None,
        help="Which epoch is the ICP reference (t1=earlier, t2=later). Overrides config.",
    )
    return parser


def parse_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, AppConfig, list[str]]:
    """Parse CLI arguments, load config, and return the resolved triple.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        ``(args, cfg, cli_overrides)`` — the parsed namespace, the fully
        resolved :class:`AppConfig`, and the list of dot-path overrides that
        were applied on top of the YAML files.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if (
        args.m3c2_normal_scale is not None or args.m3c2_depth_factor is not None
    ) and args.m3c2_radius is None:
        parser.error(
            "--m3c2-normal-scale and --m3c2-depth-factor require --m3c2-radius"
        )

    cli_overrides = build_cli_overrides(args)

    cfg: AppConfig = load_config(
        config_paths=args.config,
        overrides=cli_overrides,
        allow_missing=False,
    )

    # Handle --show-config fast path
    if args.show_config:
        import yaml

        print(yaml.safe_dump(cfg.model_dump(), sort_keys=False), end="")
        sys.exit(0)

    return args, cfg, cli_overrides


def main(argv: list[str] | None = None) -> None:
    """Entry point for the workflow — called by the compatibility shim."""
    from .runner import run

    args, cfg, cli_overrides = parse_args(argv)
    run(args, cfg, cli_overrides)
