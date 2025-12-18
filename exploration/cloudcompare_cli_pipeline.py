"""
CloudCompare CLI Pipeline: Load, ICP Alignment, M3C2 Distance.

This script demonstrates a terrain change detection workflow using
CloudCompare's command-line interface (CLI). It performs:

1. Loading point clouds (reference T1 and moving T2)
2. ICP alignment (registration of T2 to T1)
3. M3C2 distance computation (robust multi-scale change detection)
4. Export of aligned cloud with M3C2 scalar fields

The script generates a CloudCompare command file and executes it in headless mode.
This enables reproducible, automated batch processing without GUI interaction.

M3C2 PARAMETER FILE
-------------------
M3C2 requires a parameter file. Generate one via:
1. CloudCompare GUI: Tools -> Distances -> M3C2 -> Save parameters
2. Or use the default template created by this script

Requirements
------------
- CloudCompare installed with CLI support and M3C2 plugin
- LAZ support requires LASzip plugin (typically included)
- Python 3.8+ with laspy/lazrs for post-processing (optional)

Usage Examples (PowerShell)
---------------------------
# Basic usage with synthetic test data (creates default M3C2 params)
python exploration/cloudcompare_cli_pipeline.py

# Custom M3C2 parameter file
python exploration/cloudcompare_cli_pipeline.py --m3c2-params my_params.txt

# Dry run to see commands without executing
python exploration/cloudcompare_cli_pipeline.py --dry-run

Environment Variables
---------------------
CLOUDCOMPARE_BIN : Path to CloudCompare executable (if not on PATH)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the CloudCompare CLI pipeline."""
    
    # Input files
    reference_path: Path
    moving_path: Path
    output_path: Path
    
    # CloudCompare executable
    cc_bin: Optional[str] = None
    
    # ICP parameters
    icp_iterations: int = 60
    icp_overlap: int = 80  # percentage (10-100)
    icp_sample_limit: int = 60000
    icp_min_error: float = 1e-6
    icp_reference_is_first: bool = False  # If True, first cloud is reference
    icp_adjust_scale: bool = False
    icp_farthest_removal: bool = False
    
    # Preprocessing
    enable_sor: bool = False  # Statistical Outlier Removal
    sor_neighbors: int = 8
    sor_sigma: float = 2.0
    
    enable_subsampling: bool = False
    subsample_method: str = "SPATIAL"  # RANDOM, SPATIAL, OCTREE
    subsample_param: float = 0.1  # distance for SPATIAL, count for RANDOM, level for OCTREE
    
    # M3C2 Distance computation
    m3c2_params_file: Optional[Path] = None  # Required for M3C2
    m3c2_normal_scale: float = 1.0
    m3c2_projection_scale: float = 1.0
    m3c2_max_depth: float = 5.0
    
    # Output options
    output_format: str = "LAS"  # LAS, PLY, E57, BIN
    output_extension: Optional[str] = None  # Override extension (e.g., laz)
    no_timestamp: bool = True  # Disable automatic timestamp suffix
    
    # Execution options
    verbosity: int = 2  # 0=verbose, 4=errors only
    dry_run: bool = False  # Print commands without executing
    keep_command_file: bool = False  # Save command file for debugging
    
    # Post-processing
    analyze_output: bool = False  # Run analysis on output file


# ==============================================================================
# CLOUDCOMPARE BINARY DETECTION
# ==============================================================================

def find_cloudcompare(user_path: Optional[str] = None) -> str:
    """Find the CloudCompare executable.
    
    Search order:
    1. User-provided path
    2. CLOUDCOMPARE_BIN environment variable
    3. System PATH
    4. Common installation locations (Windows/Linux/macOS)
    """
    # User-provided path
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"CloudCompare not found at: {user_path}")
    
    # Environment variable
    env_path = os.environ.get("CLOUDCOMPARE_BIN")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return str(p)
        print(f"[WARNING] CLOUDCOMPARE_BIN set but not found: {env_path}")
    
    # System PATH
    which_result = shutil.which("CloudCompare") or shutil.which("CloudCompare.exe")
    if which_result:
        return which_result
    
    # Common installation locations
    common_paths = [
        # Windows
        r"C:\Program Files\CloudCompare\CloudCompare.exe",
        r"C:\Program Files (x86)\CloudCompare\CloudCompare.exe",
        # Linux
        "/usr/bin/CloudCompare",
        "/usr/local/bin/CloudCompare",
        "/opt/CloudCompare/CloudCompare",
        # macOS
        "/Applications/CloudCompare.app/Contents/MacOS/CloudCompare",
    ]
    
    for path in common_paths:
        if Path(path).exists():
            return path
    
    raise FileNotFoundError(
        "CloudCompare executable not found.\n"
        "Options:\n"
        "  1. Add CloudCompare to your PATH\n"
        "  2. Set CLOUDCOMPARE_BIN environment variable\n"
        "  3. Use --cc-bin argument"
    )


# ==============================================================================
# COMMAND FILE GENERATION
# ==============================================================================

def build_command_file(config: PipelineConfig) -> List[str]:
    """Build CloudCompare command file lines.
    
    The command file uses one command per line, which CloudCompare
    processes sequentially (state machine model).
    
    Load order is important for ICP and M3C2:
    - First loaded cloud = "compared" cloud (the one that gets aligned/distances)
    - Second loaded cloud = "reference" cloud
    
    So we load: moving (T2) first, then reference (T1)
    """
    commands: List[str] = []
    
    # Header comment (for debugging)
    commands.append("# CloudCompare CLI Pipeline - Load -> ICP -> M3C2")
    commands.append(f"# Reference: {config.reference_path}")
    commands.append(f"# Moving: {config.moving_path}")
    commands.append(f"# Output: {config.output_path}")
    commands.append("")
    
    # Global settings
    commands.append(f"-VERBOSITY {max(0, min(4, config.verbosity))}")
    commands.append("-AUTO_SAVE OFF")
    if config.no_timestamp:
        commands.append("-NO_TIMESTAMP")
    
    # Load point clouds
    # Order: moving first (will be aligned), reference second
    commands.append("")
    commands.append("# Load point clouds")
    commands.append(f'-O "{config.moving_path.resolve()}"')
    commands.append(f'-O "{config.reference_path.resolve()}"')
    
    # Preprocessing: Subsampling (optional)
    if config.enable_subsampling:
        commands.append("")
        commands.append("# Subsampling")
        if config.subsample_method == "RANDOM":
            commands.append(f"-SS RANDOM {int(config.subsample_param)}")
        elif config.subsample_method == "SPATIAL":
            commands.append(f"-SS SPATIAL {config.subsample_param}")
        elif config.subsample_method == "OCTREE":
            commands.append(f"-SS OCTREE {int(config.subsample_param)}")
    
    # Preprocessing: Statistical Outlier Removal (optional)
    if config.enable_sor:
        commands.append("")
        commands.append("# Statistical Outlier Removal")
        commands.append(f"-SOR {config.sor_neighbors} {config.sor_sigma}")
    
    # ICP Registration
    commands.append("")
    commands.append("# ICP Registration (align first cloud to second)")
    
    icp_parts = ["-ICP"]
    if config.icp_iterations > 0:
        icp_parts.extend(["-ITER", str(config.icp_iterations)])
    if config.icp_overlap:
        icp_parts.extend(["-OVERLAP", str(config.icp_overlap)])
    if config.icp_sample_limit:
        icp_parts.extend(["-RANDOM_SAMPLING_LIMIT", str(config.icp_sample_limit)])
    if config.icp_min_error:
        icp_parts.extend(["-MIN_ERROR_DIFF", str(config.icp_min_error)])
    if config.icp_reference_is_first:
        icp_parts.append("-REFERENCE_IS_FIRST")
    if config.icp_adjust_scale:
        icp_parts.append("-ADJUST_SCALE")
    if config.icp_farthest_removal:
        icp_parts.append("-FARTHEST_REMOVAL")
    
    commands.append(" ".join(icp_parts))
    
    # M3C2 Distance computation
    commands.append("")
    commands.append("# M3C2 Distance computation")
    if config.m3c2_params_file and config.m3c2_params_file.exists():
        commands.append(f'-M3C2 "{config.m3c2_params_file.resolve()}"')
    else:
        commands.append("# WARNING: M3C2 requires a parameter file!")
        commands.append("# Generate one via CloudCompare GUI: Tools -> Distances -> M3C2 -> Save params")
        commands.append("# Or run with --create-m3c2-params to create a default template")
        commands.append("# Skipping M3C2 - no parameter file provided")
    
    # Export settings
    commands.append("")
    commands.append("# Export settings")
    commands.append(f"-C_EXPORT_FMT {config.output_format}")
    
    ext = config.output_extension or config.output_path.suffix.lstrip(".")
    if ext.lower() in ("laz", "las"):
        commands.append(f"-EXT {ext.lower()}")
    
    # Select and save the aligned cloud (first cloud with distances)
    commands.append("")
    commands.append("# Save aligned cloud with M3C2 distances")
    commands.append("-SELECT_ENTITIES -FIRST 1 -CLOUD")
    commands.append(f'-SAVE_CLOUDS FILE "{config.output_path.resolve()}"')
    
    # Cleanup
    commands.append("")
    commands.append("# Cleanup")
    commands.append("-CLEAR")
    
    return commands


# ==============================================================================
# EXECUTION
# ==============================================================================

def execute_cloudcompare(
    cc_bin: str,
    command_file: Path,
    capture_output: bool = True
) -> subprocess.CompletedProcess:
    """Execute CloudCompare with the given command file."""
    
    args = [cc_bin, "-SILENT", "-COMMAND_FILE", str(command_file)]
    
    print("\n" + "=" * 60)
    print("[CloudCompare CLI] Executing:")
    print(f"  {' '.join(args)}")
    print("=" * 60 + "\n")
    
    result = subprocess.run(
        args,
        capture_output=capture_output,
        text=True
    )
    
    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result


# ==============================================================================
# POST-PROCESSING & ANALYSIS
# ==============================================================================

def analyze_output_file(output_path: Path) -> Dict[str, Any]:
    """Analyze the output point cloud file and compute statistics.
    
    Requires laspy for LAS/LAZ files or plyfile for PLY files.
    """
    print("\n" + "=" * 60)
    print("[Analysis] Analyzing output file...")
    print("=" * 60 + "\n")
    
    results: Dict[str, Any] = {
        "file": str(output_path),
        "file_size_mb": output_path.stat().st_size / (1024 * 1024),
    }
    
    suffix = output_path.suffix.lower()
    
    if suffix in (".las", ".laz"):
        try:
            import laspy
            import numpy as np
            
            las = laspy.read(str(output_path))
            results["point_count"] = len(las.points)
            results["point_format"] = las.header.point_format.id
            
            # Bounding box
            results["bbox"] = {
                "min": [float(las.header.mins[i]) for i in range(3)],
                "max": [float(las.header.maxs[i]) for i in range(3)],
            }
            
            # Scalar fields (extra dimensions)
            results["scalar_fields"] = list(las.point_format.dimension_names)
            
            # Find M3C2 distance field
            distance_fields = [
                d for d in las.point_format.dimension_names
                if "m3c2" in d.lower() or "dist" in d.lower()
            ]
            
            if distance_fields:
                dist_field = distance_fields[0]
                distances = np.array(las[dist_field])
                valid = distances[~np.isnan(distances)]
                
                if len(valid) > 0:
                    results["distance_stats"] = {
                        "field_name": dist_field,
                        "count": int(len(valid)),
                        "min": float(np.min(valid)),
                        "max": float(np.max(valid)),
                        "mean": float(np.mean(valid)),
                        "std": float(np.std(valid)),
                        "median": float(np.median(valid)),
                        "p5": float(np.percentile(valid, 5)),
                        "p95": float(np.percentile(valid, 95)),
                    }
                    
                    # Significant changes (> 2*std from mean)
                    threshold = results["distance_stats"]["mean"] + 2 * results["distance_stats"]["std"]
                    significant = np.sum(valid > threshold)
                    results["distance_stats"]["significant_changes"] = int(significant)
                    results["distance_stats"]["significant_percent"] = float(significant / len(valid) * 100)
            
            print(f"  Point count: {results['point_count']:,}")
            print(f"  Scalar fields: {', '.join(results['scalar_fields'][:5])}")
            if "distance_stats" in results:
                stats = results["distance_stats"]
                print(f"  Distance field: {stats['field_name']}")
                print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"    Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"    Significant changes (>2σ): {stats['significant_changes']:,} ({stats['significant_percent']:.1f}%)")
            
        except ImportError:
            print("  [WARNING] laspy not installed. Install with: pip install laspy[lazrs]")
            results["error"] = "laspy not installed"
        except Exception as e:
            print(f"  [ERROR] Analysis failed: {e}")
            results["error"] = str(e)
    
    elif suffix == ".ply":
        try:
            from plyfile import PlyData
            import numpy as np
            
            ply = PlyData.read(str(output_path))
            vertex = ply["vertex"]
            results["point_count"] = len(vertex.data)
            results["properties"] = [p.name for p in vertex.properties]
            
            print(f"  Point count: {results['point_count']:,}")
            print(f"  Properties: {', '.join(results['properties'][:5])}")
            
        except ImportError:
            print("  [WARNING] plyfile not installed. Install with: pip install plyfile")
            results["error"] = "plyfile not installed"
        except Exception as e:
            print(f"  [ERROR] Analysis failed: {e}")
            results["error"] = str(e)
    
    else:
        print(f"  [WARNING] Analysis not supported for {suffix} files")
        results["error"] = f"Unsupported format: {suffix}"
    
    return results


def export_analysis_json(results: Dict[str, Any], output_path: Path) -> None:
    """Export analysis results to JSON file."""
    json_path = output_path.with_suffix(".analysis.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Analysis saved to: {json_path}")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def get_default_paths() -> tuple[Path, Path, Path]:
    """Get default input/output paths for testing."""
    repo_root = Path(__file__).resolve().parent.parent
    
    ref = repo_root / "data" / "synthetic" / "synthetic_area" / "2015" / "data" / "synthetic_tile_01.laz"
    mov = repo_root / "data" / "synthetic" / "synthetic_area" / "2020" / "data" / "synthetic_tile_01.laz"
    out = repo_root / "data" / "synthetic" / "synthetic_area" / "outputs" / "2020_aligned_m3c2_cli.laz"
    
    return ref, mov, out


def create_default_m3c2_params(output_path: Path) -> Path:
    """Create a default M3C2 parameter file.
    
    These are reasonable defaults for terrain change detection.
    Adjust based on your data characteristics.
    """
    params = """[General]
M3C2VER=1
ExportDensityAtProjScale=false
ExportStdDevInfo=true
UseMinPoints4Stat=false
UsePrecisionMaps=false

[Normal]
Mode=0
PreferedOri=3
MinScale=1.0
MaxScale=1.0
Step=1.0
UseCorePointsForNormals=true

[Projection]
Scale=1.0
UseMedian=false
MaxDepth=5.0

[Registration]
ComputeConfidence=true
"""
    
    params_path = output_path.parent / "m3c2_params.txt"
    params_path.write_text(params)
    print(f"Created default M3C2 parameter file: {params_path}")
    return params_path


def main():
    """Main entry point with argument parsing."""
    
    default_ref, default_mov, default_out = get_default_paths()
    
    parser = argparse.ArgumentParser(
        description="CloudCompare CLI Pipeline: Load -> ICP -> M3C2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with synthetic test data (creates default M3C2 params)
  python cloudcompare_cli_pipeline.py

  # Custom M3C2 parameter file
  python cloudcompare_cli_pipeline.py --m3c2-params my_params.txt

  # Dry run to see commands without executing
  python cloudcompare_cli_pipeline.py --dry-run

  # With preprocessing
  python cloudcompare_cli_pipeline.py --enable-sor --analyze
        """
    )
    
    # Input/Output
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--ref", type=Path, default=default_ref,
                         help="Reference point cloud (T1, older epoch)")
    io_group.add_argument("--mov", type=Path, default=default_mov,
                         help="Moving point cloud (T2, newer epoch)")
    io_group.add_argument("--out", type=Path, default=default_out,
                         help="Output file path")
    io_group.add_argument("--cc-bin", type=str, default=None,
                         help="Path to CloudCompare executable")
    
    # ICP parameters
    icp_group = parser.add_argument_group("ICP Registration")
    icp_group.add_argument("--icp-iter", type=int, default=60,
                          help="Maximum ICP iterations (default: 60)")
    icp_group.add_argument("--icp-overlap", type=int, default=80,
                          help="Expected overlap percentage 10-100 (default: 80)")
    icp_group.add_argument("--icp-sample", type=int, default=60000,
                          help="Random sampling limit per iteration (default: 60000)")
    icp_group.add_argument("--icp-adjust-scale", action="store_true",
                          help="Allow scale adjustment during ICP")
    
    # Preprocessing
    pre_group = parser.add_argument_group("Preprocessing")
    pre_group.add_argument("--enable-sor", action="store_true",
                          help="Enable Statistical Outlier Removal")
    pre_group.add_argument("--sor-neighbors", type=int, default=8,
                          help="SOR number of neighbors (default: 8)")
    pre_group.add_argument("--sor-sigma", type=float, default=2.0,
                          help="SOR sigma multiplier (default: 2.0)")
    pre_group.add_argument("--enable-subsampling", action="store_true",
                          help="Enable spatial subsampling")
    pre_group.add_argument("--subsample-dist", type=float, default=0.1,
                          help="Subsampling minimum distance (default: 0.1)")
    
    # Distance computation (M3C2)
    dist_group = parser.add_argument_group("M3C2 Distance Computation")
    dist_group.add_argument("--m3c2-params", type=Path, default=None,
                           help="M3C2 parameters file (created if not provided)")
    dist_group.add_argument("--create-m3c2-params", action="store_true",
                           help="Create default M3C2 parameter file")
    
    # Execution options
    exec_group = parser.add_argument_group("Execution")
    exec_group.add_argument("--verbosity", type=int, default=2,
                           help="CloudCompare verbosity 0-4 (default: 2)")
    exec_group.add_argument("--dry-run", action="store_true",
                           help="Print commands without executing")
    exec_group.add_argument("--keep-cmd", action="store_true",
                           help="Keep command file for debugging")
    exec_group.add_argument("--analyze", action="store_true",
                           help="Analyze output file after processing")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.ref.exists():
        print(f"[ERROR] Reference file not found: {args.ref}")
        sys.exit(1)
    if not args.mov.exists():
        print(f"[ERROR] Moving file not found: {args.mov}")
        sys.exit(1)
    
    # Create output directory
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle M3C2 parameters
    m3c2_params = args.m3c2_params
    if m3c2_params is None or args.create_m3c2_params:
        m3c2_params = create_default_m3c2_params(args.out)
    
    # Build configuration
    config = PipelineConfig(
        reference_path=args.ref,
        moving_path=args.mov,
        output_path=args.out,
        cc_bin=args.cc_bin,
        icp_iterations=args.icp_iter,
        icp_overlap=args.icp_overlap,
        icp_sample_limit=args.icp_sample,
        icp_adjust_scale=args.icp_adjust_scale,
        enable_sor=args.enable_sor,
        sor_neighbors=args.sor_neighbors,
        sor_sigma=args.sor_sigma,
        enable_subsampling=args.enable_subsampling,
        subsample_method="SPATIAL",
        subsample_param=args.subsample_dist,
        m3c2_params_file=m3c2_params,
        verbosity=args.verbosity,
        dry_run=args.dry_run,
        keep_command_file=args.keep_cmd,
        analyze_output=args.analyze,
    )
    
    # Build command file
    commands = build_command_file(config)
    
    print("\n" + "=" * 60)
    print("[CloudCompare CLI Pipeline]")
    print("=" * 60)
    print(f"\nReference: {config.reference_path}")
    print(f"Moving:    {config.moving_path}")
    print(f"Output:    {config.output_path}")
    
    if config.dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
        print("-" * 40)
        for cmd in commands:
            if not cmd.startswith("#"):
                print(f"  {cmd}")
        print("-" * 40)
        return
    
    # Find CloudCompare
    try:
        cc_bin = find_cloudcompare(config.cc_bin)
        print(f"\nCloudCompare: {cc_bin}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    
    # Write and execute command file
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd_file = Path(tmpdir) / "cc_commands.txt"
        cmd_file.write_text("\n".join(commands), encoding="utf-8")
        
        if config.keep_command_file:
            keep_path = config.output_path.with_suffix(".cc_commands.txt")
            shutil.copy(cmd_file, keep_path)
            print(f"Command file saved: {keep_path}")
        
        # Execute CloudCompare
        result = execute_cloudcompare(cc_bin, cmd_file)
        
        if result.returncode != 0:
            print(f"\n[ERROR] CloudCompare failed with exit code {result.returncode}")
            sys.exit(result.returncode)
    
    # Verify output
    if not config.output_path.exists():
        print(f"\n[ERROR] Output file not created: {config.output_path}")
        sys.exit(1)
    
    print(f"\n[SUCCESS] Output saved to: {config.output_path}")
    print(f"  File size: {config.output_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Optional analysis
    if config.analyze_output:
        results = analyze_output_file(config.output_path)
        export_analysis_json(results, config.output_path)


if __name__ == "__main__":
    main()
