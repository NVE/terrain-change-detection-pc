"""
CloudCompare PythonRuntime Pipeline: Load, ICP Alignment, M3C2 Distance.

This script demonstrates a terrain change detection workflow using
CloudCompare's embedded Python plugin (PythonRuntime). It performs:

1. Loading point clouds (reference T1 and moving T2)
2. ICP alignment (registration of T2 to T1)
3. M3C2 distance computation (robust multi-scale change detection)
4. Export of results

IMPORTANT LIMITATION
--------------------
The M3C2 plugin is NOT fully exposed via CloudCompare's Python API.
This script demonstrates what IS possible and documents the limitations.
For production M3C2 workflows, use the CLI with -M3C2 command or our
custom implementation in src/terrain_change_detection/.

How to Run
----------
1. Open CloudCompare
2. Go to Plugins -> Python Runtime -> File Runner (or Python Editor)
3. Load this script and execute

Author: Terrain Change Detection Team
Date: December 2024
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, List

# ==============================================================================
# IMPORT CLOUDCOMPARE MODULES
# ==============================================================================

try:
    import pycc
    import cccorelib
except ImportError as e:
    raise RuntimeError(
        "This script must be run inside CloudCompare with PythonRuntime plugin.\n"
        f"Import error: {e}"
    )

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Configuration parameters for the pipeline."""
    
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent
    
    # Input paths
    T1_PATH = REPO_ROOT / "data" / "synthetic" / "synthetic_area" / "2015" / "data" / "synthetic_tile_01.laz"
    T2_PATH = REPO_ROOT / "data" / "synthetic" / "synthetic_area" / "2020" / "data" / "synthetic_tile_01.laz"
    
    # Output paths
    OUTPUT_DIR = REPO_ROOT / "data" / "synthetic" / "synthetic_area" / "outputs"
    OUTPUT_FILE = OUTPUT_DIR / "2020_aligned_m3c2_plugin.laz"
    
    # ICP parameters
    ICP_MAX_ITERATIONS = 60
    ICP_SAMPLE_LIMIT = 50000
    ICP_TRIM_RATIO = 0.8
    
    # M3C2 parameters (for reference - not all exposed via API)
    M3C2_NORMAL_SCALE = 1.0
    M3C2_PROJECTION_SCALE = 1.0
    M3C2_MAX_DEPTH = 5.0
    
    CLEAR_DB_ON_START = True
    
    @classmethod
    def validate(cls):
        errors = []
        if not cls.T1_PATH.exists():
            errors.append(f"Reference cloud not found: {cls.T1_PATH}")
        if not cls.T2_PATH.exists():
            errors.append(f"Moving cloud not found: {cls.T2_PATH}")
        return errors


# ==============================================================================
# LOGGING
# ==============================================================================

def log(msg, level="INFO"):
    formatted = f"[{level}] {msg}"
    try:
        if level == "ERROR":
            pycc.ccLog.Error(msg)
        elif level == "WARNING":
            pycc.ccLog.Warning(msg)
        else:
            pycc.ccLog.Print(formatted)
    except Exception:
        pass
    print(formatted)


def log_section(title):
    sep = "=" * 60
    log(sep)
    log(f"  {title}")
    log(sep)


# ==============================================================================
# UTILITIES
# ==============================================================================

def get_xyz(vec):
    try:
        return float(vec.x), float(vec.y), float(vec.z)
    except AttributeError:
        return float(vec[0]), float(vec[1]), float(vec[2])


def find_first_cloud(hobj):
    if hobj is None:
        return None
    if isinstance(hobj, pycc.ccPointCloud):
        return hobj
    try:
        for i in range(hobj.getChildrenNumber()):
            result = find_first_cloud(hobj.getChild(i))
            if result is not None:
                return result
    except Exception:
        pass
    return None


def load_point_cloud(path, name=None):
    CC = pycc.GetInstance()
    params = pycc.FileIOFilter.LoadParameters()
    params.parentWidget = CC.getMainWindow()
    
    log(f"Loading: {path}")
    result = CC.loadFile(str(path), params)
    
    cloud = find_first_cloud(result)
    if cloud is None:
        raise RuntimeError(f"No point cloud found in: {path}")
    
    if name:
        cloud.setName(name)
    
    log(f"  Loaded {cloud.size():,} points")
    return cloud


def duplicate_cloud(src, name):
    log(f"Duplicating cloud to '{name}'...")
    try:
        dup = src.clone()
        dup.setName(name)
        return dup
    except Exception:
        pass
    
    dup = pycc.ccPointCloud(name)
    dup.reserve(src.size())
    for i in range(src.size()):
        dup.addPoint(src.getPoint(i))
    return dup


def subsample_cloud(cloud, target_count):
    if cloud.size() <= target_count:
        return cloud
    
    log(f"Subsampling to ~{target_count:,} points...")
    try:
        result = cccorelib.CloudSamplingTools.subsampleCloudRandomly(cloud, target_count)
        if result is not None and result.size() > 0:
            return cloud.partialClone(result)
    except Exception as e:
        log(f"  Subsampling failed: {e}", level="WARNING")
    return cloud


# ==============================================================================
# ICP REGISTRATION
# ==============================================================================

def compute_icp(moving, reference, max_iter=60, sample_limit=50000, trim_ratio=0.8):
    """Compute ICP registration using SVD-based point-to-point alignment."""
    log_section("ICP Registration")
    
    import numpy as np
    
    mov_icp = subsample_cloud(moving, sample_limit) if moving.size() > sample_limit else moving
    ref_icp = subsample_cloud(reference, sample_limit) if reference.size() > sample_limit else reference
    
    log("Building KD-tree on reference...")
    kd = cccorelib.KDTree()
    kd.buildFromCloud(ref_icp)
    
    def cloud_to_array(c):
        pts = np.zeros((c.size(), 3), dtype=np.float64)
        for i in range(c.size()):
            pts[i] = get_xyz(c.getPoint(i))
        return pts
    
    A = cloud_to_array(mov_icp)
    B = cloud_to_array(ref_icp)
    
    R_total = np.eye(3)
    t_total = np.zeros(3)
    rms_history = []
    
    log(f"Starting ICP ({max_iter} max iterations)...")
    
    for it in range(max_iter):
        pairs_A, pairs_B, dists = [], [], []
        
        for i in range(A.shape[0]):
            a = A[i]
            try:
                nn = kd.findNearestNeighbour(cccorelib.CCVector3(float(a[0]), float(a[1]), float(a[2])))
                try:
                    idx, d2 = int(nn[0]), float(nn[1])
                except (TypeError, IndexError):
                    idx = int(nn)
                    d2 = float(np.sum((a - B[idx])**2))
                pairs_A.append(a)
                pairs_B.append(B[idx])
                dists.append(math.sqrt(d2))
            except Exception:
                continue
        
        if len(pairs_A) < 10:
            break
        
        pairs_A = np.array(pairs_A)
        pairs_B = np.array(pairs_B)
        dists = np.array(dists)
        
        if trim_ratio < 1.0:
            n_keep = max(10, int(len(dists) * trim_ratio))
            keep_idx = np.argsort(dists)[:n_keep]
            pairs_A, pairs_B, dists = pairs_A[keep_idx], pairs_B[keep_idx], dists[keep_idx]
        
        rms = float(np.sqrt(np.mean(dists**2)))
        rms_history.append(rms)
        
        cA, cB = pairs_A.mean(axis=0), pairs_B.mean(axis=0)
        H = (pairs_A - cA).T @ (pairs_B - cB)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = cB - R @ cA
        
        R_total = R @ R_total
        t_total = R @ t_total + t
        A = (A @ R.T) + t
        
        if (it + 1) % 10 == 0:
            log(f"  Iter {it+1}: RMS={rms:.6f}")
        
        if len(rms_history) >= 2 and abs(rms_history[-2] - rms_history[-1]) < 1e-7:
            log(f"  Converged at iteration {it+1}")
            break
    
    if rms_history:
        log(f"ICP done: Final RMS={rms_history[-1]:.6f}")
    
    return R_total, t_total, rms_history


def apply_transform(cloud, R, t):
    import numpy as np
    log(f"Applying transform to '{cloud.getName()}'...")
    
    for i in range(cloud.size()):
        p = cloud.getPoint(i)
        v = np.array(get_xyz(p))
        v2 = R @ v + t
        cloud.setPoint(i, cccorelib.CCVector3(float(v2[0]), float(v2[1]), float(v2[2])))


# ==============================================================================
# M3C2 DISTANCE COMPUTATION
# ==============================================================================

def compute_m3c2(cloud1, cloud2):
    """
    Attempt to compute M3C2 distances.
    
    LIMITATION: The M3C2 plugin is NOT exposed via CloudCompare's Python API.
    This function documents the limitation and provides guidance.
    
    For M3C2, use one of these alternatives:
    1. CloudCompare GUI: Tools -> Distances -> M3C2
    2. CloudCompare CLI: -M3C2 params.txt
    3. Our custom implementation: src/terrain_change_detection/detection/
    """
    log_section("M3C2 Distance Computation")
    
    log("IMPORTANT: M3C2 is NOT exposed via CloudCompare's Python API!")
    log("")
    log("The M3C2 plugin provides robust multi-scale change detection,")
    log("but it can only be accessed via:")
    log("  1. CloudCompare GUI: Tools -> Distances -> M3C2")
    log("  2. CloudCompare CLI: -M3C2 <params_file>")
    log("  3. Our custom implementation in src/terrain_change_detection/")
    log("")
    log("This script has loaded and aligned the clouds. You can now:")
    log("  - Use GUI menu to run M3C2 manually")
    log("  - Save clouds and use CLI with -M3C2 command")
    log("  - Use our production implementation for automated processing")
    log("")
    
    # Check if M3C2 plugin is available (it won't be callable, but we can check)
    try:
        # Try to access M3C2 - this will fail but shows the limitation
        plugins = dir(pycc)
        m3c2_related = [p for p in plugins if 'm3c2' in p.lower()]
        if m3c2_related:
            log(f"M3C2-related symbols found (may not be functional): {m3c2_related}")
        else:
            log("No M3C2 symbols found in pycc module")
    except Exception as e:
        log(f"Plugin check error: {e}")
    
    return None


def save_cloud(cloud, path):
    log(f"Saving to: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    CC = pycc.GetInstance()
    params = pycc.FileIOFilter.SaveParameters()
    params.parentWidget = CC.getMainWindow()
    
    try:
        pycc.FileIOFilter.SaveToFile(cloud, str(path), params)
        return True
    except Exception as e:
        log(f"Save failed: {e}", level="ERROR")
        return False


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_pipeline():
    """Execute the Load -> ICP -> M3C2 pipeline."""
    log_section("CloudCompare Python Plugin Pipeline")
    log("Workflow: Load -> ICP Alignment -> M3C2 Distance")
    
    CC = pycc.GetInstance()
    
    errors = Config.validate()
    if errors:
        for e in errors:
            log(e, level="ERROR")
        raise RuntimeError("Config validation failed")
    
    # Clear database
    if Config.CLEAR_DB_ON_START:
        log("Clearing database...")
        try:
            root = CC.dbRootObject()
            for i in range(root.getChildrenNumber() - 1, -1, -1):
                CC.removeFromDB(root.getChild(i))
        except Exception:
            pass
    
    # Step 1: Load Point Clouds
    log_section("Step 1: Loading Point Clouds")
    ref = load_point_cloud(Config.T1_PATH, "Reference_T1_2015")
    mov = load_point_cloud(Config.T2_PATH, "Moving_T2_2020")
    CC.addToDB(ref)
    CC.addToDB(mov)
    CC.updateUI()
    
    # Create working copies
    ref_work = duplicate_cloud(ref, "Reference_Work")
    mov_work = duplicate_cloud(mov, "Moving_Work")
    CC.addToDB(ref_work)
    CC.addToDB(mov_work)
    
    # Step 2: ICP Registration
    log_section("Step 2: ICP Alignment")
    R, t, rms_hist = compute_icp(
        mov_work, ref_work,
        max_iter=Config.ICP_MAX_ITERATIONS,
        sample_limit=Config.ICP_SAMPLE_LIMIT,
        trim_ratio=Config.ICP_TRIM_RATIO
    )
    
    if R is not None:
        apply_transform(mov_work, R, t)
        mov_work.setName("Moving_Aligned_2020")
    
    CC.updateUI()
    
    # Step 3: M3C2 Distance (demonstrates limitation)
    log_section("Step 3: M3C2 Distance")
    compute_m3c2(ref_work, mov_work)
    
    # Save aligned cloud for CLI M3C2 processing
    log_section("Export Aligned Cloud")
    save_cloud(mov_work, Config.OUTPUT_FILE)
    
    # Summary
    log_section("Pipeline Summary")
    log(f"Reference (T1): {Config.T1_PATH.name}")
    log(f"Moving (T2): {Config.T2_PATH.name}")
    log(f"ICP iterations: {len(rms_hist)}")
    if rms_hist:
        log(f"Final RMS: {rms_hist[-1]:.6f}")
    log(f"Aligned cloud saved: {Config.OUTPUT_FILE}")
    log("")
    log("NEXT STEPS for M3C2:")
    log("  Option 1: Use GUI menu Tools -> Distances -> M3C2")
    log("  Option 2: Run CLI: CloudCompare -O ref.laz -O aligned.laz -M3C2 params.txt")
    log("  Option 3: Use our implementation: python -m terrain_change_detection")
    
    return True


if __name__ == "__main__":
    run_pipeline()
