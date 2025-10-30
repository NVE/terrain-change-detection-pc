"""
CloudCompare PythonRuntime pipeline for synthetic pair (2015 vs 2020).

This script is designed to be run inside CloudCompare's PythonRuntime plugin
(https://tmontaigu.github.io/CloudCompare-PythonRuntime/), not via this repo's
own Python code. It loads the two synthetic LAZ files in data/synthetic/, runs
basic preprocessing, attempts ICP alignment (if available), and computes a
cloud-to-cloud (C2C) distance scalar field as a baseline change metric.

How to run (GUI):
- Open CloudCompare with the Python plugin enabled
- Open the Python Editor or File Runner (Plugins -> Python Runtime)
- Run this script

Notes:
- Uses only pycc/cccorelib APIs; no imports from this repository.
- Attempts to call cccorelib.RegistrationTools ICP if exposed; otherwise falls
  back to a simple centroid pre-alignment and computes distances.
- C2C is implemented via KD-tree fallback if DistanceComputationTools bindings
  are not available.
- Results: the 2020 cloud is duplicated as an "aligned" cloud with a scalar
  field named "C2C Absolute Dist" for easy visualization and optional saving.

If your repo path differs from the detected base, set REPO_BASE manually below.
"""

from __future__ import annotations

import os
from pathlib import Path
import math
from typing import Optional, Tuple

try:
    import pycc
    import cccorelib
except Exception as e:  # pragma: no cover - only runs inside CloudCompare
    raise RuntimeError(
        "This script must be run inside CloudCompare with the PythonRuntime plugin.\n"
        f"Import error: {e}"
    )


# --- Configuration ---------------------------------------------------------

# Attempt to infer the repo root from this script's location
_HERE = Path(__file__).resolve()
DEFAULT_REPO_BASE = _HERE.parent.parent  # repo root (.. from scripts/)

# Override this if your script is not inside the repo directory when run
REPO_BASE = Path(os.environ.get("TCD_REPO_BASE", DEFAULT_REPO_BASE))

T1_PATH = REPO_BASE / "data" / "synthetic" / "synthetic_area" / "2015" / "data" / "synthetic_tile_01.laz"
T2_PATH = REPO_BASE / "data" / "synthetic" / "synthetic_area" / "2020" / "data" / "synthetic_tile_01.laz"

# Subsampling target count to speed up ICP (applied on duplicates for ICP only)
ICP_RANDOM_LIMIT = 60000  # limit subset size used for ICP fallback

# Distance computation settings
KD_CHUNK_SIZE = 20000  # process points in chunks to remain responsive
MAX_EXPORT = True  # set to True to save the aligned cloud with distances

# Clean workspace option: remove all objects from the DB tree before running
def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

CLEAR_DB_AT_START = _env_flag("CC_CLEAR_DB", True)


# --- Helpers ---------------------------------------------------------------

def _log(msg: str) -> None:
    try:
        pycc.ccLog.Print(msg)
    except Exception:
        print(msg)


def _load_cloud(path: Path) -> pycc.ccPointCloud:
    CC = pycc.GetInstance()
    params = pycc.FileIOFilter.LoadParameters()
    params.parentWidget = CC.getMainWindow()

    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")

    obj = CC.loadFile(str(path), params)
    # The loadFile call adds to DB on success and returns the root object loaded.
    # Search for the first ccPointCloud in the returned hierarchy.
    cloud = _first_point_cloud_from(obj)
    if cloud is None:
        raise RuntimeError(f"No point cloud found in {path}")

    _log(f"Loaded: {path.name} -> {cloud.size()} points")
    return cloud


def _first_point_cloud_from(hobj: pycc.ccHObject) -> Optional[pycc.ccPointCloud]:
    # Walk hierarchy to find first ccPointCloud using only widely available APIs
    try:
        if isinstance(hobj, pycc.ccPointCloud):
            return hobj
    except Exception:
        pass

    try:
        n = hobj.getChildrenNumber()
    except Exception:
        return None

    for i in range(n):
        try:
            child = hobj.getChild(i)
        except Exception:
            continue
        if child is None:
            continue
        if isinstance(child, pycc.ccPointCloud):
            return child
        found = _first_point_cloud_from(child)
        if found is not None:
            return found
    return None


def _duplicate_cloud(src: pycc.ccPointCloud, name: str) -> pycc.ccPointCloud:
    dup = pycc.ccPointCloud(name)
    dup.reserve(src.size())
    for i in range(src.size()):
        dup.addPoint(src.getPoint(i))
    # Copy point size if the API exposes it; otherwise ignore
    try:
        ps = getattr(src, "getPointSize", None)
        if callable(ps):
            dup.setPointSize(ps())
    except Exception:
        pass
    return dup


def _deselect_all_in_db(CC: "pycc.ccPythonInstance") -> None:
    root = CC.dbRootObject()
    def walk(h: pycc.ccHObject):
        try:
            CC.setSelectedInDB(h, False)
        except Exception:
            pass
        try:
            n = h.getChildrenNumber()
        except Exception:
            n = 0
        for i in range(n):
            try:
                child = h.getChild(i)
            except Exception:
                child = None
            if child is not None:
                walk(child)
    walk(root)


def _set_selection(CC: "pycc.ccPythonInstance", objs: list) -> None:
    _deselect_all_in_db(CC)
    for o in objs:
        try:
            CC.setSelectedInDB(o, True)
        except Exception:
            pass
    CC.updateUI()


def _clear_db_tree(CC: "pycc.ccPythonInstance") -> None:
    """Remove all top-level objects from the DB tree."""
    root = CC.dbRootObject()
    try:
        n = root.getChildrenNumber()
    except Exception:
        n = 0
    to_remove = []
    for i in range(n):
        try:
            child = root.getChild(i)
            if child is not None:
                to_remove.append(child)
        except Exception:
            continue
    # Remove in reverse order for safety
    for obj in reversed(to_remove):
        try:
            CC.removeFromDB(obj)
        except Exception:
            pass
    CC.updateUI()


def _as_numpy(cloud: pycc.ccPointCloud, max_points: Optional[int] = None):
    # Extract points into a Python list of tuples (or fewer if max_points)
    # Kept simple to reduce API assumptions; sufficient for small ICP subsets.
    n = cloud.size()
    if max_points is not None:
        n = min(n, max_points)
    pts = []
    step = max(1, cloud.size() // n) if n > 0 else 1
    idx = 0
    for _ in range(n):
        p = cloud.getPoint(idx)
        # CCVector3 supports array-like or x,y,z attributes depending on binding
        try:
            x, y, z = float(p.x), float(p.y), float(p.z)
        except Exception:
            # Some bindings expose vector as indexable sequence
            x, y, z = float(p[0]), float(p[1]), float(p[2])
        pts.append((x, y, z))
        idx += step
    try:
        import numpy as np  # CloudCompare often ships with a numpy
    except Exception as e:
        raise RuntimeError(f"numpy is required inside CC for ICP fallback: {e}")
    return np.asarray(pts, dtype=float)


def _get_xyz(v) -> Tuple[float, float, float]:
    try:
        return float(v.x), float(v.y), float(v.z)
    except Exception:
        return float(v[0]), float(v[1]), float(v[2])


def _centroid_align(src_pts, dst_pts) -> Tuple["np.ndarray", "np.ndarray"]:
    import numpy as np
    c_src = src_pts.mean(axis=0)
    c_dst = dst_pts.mean(axis=0)
    R = np.eye(3)
    t = c_dst - c_src
    return R, t


def _best_fit_transform(A, B):
    """Compute rigid transform (R,t) that best aligns A->B (point-to-point, SVD)."""
    import numpy as np
    assert A.shape == B.shape
    cA = A.mean(axis=0)
    cB = B.mean(axis=0)
    AA = A - cA
    BB = B - cB
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Reflection fix
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cB - R @ cA
    return R, t


def _apply_rigid_transform_inplace(cloud: pycc.ccPointCloud, R, t) -> None:
    # Build a 4x4 ccGLMatrix(d) and apply permanently to the cloud
    import numpy as np

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t

    # Try double precision matrix first
    glmat = None
    try:
        glmat = pycc.ccGLMatrixd()
        arr = glmat.data()  # may expose a view to the 4x4 matrix
        # Copy values; fallback if data() not writable
        for r in range(4):
            for c in range(4):
                arr[r][c] = float(M[r, c])
    except Exception:
        try:
            glmat = pycc.ccGLMatrix()
            arr = glmat.data()
            for r in range(4):
                for c in range(4):
                    arr[r][c] = float(M[r, c])
        except Exception:
            glmat = None

    # Apply permanent transformation
    applied = False
    if glmat is not None:
        for applier in ("applyRigidTransformation", "applyGLTransformation_recursive"):
            fn = getattr(cloud, applier, None)
            if callable(fn):
                try:
                    fn(glmat)
                    applied = True
                    break
                except Exception:
                    pass

    if not applied:
        # Manual per-point transform as worst-case fallback
        for i in range(cloud.size()):
            p = cloud.getPoint(i)
            x, y, z = _get_xyz(p)
            v = np.array([x, y, z])
            v2 = R @ v + t
            cloud.setPoint(i, cccorelib.CCVector3(float(v2[0]), float(v2[1]), float(v2[2])))


def _icp_fallback_kdtree(
    mov_icp: pycc.ccPointCloud,
    ref_icp: pycc.ccPointCloud,
    max_iters: int = 60,
    trim_ratio: float = 0.8,
    max_pairing_dist: Optional[float] = None,
) -> Tuple["np.ndarray", "np.ndarray", list]:
    """Robust ICP (point-to-point) using cccorelib.KDTree + SVD.

    - trim_ratio: keep best fraction of pairs per-iteration (outlier rejection)
    - max_pairing_dist: optional hard cap (units of the clouds)

    Returns (R_total, t_total, rms_history)
    """
    import numpy as np

    # KD-tree on reference
    kd = cccorelib.KDTree()
    kd.buildFromCloud(ref_icp)

    # Helpers to extract arrays from ccPointCloud efficiently
    def cloud_to_np(c: pycc.ccPointCloud) -> "np.ndarray":
        pts = np.zeros((c.size(), 3), dtype=float)
        for i in range(c.size()):
            x, y, z = _get_xyz(c.getPoint(i))
            pts[i, :] = (x, y, z)
        return pts

    A = cloud_to_np(mov_icp)  # moving
    B = cloud_to_np(ref_icp)  # reference

    R_total = np.eye(3)
    t_total = np.zeros(3)
    rms_hist = []

    # AABB-based default cap (5% of diagonal) if not provided
    if max_pairing_dist is None:
        try:
            bbmin = cccorelib.CCVector3()
            bbmax = cccorelib.CCVector3()
            ref_icp.getBoundingBox(bbmin, bbmax)
            diag = math.sqrt((bbmax.x - bbmin.x) ** 2 + (bbmax.y - bbmin.y) ** 2 + (bbmax.z - bbmin.z) ** 2)
            max_pairing_dist = 0.05 * diag
        except Exception:
            max_pairing_dist = None

    for it in range(max_iters):
        # Find nearest neighbors of transformed A in B
        pairs_A = []  # moving
        pairs_B = []  # reference
        d_vals = []
        for i in range(A.shape[0]):
            a = A[i, :]
            a_vec = cccorelib.CCVector3(float(a[0]), float(a[1]), float(a[2]))
            nn = kd.findNearestNeighbour(a_vec)
            idx, d2 = None, None
            try:
                idx, d2 = int(nn[0]), float(nn[1])
            except Exception:
                # Some wrappers might return only the index; recompute distance
                try:
                    idx = int(nn)
                    pb = B[idx]
                    d2 = float(((a - pb) ** 2).sum())
                except Exception:
                    continue
            d = math.sqrt(d2)
            if max_pairing_dist is not None and d > max_pairing_dist:
                continue
            pairs_A.append(a)
            pairs_B.append(B[idx])
            d_vals.append(d)

        if not pairs_A:
            _log("[ICP-fallback] No valid correspondences; stopping.")
            break

        # Trim worst matches
        order = np.argsort(d_vals)
        keep = max(10, int(len(order) * float(trim_ratio)))
        idx_keep = order[:keep]
        A_corr = np.asarray(pairs_A)[idx_keep]
        B_corr = np.asarray(pairs_B)[idx_keep]
        R, t = _best_fit_transform(A_corr, B_corr)

        # Accumulate
        R_total = R @ R_total
        t_total = R @ t_total + t

        # Apply to A for next iteration
        A = (A @ R.T) + t

        import numpy as np
        d_used = np.asarray(d_vals)[idx_keep]
        rms = float(np.sqrt(np.mean(d_used ** 2)))
        rms_hist.append(rms)
        if (it + 1) % 5 == 0 or it < 3:
            _log(f"[ICP-fallback] iter {it+1:02d} RMS={rms:.6f} pairs={len(idx_keep)}")

        # Early stop if tiny improvement
        if len(rms_hist) > 2 and abs(rms_hist[-2] - rms_hist[-1]) < 1e-7:
            break

    return R_total, t_total, rms_hist


def _icp_align_if_possible(src: pycc.ccPointCloud, dst: pycc.ccPointCloud) -> Tuple[bool, Optional[Tuple]]:
    """Try to use cccorelib.RegistrationTools ICP. Returns (ok, extra_info)."""
    # Bindings may differ across builds; attempt a few signatures.
    # If not available, return False so caller can do a fallback.
    try:
        RT = cccorelib.RegistrationTools
    except Exception:
        return False, None

    # Some builds might expose a convenience ICP; try calling defensively
    for candidate in ("ICP", "RegisterClouds", "PerformRegistration"):
        fn = getattr(RT, candidate, None)
        if not callable(fn):
            continue
        try:
            # Minimalistic call: let defaults drive most parameters
            result = fn(src, dst)
            # Result could be a matrix, a tuple, or an object with members
            return True, (candidate, result)
        except TypeError:
            # Try a signature with maxIterations / minError
            try:
                result = fn(src, dst, 50, 1e-6)
                return True, (candidate, result)
            except Exception:
                pass
        except Exception:
            pass
    return False, None


def _compute_c2c_with_kdtree(moving: pycc.ccPointCloud, reference: pycc.ccPointCloud, sf_name: str = "C2C Absolute Dist") -> str:
    """Assigns per-point nearest neighbor distance from reference to moving as a scalar field."""
    # Add scalar field
    sf_idx = moving.getScalarFieldIndexByName(sf_name)
    if sf_idx < 0:
        sf_idx = moving.addScalarField(sf_name)
    if sf_idx < 0:
        raise RuntimeError("Failed to create scalar field for distances")
    sf = moving.getScalarField(sf_idx)

    # Build KD-tree on reference
    kd = None
    try:
        kd = cccorelib.KDTree()
        kd.buildFromCloud(reference)
    except Exception:
        kd = None

    # Fallback if KDTree wrapper not available: brute-force is too slow; abort
    if kd is None:
        raise RuntimeError("KDTree not available; cannot compute C2C distances here.")

    # Iterate in chunks
    n = moving.size()
    processed = 0
    while processed < n:
        end = min(n, processed + KD_CHUNK_SIZE)
        for i in range(processed, end):
            p = moving.getPoint(i)
            # Expect KDTree.findNearestNeighbour to accept CCVector3 and return (idx, dist2) or dist2
            try:
                nn = kd.findNearestNeighbour(p)
            except Exception:
                nn = None

            dist2 = None
            if nn is None:
                dist2 = math.nan
            else:
                # nn may be a tuple (index, dist2) or just dist2 depending on wrapper
                try:
                    # typical C++ returns index and squared dist
                    _, dist2 = int(nn[0]), float(nn[1])
                except Exception:
                    try:
                        dist2 = float(nn)
                    except Exception:
                        dist2 = math.nan

            dist = math.sqrt(dist2) if dist2 is not None and not math.isnan(dist2) else math.nan
            sf.setValue(i, dist)

        processed = end

    sf.computeMinAndMax()
    moving.setCurrentDisplayedScalarField(sf_idx)
    moving.showSF(True)
    return sf_name


def _approx_nn_rms(moving: pycc.ccPointCloud, reference: pycc.ccPointCloud, sample: int = 10000) -> float:
    """Quick approximate NN RMS distance between two clouds using KD-tree and at most 'sample' points."""
    try:
        kd = cccorelib.KDTree()
        kd.buildFromCloud(reference)
    except Exception:
        return float("nan")

    n = moving.size()
    if n == 0:
        return float("nan")
    step = max(1, n // min(n, sample))
    d2s = []
    for i in range(0, n, step):
        p = moving.getPoint(i)
        try:
            nn = kd.findNearestNeighbour(p)
            try:
                _, d2 = int(nn[0]), float(nn[1])
            except Exception:
                # Index-only
                idx = int(nn)
                q = reference.getPoint(idx)
                x1, y1, z1 = _get_xyz(p)
                x2, y2, z2 = _get_xyz(q)
                d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
            d2s.append(d2)
        except Exception:
            continue
    if not d2s:
        return float("nan")
    import numpy as np
    return float(np.sqrt(np.mean(np.asarray(d2s))))


def _try_distance_tools(moving: pycc.ccPointCloud, reference: pycc.ccPointCloud) -> bool:
    """Prefer native DistanceComputationTools if available; else return False."""
    try:
        DT = cccorelib.DistanceComputationTools
    except Exception:
        return False

    # Try some likely function names
    for fname in (
        "computeCloud2CloudDistance",
        "computeApproxCloud2CloudDistance",
        "ComputeCloud2CloudDistance",
    ):
        fn = getattr(DT, fname, None)
        if not callable(fn):
            continue
        try:
            ok = fn(moving, reference)
            if isinstance(ok, bool) and ok:
                # Active SF should already be set by CC, but enforce
                idx = moving.getCurrentDisplayedScalarFieldIndex()
                if idx >= 0:
                    moving.showSF(True)
                return True
        except Exception:
            continue
    return False


def _run_m3c2_if_available(reference: pycc.ccPointCloud, compared: pycc.ccPointCloud):
    """Run qM3C2 if the plugin is available. Returns (ok, result_cloud_or_None)."""
    try:
        from pycc.plugins import qM3C2 as qm3c2  # type: ignore
    except Exception:
        return False, None

    CC = pycc.GetInstance()
    # Ensure the 2 clouds are selected as the plugin relies on selection
    _set_selection(CC, [reference, compared])

    try:
        # Constructor expects the two clouds explicitly (ref, compared)
        dlg = qm3c2.qM3C2Dialog(reference, compared)
    except Exception as e:
        _log(f"[M3C2] Failed to create dialog: {e}")
        return False, None

    # Load saved/default params if possible (non-fatal)
    for try_load in ("loadParamsFromPersistentSettings", "loadParamsFromFile"):
        fn = getattr(dlg, try_load, None)
        if callable(fn):
            try:
                if try_load == "loadParamsFromFile":
                    # If you have a param file, set its path here
                    pass
                else:
                    fn()
            except Exception:
                pass

    _log("[M3C2] Computing distances (plugin) ...")
    try:
        # allowsDialog=False to run headless; the plugin uses the selection
        result = qm3c2.qM3C2Process.Compute(dlg, False)
    except Exception as e:
        _log(f"[M3C2] Compute failed: {e}")
        return False, None

    if result is None:
        _log("[M3C2] No result returned.")
        return False, None

    try:
        result.setName("M3C2_Result")
    except Exception:
        pass

    try:
        # Some builds may already add to DB, still safe to add
        CC.addToDB(result)
        idx = result.getCurrentDisplayedScalarFieldIndex()
        if idx < 0 and result.getNumberOfScalarFields() > 0:
            result.setCurrentDisplayedScalarField(0)
        result.showSF(True)
        CC.updateUI()
    except Exception:
        pass

    _log("[M3C2] Done.")
    return True, result


def main():  # pragma: no cover - executed inside CloudCompare
    CC = pycc.GetInstance()
    CC.disableAll()
    try:
        if CLEAR_DB_AT_START:
            _log("[Init] Clearing DB tree (CC_CLEAR_DB=on) ...")
            try:
                CC.freezeUI(True)
            except Exception:
                pass
            _clear_db_tree(CC)
            try:
                CC.freezeUI(False)
            except Exception:
                pass
            CC.updateUI()
            _log("[Init] Workspace cleared.")
        _log("[Pipeline] Loading synthetic clouds ...")
        ref_cloud = _load_cloud(T1_PATH)
        mov_cloud = _load_cloud(T2_PATH)

        # Create working copies for alignment and metrics so originals stay intact
        _log("[Pipeline] Duplicating clouds for processing ...")
        ref = _duplicate_cloud(ref_cloud, "2015_ref")
        mov = _duplicate_cloud(mov_cloud, "2020_mov")
        CC.addToDB(ref)
        CC.addToDB(mov)

        # Optional lightweight subsampling for ICP attempt
        _log("[Pipeline] Preparing ICP inputs (subsampling for speed) ...")
        try:
            # Random-limit-like subset for quick alignment estimation
            target_n = min(ICP_RANDOM_LIMIT, ref.size(), mov.size())
            if target_n > 1000:  # only bother if sufficiently large
                ref_subset_idx = cccorelib.CloudSamplingTools.subsampleCloudRandomly(ref, target_n)
                mov_subset_idx = cccorelib.CloudSamplingTools.subsampleCloudRandomly(mov, target_n)
                ref_icp = ref.partialClone(ref_subset_idx)
                mov_icp = mov.partialClone(mov_subset_idx)
            else:
                ref_icp = ref
                mov_icp = mov
        except Exception:
            ref_icp = ref
            mov_icp = mov

        # Try ICP via bindings
        # Quick metric before alignment
        pre_rms = _approx_nn_rms(mov, ref)
        _log(f"[Pipeline] Approx NN RMS before alignment: {pre_rms:.6f}")

        _log("[Pipeline] Attempting ICP alignment via cccorelib.RegistrationTools ...")
        icp_ok, icp_info = _icp_align_if_possible(mov_icp, ref_icp)
        if icp_ok:
            _log(f"[ICP] Succeeded via {icp_info[0]}; updating full cloud with estimated transform ...")
            # Best effort: if result includes a transformation matrix, apply to full moving cloud
            candidate, result = icp_info
            R, t = None, None
            # Try to extract transform from common patterns
            try:
                # If result has a 4x4 matrix-like data() or attributes
                mat = getattr(result, "transformation", None)
                if mat is None:
                    mat = getattr(result, "T", None)
                if mat is None:
                    mat = result
                import numpy as np
                M = np.eye(4)
                # matrix may support indexing
                for r in range(4):
                    for c in range(4):
                        M[r, c] = float(mat[r][c])
                R = M[:3, :3]
                t = M[:3, 3]
            except Exception:
                # Fall back to estimating transform between the two ICP subsets
                try:
                    import numpy as np
                    A = _as_numpy(mov_icp, max_points=5000)
                    B = _as_numpy(ref_icp, max_points=5000)
                    R, t = _best_fit_transform(A, B)
                except Exception:
                    R = t = None

            if R is not None and t is not None:
                _apply_rigid_transform_inplace(mov, R, t)
            else:
                _log("[ICP] Could not extract transform; skipping transform application.")
        else:
            _log("[ICP] RegistrationTools not available; running robust ICP fallback (KD-tree) ...")
            try:
                R, t, hist = _icp_fallback_kdtree(mov_icp, ref_icp, max_iters=60, trim_ratio=0.8, max_pairing_dist=None)
                if hist:
                    _log(f"[ICP-fallback] Done. Final RMS={hist[-1]:.6f} iters={len(hist)}")
                _apply_rigid_transform_inplace(mov, R, t)
                mov.setName("2020_mov_aligned")
            except Exception as e:
                _log(f"[ICP-fallback] Failed: {e}. Applying simple centroid pre-alignment ...")
                try:
                    import numpy as np
                    A = _as_numpy(mov, max_points=5000)
                    B = _as_numpy(ref, max_points=5000)
                    R, t = _centroid_align(A, B)
                    _apply_rigid_transform_inplace(mov, R, t)
                    mov.setName("2020_mov_centroid_aligned")
                except Exception as e2:
                    _log(f"[ICP] Fallback centroid align failed: {e2}")

        post_rms = _approx_nn_rms(mov, ref)
        _log(f"[Pipeline] Approx NN RMS after alignment: {post_rms:.6f}")

        # Compute C2C distances (prefer native tool, else KD-tree fallback)
        _log("[Pipeline] Computing C2C distances ...")
        dist_ok = _try_distance_tools(mov, ref)
        if dist_ok:
            _log("[Distances] Computed via DistanceComputationTools.")
        else:
            _log("[Distances] Using KD-tree fallback for C2C distances ...")
            name = _compute_c2c_with_kdtree(mov, ref)
            _log(f"[Distances] Scalar field '{name}' assigned on moving cloud.")

        # Update UI and optionally save result
        CC.updateUI()

        aligned_for_m3c2 = mov  # default to in-memory aligned cloud
        if MAX_EXPORT:
            out_dir = REPO_BASE / "data" / "synthetic" / "synthetic_area" / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "2020_aligned_with_c2c.laz"
            _log(f"[Export] Saving aligned moving cloud with distances to: {out_path}")
            sp = pycc.FileIOFilter.SaveParameters()
            sp.parentWidget = CC.getMainWindow()
            ok = pycc.FileIOFilter.SaveToFile(mov, str(out_path), sp)
            # Some bindings return None even on success; rely on file existence
            if ok is False and not out_path.exists():
                _log("[Export] SaveToFile reported failure and file not found.")
            # Load the just-saved file back into the GUI so it is visible
            if out_path.exists():
                try:
                    _log("[Export] Loading saved file into the GUI ...")
                    lp = pycc.FileIOFilter.LoadParameters()
                    lp.parentWidget = CC.getMainWindow()
                    obj = CC.loadFile(str(out_path), lp)
                    oc = _first_point_cloud_from(obj)
                    if oc is not None:
                        oc.setName("2020_aligned_with_c2c (saved)")
                        aligned_for_m3c2 = oc
                    CC.updateUI()
                except Exception as e:
                    _log(f"[Export] Reload failed (non-fatal): {e}")

        # Optional M3C2 (plugin) – run on reference + aligned (prefer the reloaded file)
        ok_m3c2, res = _run_m3c2_if_available(ref, aligned_for_m3c2)
        if ok_m3c2:
            _log("[Done] Pipeline + M3C2 completed.")
        else:
            _log("[Done] Pipeline completed (M3C2 not available or failed).")
    finally:
        CC.enableAll()
        CC.updateUI()


if __name__ == "__main__":  # pragma: no cover
    main()
