"""
CloudCompare Python Plugin Pipeline Demonstration.

This script demonstrates using CloudCompare's Python Plugin (PythonRuntime)
for terrain change detection workflows. Based on the official API documentation:
https://tmontaigu.github.io/CloudCompare-PythonRuntime/

WHAT THIS SCRIPT DEMONSTRATES
-----------------------------
1. Loading multi-temporal point clouds
2. Point cloud subsampling using CloudSamplingTools
3. M3C2 distance computation (if qM3C2 plugin is available)
4. Manual transformation application via ccGLMatrix

API LIMITATIONS DISCOVERED
--------------------------
- ICP: ICPRegistrationTools is NOT exposed in the Python API. Only RegistrationTools
  with transformation filters is available. For ICP, use CLI or GUI.
- C2C: DistanceComputationTools.computeCloud2CloudDistances is NOT exposed.
  Only error measure constants are available.
- M3C2: Available via pycc.plugins.qM3C2 IF compiled with PLUGIN_STANDARD_QM3C2=ON

How to Run
----------
1. Open CloudCompare
2. Go to Plugins -> Python Plugin -> File Runner (or Python Editor)
3. Load this script and execute

References
----------
- API Docs: https://tmontaigu.github.io/CloudCompare-PythonRuntime/
- Examples: https://tmontaigu.github.io/CloudCompare-PythonRuntime/examples.html
- cccorelib: https://tmontaigu.github.io/CloudCompare-PythonRuntime/python/cccorelib/index.html
- pycc: https://tmontaigu.github.io/CloudCompare-PythonRuntime/python/pycc/index.html
"""

from __future__ import annotations
from pathlib import Path
import math

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

# Check if qM3C2 plugin is available
try:
    from pycc.plugins import qM3C2
    HAS_QM3C2 = True
except ImportError:
    HAS_QM3C2 = False


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Configuration parameters for the pipeline."""
    
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent
    
    # Input paths - update these to match your data
    T1_PATH = REPO_ROOT / "data" / "synthetic" / "synthetic_area" / "2015" / "data" / "synthetic_tile_01.laz"
    T2_PATH = REPO_ROOT / "data" / "synthetic" / "synthetic_area" / "2020" / "data" / "synthetic_tile_01.laz"
    
    # Output directory
    OUTPUT_DIR = REPO_ROOT / "data" / "synthetic" / "synthetic_area" / "outputs"
    
    # Subsampling parameters
    SUBSAMPLE_COUNT = 100000  # Number of points for subsampling
    
    # M3C2 parameters file (optional)
    M3C2_PARAMS_FILE = None  # Set to path if you have a params file
    
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
    """Log to CloudCompare console only (no print to avoid duplicates)."""
    formatted = f"[{level}] {msg}"
    try:
        if level == "ERROR":
            pycc.ccLog.Error(msg)
        elif level == "WARNING":
            pycc.ccLog.Warning(msg)
        else:
            pycc.ccLog.Print(formatted)
    except Exception:
        # Fallback to print only if ccLog fails
        print(formatted)


def log_section(title):
    sep = "=" * 60
    log(sep)
    log(f"  {title}")
    log(sep)


# ==============================================================================
# POINT CLOUD LOADING
# ==============================================================================

def find_first_cloud(hobj):
    """Recursively find the first point cloud in a hierarchy object."""
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
    """Load a point cloud file into CloudCompare."""
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


# ==============================================================================
# POINT CLOUD SUBSAMPLING
# ==============================================================================

def subsample_cloud(cloud, target_count):
    """
    Subsample a point cloud using CloudSamplingTools.
    
    Uses: cccorelib.CloudSamplingTools.subsampleCloudRandomly()
    
    Parameters
    ----------
    cloud : ccPointCloud
        The cloud to subsample
    target_count : int
        Target number of points
    
    Returns
    -------
    ccPointCloud
        Subsampled cloud (reference to original if no subsampling needed)
    """
    if cloud.size() <= target_count:
        log(f"  Cloud already has {cloud.size():,} points (target: {target_count:,})")
        return cloud
    
    log(f"  Subsampling from {cloud.size():,} to ~{target_count:,} points...")
    
    try:
        # Use CloudSamplingTools.subsampleCloudRandomly
        ref_cloud = cccorelib.CloudSamplingTools.subsampleCloudRandomly(
            cloud, 
            target_count
        )
        
        if ref_cloud is None:
            log("  Subsampling returned None, using original cloud", level="WARNING")
            return cloud
        
        # Create a proper ccPointCloud from the reference
        subsampled = cloud.partialClone(ref_cloud)
        if subsampled is not None:
            subsampled.setName(f"{cloud.getName()}_subsampled")
            log(f"  Result: {subsampled.size():,} points")
            return subsampled
        else:
            log("  Could not create subsampled cloud", level="WARNING")
            return cloud
            
    except Exception as e:
        log(f"  Subsampling failed: {e}", level="WARNING")
        return cloud


# ==============================================================================
# ALIGNMENT USING GRAVITY CENTER
# ==============================================================================

def align_clouds_by_centroid(reference_cloud, moving_cloud):
    """
    Align clouds by matching their gravity centers (centroids).
    
    This is a simple alignment method using GeometricalAnalysisTools.ComputeGravityCenter()
    which IS exposed in the Python API. For more precise alignment (ICP), use GUI or CLI.
    
    This is useful when:
    - Clouds are roughly aligned but have a translation offset
    - As a coarse alignment before fine registration
    
    Parameters
    ----------
    reference_cloud : ccPointCloud
        The reference cloud (target)
    moving_cloud : ccPointCloud
        The cloud to be aligned (will be modified in place)
    
    Returns
    -------
    tuple
        (success: bool, translation: CCVector3 or None)
    """
    log_section("Centroid-Based Alignment")
    log(f"Reference: {reference_cloud.getName()} ({reference_cloud.size():,} points)")
    log(f"Moving: {moving_cloud.getName()} ({moving_cloud.size():,} points)")
    
    try:
        # Compute gravity centers
        ref_center = cccorelib.GeometricalAnalysisTools.ComputeGravityCenter(reference_cloud)
        mov_center = cccorelib.GeometricalAnalysisTools.ComputeGravityCenter(moving_cloud)
        
        log(f"  Reference center: ({ref_center.x:.2f}, {ref_center.y:.2f}, {ref_center.z:.2f})")
        log(f"  Moving center: ({mov_center.x:.2f}, {mov_center.y:.2f}, {mov_center.z:.2f})")
        
        # Compute translation needed
        tx = ref_center.x - mov_center.x
        ty = ref_center.y - mov_center.y
        tz = ref_center.z - mov_center.z
        
        distance = (tx**2 + ty**2 + tz**2)**0.5
        log(f"  Translation needed: ({tx:.4f}, {ty:.4f}, {tz:.4f})")
        log(f"  Distance: {distance:.4f}")
        
        if distance < 0.001:
            log("  Clouds are already aligned (distance < 0.001)")
            return True, None
        
        # Apply translation using ccGLMatrix
        glMat = moving_cloud.getGLTransformation()
        translation = glMat.getTranslationAsVec3D()
        translation.x += tx
        translation.y += ty
        translation.z += tz
        glMat.setTranslation(translation)
        moving_cloud.setGLTransformation(glMat)
        moving_cloud.applyGLTransformation_recursive()
        
        log("  Centroid alignment applied successfully!")
        log("")
        log("NOTE: This is a coarse alignment (translation only).")
        log("For precise alignment with rotation, use:")
        log("  1. GUI: Tools -> Registration -> Fine Registration (ICP)")
        log("  2. CLI: CloudCompare -O ref.laz -O mov.laz -ICP")
        
        return True, cccorelib.CCVector3(tx, ty, tz)
        
    except Exception as e:
        log(f"Centroid alignment failed: {e}", level="ERROR")
        return False, None


# ==============================================================================
# M3C2 DISTANCE COMPUTATION
# ==============================================================================

def compute_m3c2(cloud1, cloud2, params_file=None):
    """
    Compute M3C2 distances using CloudCompare's qM3C2 plugin.
    
    This uses pycc.plugins.qM3C2 which is available when CloudCompare
    is compiled with PLUGIN_STANDARD_QM3C2=ON.
    
    Parameters
    ----------
    cloud1 : ccPointCloud
        First point cloud (reference)
    cloud2 : ccPointCloud
        Second point cloud (compared)
    params_file : str, optional
        Path to M3C2 parameters file
    
    Returns
    -------
    ccPointCloud or None
        Result cloud with M3C2 distances, or None if failed
    """
    log_section("M3C2 Distance Computation")
    
    if not HAS_QM3C2:
        log("qM3C2 plugin is NOT available in this CloudCompare build.", level="WARNING")
        log("")
        log("The qM3C2 plugin requires:")
        log("  - CloudCompare compiled with PLUGIN_STANDARD_QM3C2=ON")
        log("  - The plugin DLL/SO to be present")
        log("")
        log("Alternative options for M3C2:")
        log("  1. Use CloudCompare GUI: Tools -> Distances -> M3C2")
        log("  2. Use CloudCompare CLI: CloudCompare -O ref.laz -O mov.laz -M3C2 params.txt")
        log("  3. Use our custom Python implementation")
        return None
    
    log(f"Cloud 1: {cloud1.getName()} ({cloud1.size():,} points)")
    log(f"Cloud 2: {cloud2.getName()} ({cloud2.size():,} points)")
    
    try:
        # Create M3C2 dialog to configure parameters
        dialog = qM3C2.qM3C2Dialog(cloud1, cloud2)
        
        # Load parameters from file if provided
        if params_file and Path(params_file).exists():
            log(f"Loading M3C2 params from: {params_file}")
            dialog.loadParamsFromFile(str(params_file))
        else:
            # Use persistent settings or defaults
            dialog.loadParamsFromPersistentSettings()
        
        # Run M3C2 computation
        # Compute(dialog, allowsDialog) - set allowsDialog=False for non-interactive
        log("Running M3C2 computation...")
        result = qM3C2.qM3C2Process.Compute(dialog, False)
        
        if result is not None:
            log(f"M3C2 completed! Result cloud: {result.getName()}")
            log(f"  Points: {result.size():,}")
            
            # Check for M3C2 scalar fields
            sf_count = result.getNumberOfScalarFields()
            if sf_count > 0:
                log(f"  Scalar fields: {sf_count}")
                for i in range(sf_count):
                    sf = result.getScalarField(i)
                    log(f"    - {sf.getName()}: [{sf.getMin():.4f}, {sf.getMax():.4f}]")
            
            return result
        else:
            log("M3C2 computation returned None", level="WARNING")
            return None
            
    except Exception as e:
        log(f"M3C2 computation failed: {e}", level="ERROR")
        return None


# ==============================================================================
# TRANSFORMATION UTILITIES
# ==============================================================================

def apply_translation(cloud, tx, ty, tz):
    """
    Apply a translation to a point cloud using ccGLMatrix.
    
    Based on official example:
    https://tmontaigu.github.io/CloudCompare-PythonRuntime/examples.html#translation
    
    Parameters
    ----------
    cloud : ccPointCloud
        The cloud to transform
    tx, ty, tz : float
        Translation in X, Y, Z
    """
    log(f"Applying translation ({tx}, {ty}, {tz}) to '{cloud.getName()}'...")
    
    glMat = cloud.getGLTransformation()
    translation = glMat.getTranslationAsVec3D()
    translation.x += tx
    translation.y += ty
    translation.z += tz
    glMat.setTranslation(translation)
    cloud.setGLTransformation(glMat)
    cloud.applyGLTransformation_recursive()
    
    log("  Translation applied")


def apply_rotation(cloud, angle_degrees, axis='z'):
    """
    Apply a rotation to a point cloud around its center.
    
    Based on official example:
    https://tmontaigu.github.io/CloudCompare-PythonRuntime/examples.html#rotation
    
    Parameters
    ----------
    cloud : ccPointCloud
        The cloud to transform
    angle_degrees : float
        Rotation angle in degrees
    axis : str
        Rotation axis: 'x', 'y', or 'z'
    """
    log(f"Applying rotation ({angle_degrees}° around {axis}) to '{cloud.getName()}'...")
    
    # Rotation axis vector
    axis_vectors = {
        'x': cccorelib.CCVector3(1.0, 0.0, 0.0),
        'y': cccorelib.CCVector3(0.0, 1.0, 0.0),
        'z': cccorelib.CCVector3(0.0, 0.0, 1.0),
    }
    axis_vec = axis_vectors.get(axis.lower(), axis_vectors['z'])
    
    # Get center of the cloud
    center = cloud.getDisplayBB_recursive(True).getCenter()
    
    # Step 1: Translate to origin
    glTrans = cloud.getGLTransformation()
    translation = glTrans.getTranslationAsVec3D()
    translation = translation - center
    glTrans.setTranslation(translation)
    cloud.setGLTransformation(glTrans)
    cloud.applyGLTransformation_recursive()
    
    # Step 2: Rotate
    glRot = pycc.ccGLMatrix()
    glRot.initFromParameters(
        math.radians(angle_degrees),
        axis_vec,
        cccorelib.CCVector3(0.0, 0.0, 0.0)
    )
    
    glMat = cloud.getGLTransformation()
    glMat = glMat * glRot
    
    # Step 3: Translate back
    translation = glMat.getTranslationAsVec3D()
    translation = translation + center
    glMat.setTranslation(translation)
    cloud.setGLTransformation(glMat)
    cloud.applyGLTransformation_recursive()
    
    log("  Rotation applied")


# ==============================================================================
# ICP REGISTRATION DOCUMENTATION
# ==============================================================================

def document_icp_limitation():
    """
    Document the ICP limitation in the Python Plugin API.
    
    Based on API research, ICPRegistrationTools is NOT exposed in the Python API.
    Only RegistrationTools with transformation filters is available.
    """
    log_section("ICP Registration - API Limitation")
    log("")
    log("ICPRegistrationTools is NOT exposed in the Python Plugin API.")
    log("The cccorelib module only exposes RegistrationTools with filters:")
    log("  - SKIP_NONE, SKIP_ROTATION, SKIP_TRANSLATION, etc.")
    log("")
    log("For ICP alignment, use one of these alternatives:")
    log("  1. CloudCompare GUI: Tools -> Registration -> Fine Registration (ICP)")
    log("  2. CloudCompare CLI: CloudCompare -O ref.laz -O mov.laz -ICP")
    log("  3. Our custom Python implementation in src/terrain_change_detection/")
    log("")
    log("The clouds are loaded and ready for manual ICP via the GUI.")
    log("")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_pipeline():
    """Execute the demonstration pipeline."""
    log_section("CloudCompare Python Plugin Demo")
    log("Based on official API documentation")
    log("")
    
    CC = pycc.GetInstance()
    
    # Validate configuration
    errors = Config.validate()
    if errors:
        for e in errors:
            log(e, level="ERROR")
        raise RuntimeError("Config validation failed")
    
    # Report available features
    log("Available features:")
    log(f"  - qM3C2 plugin: {'YES' if HAS_QM3C2 else 'NO'}")
    log(f"  - CloudSamplingTools: YES")
    log(f"  - Transformations (ccGLMatrix): YES")
    log(f"  - ICPRegistrationTools: NO (not exposed in Python API)")
    log("")
    
    # Clear database if requested
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
    
    # Step 2: Subsample for faster processing (demonstration)
    log_section("Step 2: Subsampling (optional)")
    ref_sub = subsample_cloud(ref, Config.SUBSAMPLE_COUNT)
    mov_sub = subsample_cloud(mov, Config.SUBSAMPLE_COUNT)
    
    if ref_sub != ref:
        CC.addToDB(ref_sub)
    if mov_sub != mov:
        CC.addToDB(mov_sub)
    CC.updateUI()
    
    # Step 3: Centroid-based alignment (coarse alignment)
    align_success, _ = align_clouds_by_centroid(ref, mov)
    CC.updateUI()
    
    # Step 4: Document ICP limitation (fine alignment not available via Python API)
    document_icp_limitation()
    
    # Step 5: M3C2 Distance Computation
    m3c2_result = None
    if HAS_QM3C2:
        m3c2_result = compute_m3c2(ref, mov, Config.M3C2_PARAMS_FILE)
        if m3c2_result:
            CC.addToDB(m3c2_result)
    else:
        log_section("Step 4: M3C2 Distance")
        log("Skipped - qM3C2 plugin not available")
        log("See documentation above for alternatives")
    
    # Update UI
    CC.updateUI()
    CC.redrawAll()
    
    # Summary
    log_section("Pipeline Summary")
    log(f"Reference (T1): {Config.T1_PATH.name} - {ref.size():,} points")
    log(f"Moving (T2): {Config.T2_PATH.name} - {mov.size():,} points")
    if ref_sub != ref:
        log(f"Reference subsampled: {ref_sub.size():,} points")
    if mov_sub != mov:
        log(f"Moving subsampled: {mov_sub.size():,} points")
    log(f"M3C2 computed: {'YES' if m3c2_result else 'NO'}")
    log("")
    log("WHAT WORKS in Python Plugin:")
    log("  [OK] Loading point clouds")
    log("  [OK] Cloud subsampling (CloudSamplingTools)")
    log("  [OK] Point cloud transformations (ccGLMatrix)")
    log(f"  [{'OK' if HAS_QM3C2 else '--'}] M3C2 distance (requires qM3C2 plugin)")
    log("")
    log("WHAT REQUIRES CLI/GUI:")
    log("  [--] ICP Registration (ICPRegistrationTools not exposed)")
    log("  [--] C2C Distance (DistanceComputationTools.computeCloud2CloudDistances not exposed)")
    log("")
    log("For production workflows, use our custom Python implementation")
    log("or the CloudCompare CLI scripts in this exploration/ directory.")
    
    return True


if __name__ == "__main__":
    run_pipeline()
