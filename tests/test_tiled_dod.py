"""
Test script to diagnose tiled DoD issues.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.detection import ChangeDetector
from terrain_change_detection.utils.logging import setup_logger

logger = setup_logger(__name__)

def test_tiled_dod():
    """Test the tiled DoD computation."""
    
    # Use the actual files
    files_t1 = [
        "data/raw/eksport_1225654_20250602/2015/data/eksport_1225654_416_1.laz"
    ]
    
    files_t2 = [
        "data/processed/eksport_1225654_20250602/2020_aligned/eksport_1225654_4241_1_aligned.laz"
    ]
    
    # Check if files exist
    for f in files_t1 + files_t2:
        if not Path(f).exists():
            logger.error(f"File not found: {f}")
            return
    
    logger.info("Testing tiled DoD computation...")
    logger.info(f"Files T1: {files_t1}")
    logger.info(f"Files T2: {files_t2}")
    
    try:
        result = ChangeDetector.compute_dod_streaming_files_tiled(
            files_t1=files_t1,
            files_t2=files_t2,
            cell_size=2.0,
            tile_size=1000.0,
            halo=50.0,
            ground_only=True,
            classification_filter=[2],
            chunk_points=2000000,
        )
        
        logger.info("SUCCESS! Tiled DoD computed")
        logger.info(f"Grid shape: {result.dod.shape}")
        logger.info(f"Stats: {result.stats}")
        
    except Exception as e:
        logger.error(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tiled_dod()
