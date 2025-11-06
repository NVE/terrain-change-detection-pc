"""
Streaming alignment utilities

Provides utilities for applying transformations to point cloud files
in a streaming fashion, enabling out-of-core alignment processing.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np

from ..utils.logging import setup_logger
from ..acceleration.tiling import LaspyStreamReader

logger = setup_logger(__name__)


def apply_transform_to_files(
    input_files: List[str],
    output_dir: str,
    transform: np.ndarray,
    *,
    ground_only: bool = True,
    classification_filter: Optional[List[int]] = None,
    chunk_points: int = 1_000_000,
    preserve_attributes: bool = True,
) -> List[str]:
    """Apply a transformation matrix to LAZ files in streaming fashion.
    
    Reads files in chunks, applies the transformation, and writes to new files.
    This enables transformation of datasets too large to fit in memory.
    
    Args:
        input_files: List of input LAZ/LAS file paths
        output_dir: Directory to write transformed files
        transform: 4x4 transformation matrix to apply
        ground_only: If True, only process ground points (class 2)
        classification_filter: Optional list of classification codes to process
        chunk_points: Number of points to process per chunk
        preserve_attributes: If True, copy all attributes to output files
        
    Returns:
        List of output file paths
        
    Raises:
        ImportError: If laspy is not available
        ValueError: If transform is not a 4x4 matrix
    """
    import laspy
    
    if transform.shape != (4, 4):
        raise ValueError(f"Transform must be 4x4 matrix, got {transform.shape}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = []
    R = transform[:3, :3]  # Rotation matrix
    t = transform[:3, 3]   # Translation vector
    
    logger.info(f"Applying transformation to {len(input_files)} files...")
    logger.info(f"Ground only: {ground_only}, Classification filter: {classification_filter}")
    logger.info(f"Chunk size: {chunk_points} points")
    
    for input_file in input_files:
        input_path = Path(input_file)
        output_path = output_dir / f"{input_path.stem}_aligned{input_path.suffix}"
        output_files.append(str(output_path))
        
        logger.info(f"Transforming {input_path.name} -> {output_path.name}")
        
        total_points_written = 0
        
        # Open input file
        with laspy.open(input_path) as reader:
            header = reader.header
            logger.info(f"Input file has {header.point_count} total points")
            
            # Create output file with same header
            with laspy.open(output_path, mode='w', header=header) as writer:
                # Process in chunks
                for chunk in reader.chunk_iterator(chunk_points):
                    # Apply classification filter if needed
                    if hasattr(chunk, 'classification'):
                        from ..utils.point_cloud_filters import create_classification_mask
                        classes = np.asarray(chunk.classification)
                        mask = create_classification_mask(classes, ground_only, classification_filter)
                    else:
                        mask = np.ones(len(chunk.points), dtype=bool)
                    
                    # Skip if no points pass filter
                    if not np.any(mask):
                        continue
                    
                    # Extract and transform coordinates
                    x = np.asarray(chunk.x, dtype=np.float64)[mask]
                    y = np.asarray(chunk.y, dtype=np.float64)[mask]
                    z = np.asarray(chunk.z, dtype=np.float64)[mask]
                    
                    # Apply transformation: P' = R @ P + t
                    points = np.column_stack([x, y, z])
                    transformed = (R @ points.T).T + t
                    
                    # Create output chunk
                    # Note: This is a simplified version. Full implementation would
                    # need to handle all point attributes and formats properly.
                    out_chunk = laspy.ScaleAwarePointRecord.zeros(
                        len(transformed),
                        header=header
                    )
                    
                    out_chunk.x = transformed[:, 0]
                    out_chunk.y = transformed[:, 1]
                    out_chunk.z = transformed[:, 2]
                    
                    # Copy attributes if requested
                    if preserve_attributes:
                        for attr_name in chunk.point_format.dimension_names:
                            if attr_name not in ['X', 'Y', 'Z']:
                                try:
                                    setattr(out_chunk, attr_name, getattr(chunk, attr_name)[mask])
                                except Exception:
                                    pass  # Skip attributes that can't be copied
                    
                    writer.write_points(out_chunk)
                    total_points_written += len(out_chunk)
        
        logger.info(f"Wrote transformed file: {output_path}")
        logger.info(f"  Total points written: {total_points_written}")
    
    logger.info(f"Transformation complete. Created {len(output_files)} files.")
    return output_files


def save_transform_matrix(transform: np.ndarray, output_file: str) -> None:
    """Save a transformation matrix to a text file.
    
    Args:
        transform: 4x4 transformation matrix
        output_file: Path to output file
    """
    np.savetxt(output_file, transform, fmt='%.18e', header='4x4 transformation matrix')
    logger.info(f"Saved transformation matrix to {output_file}")


def load_transform_matrix(input_file: str) -> np.ndarray:
    """Load a transformation matrix from a text file.
    
    Args:
        input_file: Path to input file
        
    Returns:
        4x4 transformation matrix
    """
    transform = np.loadtxt(input_file)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {transform.shape}")
    logger.info(f"Loaded transformation matrix from {input_file}")
    return transform
