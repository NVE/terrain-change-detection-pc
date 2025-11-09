# Parallelization Implementation Plan

**Date**: November 9, 2025  
**Branch**: `feat/gpu-acceleration`  
**Phase**: 1 of 2 (Parallelization → GPU Acceleration)

## Executive Summary

This document outlines the implementation plan for **CPU-based parallelization** of the terrain change detection pipeline. With the out-of-core tiling infrastructure now complete (from `feat/outofcore-tiling`), we have a solid foundation for parallel processing. This phase focuses on **multi-process tile processing** to leverage all CPU cores before moving to GPU acceleration in Phase 2.

**Key Insight**: The tiling system creates **naturally independent units of work**—each tile can be processed in parallel without coordination. This is the perfect entry point for parallelization.

## Current State Assessment

### What We Have ✅

From the `feat/outofcore-tiling` branch merge, we now have:

1. **Production-Ready Tiling System** (`src/terrain_change_detection/acceleration/tiling.py`):
   - `Tiler`: Grid-aligned spatial partitioning with configurable tile size and halo
   - `Tile`: Inner/outer bounds for each tile with halo support
   - `GridAccumulator`: Streaming mean aggregator for DEM building
   - `MosaicAccumulator`: Seamless tile stitching with overlap averaging
   - `LaspyStreamReader`: Chunked LAZ/LAS reading with spatial filtering

2. **Three Streaming Change Detection Methods**:
   - `compute_dod_streaming_files_tiled()`: Out-of-core DoD with mosaicking
   - `compute_c2c_streaming_files_tiled()`: Tiled C2C with radius-bounded queries
   - `compute_m3c2_streaming_files_tiled()`: Tiled M3C2 with py4dgeo integration

3. **Configuration Infrastructure**:
   - Complete YAML-based configuration system
   - `outofcore` section with tile_size_m, halo_m, chunk_points
   - Config profiles for different scales (default, large_scale, synthetic)

4. **Workflow Integration** (`scripts/run_workflow.py`):
   - Automatic routing between in-memory vs. streaming modes
   - File-based transformation pipeline
   - Comprehensive logging with timing and metrics

### What's Missing (Our Target) 🎯

**Sequential Processing Bottleneck**: All three tiled methods use **sequential for-loops**:

```python
# Current: Sequential tile processing
for tile in tiler.tiles():
    # Process one tile at a time
    acc1 = GridAccumulator(tile.inner, cell_size)
    acc2 = GridAccumulator(tile.inner, cell_size)
    # ... accumulate points, compute differences
```

**Impact**: On a machine with 8+ CPU cores, we're only using ~12-15% of available compute capacity. Processing 100 tiles sequentially that could run in parallel is leaving massive performance on the table.

**Opportunity**: With independent tiles and a proven tiling system, we can achieve **near-linear speedup** with the number of CPU cores (6-12x on typical machines).

## Implementation Strategy

### Design Principles

1. **Minimal Invasiveness**: Leverage existing tiling code; add parallel wrapper layer
2. **Backwards Compatibility**: Keep sequential mode as fallback
3. **Memory Safety**: Control worker processes to avoid memory explosion
4. **Progressive Enhancement**: Start with simple multiprocessing, optimize iteratively
5. **Configuration Driven**: Enable/disable parallelization via config

### Architecture Overview

```
Parallel Processing Layer
├── parallel_executor.py      # Core parallel execution infrastructure
├── tile_workers.py            # Worker functions for each detection method
└── io_pool.py                 # Optional: parallel I/O for LAZ reading

Integration Points
├── detection/change_detection.py  # Add parallel variants of tiled methods
└── scripts/run_workflow.py        # Route to parallel or sequential based on config
```

## Phase 1.1: Foundation (Week 1)

### Goal
Set up parallel processing infrastructure with basic tile-level parallelization for DoD.

### Implementation Tasks

#### Task 1.1: Create Parallel Executor Module

**File**: `src/terrain_change_detection/acceleration/parallel_executor.py`

**Core Class**:
```python
class TileParallelExecutor:
    """
    Parallel executor for tile-based processing.
    
    Manages worker pool, distributes tiles, collects results.
    Handles memory constraints by limiting concurrent workers.
    """
    
    def __init__(self, n_workers: Optional[int] = None, 
                 memory_limit_gb: Optional[float] = None):
        """
        Args:
            n_workers: Number of worker processes (default: cpu_count - 1)
            memory_limit_gb: Soft memory limit to control concurrency
        """
        self.n_workers = n_workers or max(1, os.cpu_count() - 1)
        self.memory_limit_gb = memory_limit_gb
    
    def map_tiles(self, 
                  tiles: List[Tile],
                  worker_fn: Callable,
                  worker_kwargs: Dict[str, Any],
                  progress_callback: Optional[Callable] = None) -> List[Any]:
        """
        Map worker function over tiles in parallel.
        
        Args:
            tiles: List of tiles to process
            worker_fn: Function to apply to each tile
            worker_kwargs: Fixed kwargs passed to each worker
            progress_callback: Optional callback(tile_idx, n_tiles)
            
        Returns:
            List of results in same order as tiles
        """
        # Implementation using multiprocessing.Pool
        # with proper error handling and progress tracking
```

**Key Features**:
- Process pool management with configurable workers
- Error handling and logging per tile
- Progress tracking with callbacks
- Memory monitoring to prevent OOM
- Graceful degradation to sequential on errors

#### Task 1.2: Create DoD Tile Worker

**File**: `src/terrain_change_detection/acceleration/tile_workers.py`

**Worker Function**:
```python
def process_dod_tile(
    tile: Tile,
    files_t1: List[Path],
    files_t2: List[Path],
    cell_size: float,
    chunk_points: int,
    classification_filter: Optional[List[int]],
    transform_matrix: Optional[np.ndarray] = None
) -> Tuple[Tile, np.ndarray, np.ndarray]:
    """
    Process single DoD tile in worker process.
    
    Args:
        tile: Tile bounds (inner and outer)
        files_t1: Epoch 1 LAZ files
        files_t2: Epoch 2 LAZ files
        cell_size: DEM grid cell size
        chunk_points: Points per chunk for streaming
        classification_filter: Optional classification codes to include
        transform_matrix: Optional transformation for epoch 2
        
    Returns:
        Tuple of (tile, dem1, dem2) where DEMs are 2D arrays
    """
    # Create accumulators for this tile
    acc1 = GridAccumulator(tile.inner, cell_size)
    acc2 = GridAccumulator(tile.inner, cell_size)
    
    # Stream and accumulate epoch 1 points
    reader1 = LaspyStreamReader(files_t1, tile.outer, chunk_points)
    for chunk in reader1.iter_chunks(classification_filter):
        acc1.accumulate(chunk)
    
    # Stream and accumulate epoch 2 points (with optional transform)
    reader2 = LaspyStreamReader(files_t2, tile.outer, chunk_points)
    for chunk in reader2.iter_chunks(classification_filter):
        if transform_matrix is not None:
            # Apply transformation on-the-fly
            chunk = apply_transform(chunk, transform_matrix)
        acc2.accumulate(chunk)
    
    # Compute mean DEMs
    dem1 = acc1.compute_mean()
    dem2 = acc2.compute_mean()
    
    return (tile, dem1, dem2)
```

**Design Notes**:
- Worker is a **pure function** (no shared state)
- All inputs passed explicitly (picklable)
- LAZ reading happens in worker (distributes I/O)
- Returns minimal data (tile + DEMs, not full accumulators)

#### Task 1.3: Add Parallel DoD Method

**File**: `src/terrain_change_detection/detection/change_detection.py`

**New Method**:
```python
def compute_dod_streaming_files_tiled_parallel(
    self,
    files_t1: List[Path],
    files_t2: List[Path],
    bounds: Bounds2D,
    cell_size: float,
    tile_size_m: float,
    halo_m: float,
    chunk_points: int = 1_000_000,
    classification_filter: Optional[List[int]] = None,
    transform_matrix: Optional[np.ndarray] = None,
    n_workers: Optional[int] = None,
    memmap_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Parallel version of compute_dod_streaming_files_tiled.
    
    Processes tiles in parallel using multiple CPU cores.
    
    Args:
        ... (same as sequential version)
        n_workers: Number of parallel workers (default: cpu_count - 1)
        
    Returns:
        Same as sequential version: mosaic DEM and statistics
    """
    from ..acceleration.parallel_executor import TileParallelExecutor
    from ..acceleration.tile_workers import process_dod_tile
    
    # Create tiler
    tiler = Tiler(bounds, tile_size_m, halo_m)
    tiles = list(tiler.tiles())
    
    logger.info(f"Processing {len(tiles)} tiles in parallel with {n_workers or 'auto'} workers")
    
    # Create parallel executor
    executor = TileParallelExecutor(n_workers=n_workers)
    
    # Process tiles in parallel
    worker_kwargs = {
        'files_t1': files_t1,
        'files_t2': files_t2,
        'cell_size': cell_size,
        'chunk_points': chunk_points,
        'classification_filter': classification_filter,
        'transform_matrix': transform_matrix,
    }
    
    results = executor.map_tiles(
        tiles=tiles,
        worker_fn=process_dod_tile,
        worker_kwargs=worker_kwargs,
    )
    
    # Mosaic results (same as sequential)
    mosaic = MosaicAccumulator(bounds, cell_size, memmap_dir=memmap_dir)
    for tile, dem1, dem2 in results:
        mosaic.add_tile(tile.inner, dem1, dem2)
    
    return mosaic.finalize()
```

#### Task 1.4: Configuration Updates

**File**: `config/default.yaml`

**Add Parallel Section**:
```yaml
parallel:
  enabled: true              # Master switch for parallelization
  n_workers: null            # Auto-detect: cpu_count - 1
  memory_limit_gb: null      # Soft limit to control concurrency
  method: "multiprocessing"  # Future: "dask", "ray"
```

**Update Workflow Logic**:
```python
# In scripts/run_workflow.py
if cfg.parallel.enabled and cfg.outofcore.enabled:
    result = detector.compute_dod_streaming_files_tiled_parallel(...)
else:
    result = detector.compute_dod_streaming_files_tiled(...)
```

### Testing & Validation

**Unit Tests** (`tests/test_parallel_executor.py`):
- Worker pool creation and teardown
- Error handling in workers
- Result ordering preservation
- Memory limit enforcement

**Integration Tests** (`tests/test_parallel_dod.py`):
- Parallel DoD produces identical results to sequential
- Performance improvement measured (speedup factor)
- Handles various tile configurations (1 tile, 10 tiles, 100 tiles)

**Benchmarks**:
- Compare sequential vs parallel on synthetic dataset
- Measure speedup with 2, 4, 8, 16 workers
- Profile memory usage under parallel load
- Document optimal worker counts for different dataset sizes

### Expected Results

**Performance**:
- **6-8x speedup** on 8-core machine (typical workstation)
- **10-12x speedup** on 16-core machine (server)
- Near-linear scaling up to I/O saturation point

**Memory**:
- Memory usage = `n_workers × (tile_working_set + DEM memory)`
- Controlled by worker count and tile size
- Mosaic accumulation remains memory-efficient

**Validation**:
- Results match sequential implementation (numerical parity)
- All existing tests continue to pass
- No regression in sequential mode

## Phase 1.2: C2C and M3C2 Parallelization (Week 2)

### Goal
Extend parallelization to C2C and M3C2 methods with appropriate worker implementations.

### Implementation Tasks

#### Task 2.1: C2C Tile Worker

**File**: `src/terrain_change_detection/acceleration/tile_workers.py`

**Worker Function**:
```python
def process_c2c_tile(
    tile: Tile,
    files_source: List[Path],
    files_target: List[Path],
    max_distance: float,
    chunk_points: int,
    classification_filter: Optional[List[int]],
    transform_matrix: Optional[np.ndarray] = None,
    k_neighbors: int = 1,
) -> Tuple[Tile, np.ndarray]:
    """
    Process single C2C tile in worker process.
    
    Returns:
        Tuple of (tile, distances) where distances is 1D array
    """
    # Load source points (inner tile only)
    source_points = []
    reader_src = LaspyStreamReader(files_source, tile.inner, chunk_points)
    for chunk in reader_src.iter_chunks(classification_filter):
        source_points.append(chunk)
    source = np.vstack(source_points) if source_points else np.empty((0, 3))
    
    # Load target points (outer tile with halo for radius coverage)
    target_points = []
    reader_tgt = LaspyStreamReader(files_target, tile.outer, chunk_points)
    for chunk in reader_tgt.iter_chunks(classification_filter):
        if transform_matrix is not None:
            chunk = apply_transform(chunk, transform_matrix)
        target_points.append(chunk)
    target = np.vstack(target_points) if target_points else np.empty((0, 3))
    
    # Compute C2C distances using sklearn KDTree
    if len(source) == 0 or len(target) == 0:
        return (tile, np.array([]))
    
    tree = KDTree(target)
    distances, _ = tree.query(source, k=k_neighbors)
    distances = distances.flatten()
    
    # Filter by max_distance
    valid = distances <= max_distance
    distances = distances[valid]
    
    return (tile, distances)
```

#### Task 2.2: M3C2 Tile Worker

**File**: `src/terrain_change_detection/acceleration/tile_workers.py`

**Worker Function**:
```python
def process_m3c2_tile(
    tile: Tile,
    tile_idx: int,
    core_points_tile: np.ndarray,  # Pre-partitioned core points for this tile
    files_t1: List[Path],
    files_t2: List[Path],
    chunk_points: int,
    classification_filter: Optional[List[int]],
    m3c2_params: Dict[str, Any],
) -> Tuple[int, np.ndarray]:
    """
    Process single M3C2 tile in worker process.
    
    Args:
        tile_idx: Tile index for result ordering
        core_points_tile: Core points assigned to this tile
        m3c2_params: M3C2 algorithm parameters
        
    Returns:
        Tuple of (tile_idx, distances) for in-order assembly
    """
    # Load epoch 1 points for this tile (outer bounds for neighborhoods)
    points_t1 = []
    reader1 = LaspyStreamReader(files_t1, tile.outer, chunk_points)
    for chunk in reader1.iter_chunks(classification_filter):
        points_t1.append(chunk)
    pc1 = np.vstack(points_t1) if points_t1 else np.empty((0, 3))
    
    # Load epoch 2 points
    points_t2 = []
    reader2 = LaspyStreamReader(files_t2, tile.outer, chunk_points)
    for chunk in reader2.iter_chunks(classification_filter):
        points_t2.append(chunk)
    pc2 = np.vstack(points_t2) if points_t2 else np.empty((0, 3))
    
    # Run py4dgeo M3C2 for this tile's core points
    m3c2 = M3C2(**m3c2_params)
    distances, _ = m3c2.compute(
        corepoints=core_points_tile,
        epoch1=pc1,
        epoch2=pc2,
    )
    
    return (tile_idx, distances)
```

#### Task 2.3: Add Parallel C2C and M3C2 Methods

**File**: `src/terrain_change_detection/detection/change_detection.py`

Add two new methods:
- `compute_c2c_streaming_files_tiled_parallel()`
- `compute_m3c2_streaming_files_tiled_parallel()`

Following the same pattern as parallel DoD.

### Testing & Validation

**Tests**:
- Parallel C2C matches sequential (same distance statistics)
- Parallel M3C2 matches sequential (same distance arrays)
- Performance benchmarks on multi-tile scenarios

**Expected Results**:
- Similar speedup factors as DoD (6-12x on typical hardware)
- Memory remains bounded by worker count

## Phase 1.3: I/O Optimization (Week 3)

### Goal
Optimize LAZ file reading to prevent I/O bottleneck with parallel tile processing.

### Challenge

With parallel tile processing, multiple workers read from the same LAZ files simultaneously. This can cause:
1. **Disk contention**: Random seeks if each worker reads different spatial regions
2. **Redundant reads**: Same file chunks read by multiple workers
3. **I/O saturation**: Spinning disk throughput becomes bottleneck

### Solutions

#### Option A: Spatial Pre-Indexing (Recommended)

**Approach**: Build spatial index for LAZ files before parallel processing.

```python
class SpatialIndex:
    """
    Pre-computed spatial index mapping tiles to file byte ranges.
    Enables efficient seeking during parallel processing.
    """
    
    def build_index(self, files: List[Path], tiler: Tiler) -> Dict[int, List[Tuple[Path, int, int]]]:
        """
        Build index: tile_id -> [(file, offset, size), ...]
        
        Scans each file once, records which chunks belong to which tiles.
        """
        pass
    
    def get_tile_chunks(self, tile_id: int) -> List[Tuple[Path, int, int]]:
        """Get pre-computed file ranges for a tile."""
        pass
```

**Benefits**:
- One sequential scan per file (build index)
- Workers seek directly to relevant byte ranges
- Eliminates redundant reads

**Cost**: Upfront index building time (5-10% of total processing)

#### Option B: Prefetch Pool

**Approach**: Dedicated I/O threads that prefetch tile data.

```python
class PrefetchPool:
    """
    Background I/O pool that prefetches tile data ahead of workers.
    """
    
    def __init__(self, n_io_threads: int = 2):
        self.pool = ThreadPoolExecutor(max_workers=n_io_threads)
        self.cache = {}  # tile_id -> cached points
    
    def prefetch_tiles(self, tiles: List[Tile], files: List[Path]):
        """Asynchronously load data for upcoming tiles."""
        pass
```

**Benefits**:
- Overlaps I/O with computation
- Simple implementation

**Drawback**: Increased memory usage for cache

#### Option C: File-Level Parallelism

**Approach**: Read files in parallel, route chunks to appropriate tiles.

```python
def parallel_file_reader(files: List[Path], tiler: Tiler) -> Dict[int, List[np.ndarray]]:
    """
    Read files in parallel, partition chunks by tile.
    
    Returns:
        tile_id -> [chunk1, chunk2, ...] for all tiles
    """
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(read_file_and_partition, f, tiler) for f in files]
        # Combine results by tile
    pass
```

**Benefits**:
- Parallel file reading
- Each file read once

**Drawback**: All data in memory simultaneously (defeats streaming)

#### Recommendation

**For Phase 1.3**: Implement **Option A (Spatial Pre-Indexing)**:
- Best balance of performance and memory efficiency
- Compatible with streaming architecture
- Scalable to large file counts

**Implementation File**: `src/terrain_change_detection/acceleration/spatial_index.py`

### Testing & Validation

**Benchmarks**:
- Measure I/O time vs compute time before/after optimization
- Compare HDD vs SSD performance
- Test with varying file counts (1, 10, 50, 100 files)

**Expected Results**:
- I/O time reduced by 40-60% with indexing
- Workers spend < 20% of time waiting on I/O
- Total speedup: 8-15x on 8-core machine (improved from 6-8x)

## Phase 1.4: Optimization and Tuning (Week 4)

### Goal
Fine-tune parallel performance, optimize memory usage, and prepare for production.

### Implementation Tasks

#### Task 4.1: Adaptive Worker Count

**Problem**: Optimal worker count varies by dataset and hardware.

**Solution**: Auto-tune based on heuristics:

```python
def estimate_optimal_workers(
    dataset_size_gb: float,
    available_memory_gb: float,
    cpu_count: int,
    tile_count: int,
) -> int:
    """
    Estimate optimal worker count based on resources.
    
    Rules:
    1. Don't exceed CPU count
    2. Ensure enough memory per worker (2-4 GB minimum)
    3. Don't spawn more workers than tiles
    4. Leave headroom for system and mosaicking
    """
    # Memory constraint
    memory_per_worker = 3.0  # GB
    max_workers_memory = int(available_memory_gb * 0.7 / memory_per_worker)
    
    # CPU constraint
    max_workers_cpu = max(1, cpu_count - 1)
    
    # Work constraint
    max_workers_work = tile_count
    
    # Take minimum
    optimal = min(max_workers_memory, max_workers_cpu, max_workers_work)
    
    return max(1, optimal)
```

#### Task 4.2: Progress Reporting

Add real-time progress reporting for long-running parallel jobs:

```python
class TileProgressReporter:
    """
    Progress reporter for parallel tile processing.
    """
    
    def __init__(self, total_tiles: int):
        self.total = total_tiles
        self.completed = 0
        self.start_time = time.time()
    
    def update(self, tile_idx: int):
        """Called when a tile completes."""
        self.completed += 1
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed
        eta = (self.total - self.completed) / rate if rate > 0 else 0
        
        logger.info(
            f"Progress: {self.completed}/{self.total} tiles "
            f"({100*self.completed/self.total:.1f}%) - "
            f"ETA: {eta:.1f}s"
        )
```

#### Task 4.3: Error Recovery

Implement robust error handling for worker failures:

```python
class RobustTileExecutor(TileParallelExecutor):
    """
    Enhanced executor with retry logic and error recovery.
    """
    
    def map_tiles(self, tiles, worker_fn, worker_kwargs, max_retries=2):
        """
        Process tiles with automatic retry on failure.
        
        If a tile fails repeatedly, log error and skip it (rather than failing entire job).
        """
        results = []
        failed_tiles = []
        
        for tile in tiles:
            for attempt in range(max_retries + 1):
                try:
                    result = self._process_tile(tile, worker_fn, worker_kwargs)
                    results.append(result)
                    break
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Tile {tile} failed after {max_retries} retries: {e}")
                        failed_tiles.append(tile)
                    else:
                        logger.warning(f"Tile {tile} attempt {attempt+1} failed, retrying...")
        
        if failed_tiles:
            logger.warning(f"Processing completed with {len(failed_tiles)} failed tiles")
        
        return results
```

#### Task 4.4: Memory Profiling

Add memory monitoring to prevent OOM:

```python
def monitor_memory(threshold_gb: float = None):
    """
    Monitor memory usage during parallel processing.
    
    Args:
        threshold_gb: Warning threshold (default: 80% of available)
    """
    import psutil
    
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / 1024**3
    
    available = psutil.virtual_memory().available / 1024**3
    total = psutil.virtual_memory().total / 1024**3
    
    if threshold_gb and mem_gb > threshold_gb:
        logger.warning(
            f"High memory usage: {mem_gb:.1f} GB / {total:.1f} GB "
            f"(available: {available:.1f} GB)"
        )
```

### Testing & Validation

**Stress Tests**:
- Process 1000+ tiles to validate stability
- Inject random worker failures to test recovery
- Monitor memory over extended runs

**Performance Profiling**:
- Measure actual speedup factors on real datasets
- Identify any remaining bottlenecks
- Document performance characteristics

## Configuration Reference

### Complete Parallel Configuration

```yaml
# config/default.yaml

outofcore:
  enabled: true
  tile_size_m: 1000.0
  halo_m: 10.0
  chunk_points: 1000000
  streaming_mode: true
  save_transformed_files: false
  output_dir: null
  memmap_dir: null

parallel:
  # Master switch for parallelization
  enabled: true
  
  # Worker configuration
  n_workers: null              # null = auto (cpu_count - 1)
  memory_limit_gb: null        # null = no hard limit
  
  # I/O optimization
  use_spatial_index: true      # Build spatial index for efficient tile I/O
  prefetch_tiles: 2            # Number of tiles to prefetch ahead
  
  # Error handling
  max_retries: 2               # Retry failed tiles
  continue_on_error: true      # Don't fail entire job on single tile error
  
  # Progress reporting
  log_progress: true           # Log tile completion progress
  progress_interval: 10        # Log every N tiles
  
  # Advanced (for tuning)
  chunk_size: 1                # Tiles per worker batch (1 = dynamic assignment)
  maxtasksperchild: null       # Restart workers after N tasks (memory cleanup)
```

## Performance Targets

### Expected Speedup Factors

| Hardware | Sequential | Parallel | Speedup |
|----------|-----------|----------|---------|
| 4-core laptop | 100 min | 20 min | 5x |
| 8-core workstation | 100 min | 12 min | 8x |
| 16-core server | 100 min | 8 min | 12x |
| 32-core HPC node | 100 min | 6 min | 16x |

*Note: Speedup limited by I/O saturation beyond ~16 cores on typical storage*

### Memory Scaling

| Workers | Tile Working Set | Total Memory | Notes |
|---------|-----------------|--------------|-------|
| 1 | 2 GB | 3 GB | Sequential baseline |
| 4 | 2 GB | 10 GB | Light parallel |
| 8 | 2 GB | 18 GB | Typical workstation |
| 16 | 2 GB | 34 GB | Server-grade RAM needed |

## Next Steps After Phase 1

Once CPU parallelization is complete and validated, we move to **Phase 2: GPU Acceleration** (see `GPU_ACCELERATION_PLAN.md`).

**Phase 2 Preview**:
- GPU-accelerated nearest neighbor searches (10-50x for C2C, M3C2, ICP)
- GPU-accelerated grid operations (5-10x for DoD)
- JIT compilation for remaining CPU hotspots (2-5x)
- **Combined speedup**: 50-100x over original sequential implementation

## Success Criteria

Phase 1 is complete when:

1. ✅ All three methods (DoD, C2C, M3C2) have parallel implementations
2. ✅ Parallel results match sequential (numerical parity)
3. ✅ 6-12x speedup demonstrated on 8-16 core machines
4. ✅ Memory usage remains bounded and predictable
5. ✅ Comprehensive tests passing (unit, integration, benchmarks)
6. ✅ Production-ready error handling and logging
7. ✅ Documentation complete (code, config, user guide)
8. ✅ Performance profiling data collected for Phase 2 planning

## References

- Out-of-Core Implementation: `feat/outofcore-tiling` branch, `docs/CHANGELOG.md`
- Tiling System: `src/terrain_change_detection/acceleration/tiling.py`
- Change Detection: `src/terrain_change_detection/detection/change_detection.py`
- Configuration Guide: `docs/CONFIGURATION_GUIDE.md`
- Algorithm Guide: `docs/ALGORITHMS.md`
