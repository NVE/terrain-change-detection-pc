"""
Runtime bootstrap for the terrain change detection workflow.

Sets up logging, NumPy RNG, thread environment variables, and GPU diagnostics.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from terrain_change_detection.utils.config import AppConfig
from terrain_change_detection.utils.logging import setup_logger

from .types import WorkflowAbort


def setup_runtime(cfg: AppConfig) -> tuple[logging.Logger, np.random.Generator]:
    """Initialise logger, RNG, thread pools, and GPU probes.

    Args:
        cfg: Fully resolved application configuration.

    Returns:
        ``(logger, rng)`` — a configured logger and a seeded NumPy RNG.
    """
    # Suppress noisy library loggers
    logging.getLogger("terrain_change_detection.preprocessing.data_discovery").setLevel(
        logging.ERROR
    )
    logging.getLogger("terrain_change_detection.preprocessing.loader").setLevel(
        logging.ERROR
    )

    # Setup logging from config
    log_level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    logger = setup_logger(
        "terrain_change_detection.workflow", level=log_level, log_file=cfg.logging.file
    )

    # Deterministic RNG
    _seed = cfg.alignment.random_seed
    rng = np.random.default_rng(int(_seed))
    logger.info("NumPy RNG seeded with %d", int(_seed))

    # Thread tuning
    _setup_threads(cfg, logger)

    # GPU diagnostics
    _log_gpu_status(cfg, logger)

    logger.info("Terrain Change Detection Workflow")
    logger.info("=================================")

    return logger, rng


def _setup_threads(cfg: AppConfig, logger: logging.Logger) -> None:
    """Set thread env vars if configured."""
    try:
        threads = cfg.performance.numpy_threads
        if threads == "auto":
            threads = os.cpu_count() or 1
        if isinstance(threads, int) and threads > 0:
            os.environ["OMP_NUM_THREADS"] = str(threads)
            os.environ["MKL_NUM_THREADS"] = str(threads)
            os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    except Exception:
        pass


def _log_gpu_status(cfg: AppConfig, logger: logging.Logger) -> None:
    """Log GPU configuration status and check for GPU libraries."""
    try:
        import platform

        from terrain_change_detection.acceleration.hardware_detection import detect_gpu

        if not getattr(cfg.gpu, "enabled", False):
            logger.info("GPU Acceleration: DISABLED (CPU only)")
            return

        # Check if GPU libraries are available
        cupy_available = False
        cuml_available = False

        try:
            import cupy as cp  # noqa: F401

            cupy_available = True
        except ImportError:
            pass

        try:
            import cuml  # noqa: F401

            cuml_available = True
        except ImportError:
            pass

        is_windows = platform.system() == "Windows"

        if not cupy_available:
            logger.error("=" * 80)
            logger.error("ERROR: GPU is enabled in config but CuPy is not available!")
            logger.error("")
            if is_windows:
                logger.error("On Windows, install CuPy for GPU acceleration:")
                logger.error(
                    "  uv add cupy-cuda12x  # or cupy-cuda11x depending on your CUDA version"
                )
            else:
                logger.error(
                    "To use GPU acceleration, you must activate the GPU environment:"
                )
                logger.error("  source activate_gpu.sh")
            logger.error("")
            logger.error("Or disable GPU in your config file:")
            logger.error("  gpu:")
            logger.error("    enabled: false")
            logger.error("=" * 80)
            logger.error(
                "Exiting workflow. Please fix the configuration and try again."
            )
            raise WorkflowAbort(
                "GPU is enabled in config but CuPy is not available. "
                "Install CuPy or set gpu.enabled=false."
            )

        # Log GPU capability level
        if cuml_available:
            gpu_mode = "FULL (CuPy + cuML)"
        else:
            gpu_mode = "PARTIAL (CuPy only - cuML not available)"
            if not is_windows:
                logger.warning(
                    "cuML not available. For full GPU acceleration on Linux, activate GPU environment:"
                )
                logger.warning("  source activate_gpu.sh")

        gpu_info = detect_gpu()
        if gpu_info.available:
            logger.info("GPU Acceleration: ENABLED - %s", gpu_mode)
            logger.info("  Device: %s", gpu_info.device_name)
            logger.info("  Memory: %.2f GB", gpu_info.memory_gb)
            logger.info(
                "  C2C: %s",
                "ENABLED" if getattr(cfg.gpu, "use_for_c2c", False) else "DISABLED",
            )
            logger.info(
                "  DoD: %s",
                "ENABLED" if getattr(cfg.gpu, "use_for_dod", False) else "DISABLED",
            )
            logger.info(
                "  Alignment: %s",
                "ENABLED"
                if getattr(cfg.gpu, "use_for_alignment", False)
                else "DISABLED",
            )

            # Check for GPU + parallel processing incompatibility
            if getattr(cfg.parallel, "enabled", False):
                logger.warning("=" * 80)
                logger.warning("WARNING: GPU and parallel processing are both enabled!")
                logger.warning(
                    "CUDA contexts cannot survive process forking (multiprocessing limitation)."
                )
                logger.warning(
                    "This may cause 'CUDARuntimeError: cudaErrorInitializationError' in workers."
                )
                logger.warning(
                    "Recommendation: Disable either GPU or parallel processing."
                )
                logger.warning("  - To disable GPU: set gpu.enabled=false in config")
                logger.warning(
                    "  - To disable parallel: set parallel.enabled=false in config"
                )
                logger.warning("=" * 80)
        else:
            logger.warning(
                "GPU Acceleration: ENABLED in config but GPU not available (%s), will use CPU fallback",
                gpu_info.error_message,
            )
    except WorkflowAbort:
        raise
    except Exception as e:
        logger.debug("Could not check GPU status: %s", e)
