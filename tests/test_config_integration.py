"""Tests for configuration integration with streaming features."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.utils.config import load_config, AppConfig


def test_default_config_outofcore_disabled():
    """Test that default config has out-of-core disabled by default."""
    cfg = load_config(None)  # Load default.yaml

    assert cfg.outofcore.enabled is False
    assert cfg.outofcore.streaming_mode is True  # Ready when enabled
    assert cfg.outofcore.tile_size_m == 500.0
    assert cfg.outofcore.halo_m == 20.0
    assert cfg.outofcore.chunk_points == 1_000_000
    # Default behavior: do not write transformed files unless explicitly enabled
    assert cfg.outofcore.save_transformed_files is False


def test_large_scale_profile_outofcore_enabled():
    """Test that large scale profile has out-of-core enabled with tuned params."""
    cfg = load_config("config/profiles/large_scale.yaml")

    assert cfg.outofcore.enabled is True
    assert cfg.outofcore.streaming_mode is True
    assert cfg.outofcore.tile_size_m == 500.0
    assert cfg.outofcore.halo_m == 30.0
    assert cfg.outofcore.chunk_points == 2_000_000
    assert cfg.outofcore.save_transformed_files is False
    # When save_transformed_files is false, output_dir is typically left null
    assert cfg.outofcore.output_dir is None


def test_synthetic_profile_outofcore_settings():
    """Test that synthetic profile has appropriate out-of-core settings."""
    cfg = load_config("config/profiles/synthetic.yaml")

    # Synthetic profile is intended for small, in-memory/dev datasets
    assert cfg.outofcore.enabled is False
    assert cfg.outofcore.streaming_mode is True
    assert cfg.outofcore.save_transformed_files is False  # Don't save for small datasets
    assert cfg.outofcore.tile_size_m == 300.0
    assert cfg.outofcore.halo_m == 20.0
    assert cfg.outofcore.chunk_points == 20_000


def test_outofcore_config_all_fields():
    """Test that OutOfCoreConfig has all expected fields."""
    cfg = AppConfig()

    # Check all fields exist
    assert hasattr(cfg.outofcore, "enabled")
    assert hasattr(cfg.outofcore, "tile_size_m")
    assert hasattr(cfg.outofcore, "halo_m")
    assert hasattr(cfg.outofcore, "chunk_points")
    assert hasattr(cfg.outofcore, "streaming_mode")
    assert hasattr(cfg.outofcore, "save_transformed_files")
    assert hasattr(cfg.outofcore, "output_dir")
    assert hasattr(cfg.outofcore, "memmap_dir")


def test_config_coordination_with_preprocessing():
    """Test that config coordinates preprocessing and out-of-core settings."""
    cfg = load_config("config/profiles/large_scale.yaml")
    
    # Preprocessing settings should work with streaming
    assert cfg.preprocessing.ground_only == True
    assert cfg.preprocessing.classification_filter == [2]
    
    # Alignment settings should be appropriate for streaming
    assert cfg.alignment.subsample_size > 0  # Need samples for alignment
    
    # Detection settings should work with tiling
    assert cfg.detection.dod.cell_size > 0


def test_config_coordination_with_alignment():
    """Test that config coordinates alignment and out-of-core settings."""
    cfg = load_config("config/profiles/large_scale.yaml")
    
    # When out-of-core enabled and save_transformed_files is true,
    # alignment should be able to work with file paths
    if cfg.outofcore.enabled and cfg.outofcore.save_transformed_files:
        assert cfg.outofcore.output_dir is not None or cfg.paths.base_dir is not None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
