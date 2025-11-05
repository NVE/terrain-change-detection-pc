import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.utils.config import load_config, AppConfig


def test_alignment_coarse_defaults():
    cfg: AppConfig = load_config(None)
    assert hasattr(cfg.alignment, "coarse")
    assert cfg.alignment.coarse.enabled is True
    assert cfg.alignment.coarse.method in {"pca", "centroid", "phase", "open3d_fpfh", "none"}

