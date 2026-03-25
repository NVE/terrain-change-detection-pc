"""
Terrain Change Detection Workflow — compatibility shim.

This script preserves the ``uv run scripts/run_workflow.py …`` and
``python scripts/run_workflow.py …`` entry points.  All workflow logic
has been moved to ``terrain_change_detection.workflow``.

**Boundary rule**: New workflow coordination logic belongs in
``terrain_change_detection.workflow``, not here.
"""
import sys
from pathlib import Path

# Path bootstrap: preserve direct `python scripts/run_workflow.py` usage
sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.workflow.cli import main  # noqa: E402
from terrain_change_detection.workflow.data_loading import resolve_subsample_count  # noqa: E402, F401

if __name__ == "__main__":
    main()
