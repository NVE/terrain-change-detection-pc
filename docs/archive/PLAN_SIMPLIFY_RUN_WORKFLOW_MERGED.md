# Plan: Simplify `run_workflow.py`

---

## Problem Statement

`scripts/run_workflow.py` is **1,815 lines** with a single `main()` spanning **1,647 lines**,
containing **122 `if` branches**, **35 `try` blocks**, **14 repeated output-dir resolutions**,
and **6 repeated CRS detections**. It mixes CLI parsing, GPU probing, data loading,
coordinate transforms, clipping, three-phase ICP alignment, three change-detection
algorithms (DoD, C2C, M3C2), visualization, and export — all interwoven with
streaming-vs-in-memory branching.

| Root cause | ~Lines | Example |
|:---|:---|:---|
| Inline orchestration (streaming vs. in-memory forks) | ~400 | Repeated for loading, DoD, C2C, M3C2 |
| Repeated utility patterns | ~150 | CRS detection + fallback, output dir, coord conversion×6 |
| Full alignment pipeline inline | ~280 | Coarse → multi-scale → fine ICP + overlap + export |
| Three detection algorithms inline | ~630 | DoD, C2C, M3C2 each with 3 code paths |
| Debug / diagnostic code | ~80 | `debug_m3c2_compare` inline Pearson + summaries |
| GPU probing | ~80 | Library checks, platform checks, error messages |
| DEM raster export (pre/post alignment) | ~70 | Nearly identical blocks |

---

## Compatibility Guarantees (Frozen for This Pass)

> [!CAUTION]
> These items must not change during refactoring.

| Concern | Guarantee |
|:---|:---|
| **User CLI** | All 15 existing flags keep working identically. |
| **Config system** | `default.yaml` → repeated `--config` → dedicated CLI flags → repeated `--set`. |
| **Entry point** | Both `uv run scripts/run_workflow.py …` and `python scripts/run_workflow.py …` remain valid via the `src` path bootstrap. |
| **Output file names & directory layout** | Frozen exactly as-is, including the current inconsistency where DoD/C2C use `base_dir/output/` (flat) while M3C2/alignment use `base_dir/output/<area>/` (area-scoped). |
| **Log-and-return error behavior** | Current "log error and return" semantics preserved. No exit-code changes. |
| **README examples** | All usage examples continue to work. |
| **Library modules** | No changes to existing code in `alignment/`, `detection/`, `preprocessing/`, `acceleration/`, `utils/`, `visualization/`. |

### Explicit "Do Not" list

- Do **not** normalize output-path inconsistencies across methods.
- Do **not** add a `pyproject.toml` console script entrypoint.
- Do **not** change config keys or wire unused config fields.
- Do **not** change exit codes or exception semantics.
- Do **not** add new top-level re-exports in `terrain_change_detection/__init__.py` (it already does `from .preprocessing import *` etc., which would trigger side effects).

---

## Target Architecture

### New package: `src/terrain_change_detection/workflow/`

```text
src/terrain_change_detection/workflow/
├── __init__.py               # Public API: run_workflow()
├── cli.py                    # CLI parser + build_cli_overrides()
├── types.py                  # WorkflowRequest, PreparedData, AlignmentResult, WorkflowResult, WorkflowAbort
├── bootstrap.py              # Logger, RNG, thread tuning, GPU diagnostics
├── coordinate_setup.py       # Local coordinate transform setup
├── clipping.py               # Boundary-file validation + AreaClipper delegation
├── export_helpers.py         # CRS detection (cached), output-dir resolution, DEM export
├── visualization_helpers.py  # Global coord conversion for vis, conditional show_plots
├── data_loading.py           # Discovery, area/year selection, streaming vs. in-memory loading
├── alignment.py              # Coarse → multi-scale → fine ICP, overlap, subsampling, export
├── detection_dod.py          # DoD: streaming/parallel/in-memory + export
├── detection_c2c.py          # C2C: streaming/parallel/in-memory + export
├── detection_m3c2.py         # M3C2: core selection, autotune, streaming/parallel/in-memory, debug compare, export
└── runner.py                 # Orchestrator: calls phases in order, catches WorkflowAbort
```

### Simplified `scripts/run_workflow.py` (~12 lines)

```python
"""Terrain Change Detection Workflow — compatibility shim."""
import sys
from pathlib import Path

# Path bootstrap: preserve direct `python scripts/run_workflow.py` usage from a clone.
sys.path.append(str(Path(__file__).parent.parent / "src"))

from terrain_change_detection.workflow.cli import main

if __name__ == "__main__":
    main()
```

### Phase-boundary contracts (typed dataclasses in `types.py`)

| Type | Purpose | Key fields |
|:---|:---|:---|
| `WorkflowRequest` | Parsed CLI + config source metadata | `args`, `cfg`, `cli_overrides`, `config_files` |
| `PreparedData` | Output of data loading + clipping | `points1`, `points2`, `pc1_data`, `pc2_data`, `use_streaming`, `clip_bounds`, `local_transform`, `selected_area`, `t1`, `t2`, `ds1`, `ds2` |
| `AlignmentResult` | Output of alignment step | `points1_aligned`, `points2_aligned`, `transform_matrix`, `aligned_epoch`, `alignment_error`, `aligned_file_paths` |
| `WorkflowResult` | Summary for tests and future automation (internal) | `selected_area`, `epochs`, `streaming_used`, `alignment_status`, `export_paths` |
| `WorkflowAbort` | Exception replacing scattered `return` statements | `message`, `level` (error/warning) |

`WorkflowAbort` is caught once in `runner.py` to preserve current "log and stop" behavior
without changing exit semantics. `WorkflowResult` is internal for now — useful for tests,
not exposed broadly.

---

## Estimated Module Sizes

| Module | ~Lines | Notes |
|:---|:---|:---|
| `scripts/run_workflow.py` | ~12 | Thin shim |
| `workflow/__init__.py` | ~5 | Re-export `run_workflow` |
| `workflow/cli.py` | ~90 | Parser + `build_cli_overrides` |
| `workflow/types.py` | ~60 | Dataclasses + `WorkflowAbort` |
| `workflow/bootstrap.py` | ~80 | Logger, RNG, GPU, threads |
| `workflow/coordinate_setup.py` | ~40 | Local transform |
| `workflow/clipping.py` | ~50 | Clipping orchestration |
| `workflow/export_helpers.py` | ~50 | CRS cache, output dir, DEM export |
| `workflow/visualization_helpers.py` | ~30 | Coord conversion + conditional vis |
| `workflow/data_loading.py` | ~120 | Discovery + loading |
| `workflow/alignment.py` | ~200 | Full alignment pipeline |
| `workflow/detection_dod.py` | ~80 | DoD |
| `workflow/detection_c2c.py` | ~100 | C2C |
| `workflow/detection_m3c2.py` | ~150 | M3C2 |
| `workflow/runner.py` | ~60 | Orchestrator + `WorkflowAbort` catch |
| **Total** | **~1,127** | **38% reduction** from 1,815 |

---

## Extraction Order

Each phase can be validated independently before proceeding to the next.

| Phase | Extract | Risk | Validation |
|:---|:---|:---|:---|
| **1** | `cli.py`, `export_helpers.py`, `visualization_helpers.py`, `coordinate_setup.py`, `clipping.py` | Low — pure extraction of isolated logic | Existing tests pass; `--show-config` works |
| **2** | `types.py` (`WorkflowRequest`, `PreparedData`, `AlignmentResult`, `WorkflowResult`, `WorkflowAbort`) | Low — new code, no behavior change yet | Type-check and import test |
| **3** | `bootstrap.py` | Low — logger, RNG, GPU diagnostics | Existing tests; visual log output check |
| **4** | `data_loading.py` | Medium — streaming vs. in-memory branching | Synthetic data end-to-end |
| **5** | `alignment.py` | Medium — most complex single step | ICP tests + synthetic end-to-end |
| **6** | `detection_dod.py`, `detection_c2c.py`, `detection_m3c2.py` | Medium — three independent methods | Enable each method individually via `--set`; compare outputs |
| **7** | `runner.py` + reduce `scripts/run_workflow.py` to shim | Low — wiring only | Full end-to-end; parity tests |

---

## Test Plan (Acceptance Gate)

### Existing test baseline (must stay green)

```bash
uv run pytest -q -s tests/test_config_integration.py
uv run pytest -q -s tests/test_icp_registration.py
uv run pytest -q -s tests/test_streaming_integration.py
uv run pytest -q -s tests/  # full suite
```

### Test import migration

Three tests in `test_icp_registration.py` (lines 163, 178, 193) import `resolve_subsample_count`
from `run_workflow`. These must be migrated to import from `workflow.alignment` instead.
Keep a backward-compatible re-export in the shim during the transition if needed.

### New CLI parity tests

- Parser produces identical `Namespace` for all 15 flags.
- `build_cli_overrides` output matches current behavior.
- `--show-config` prints resolved YAML and exits.
- `--m3c2-normal-scale` without `--m3c2-radius` raises parser error.

### New workflow-phase tests

- Area selection and year filtering (mock discovery).
- Base-dir validation and discovery failure cases.
- Clipping validation paths (missing boundary file, empty after clip).
- Reference-direction selection (t1 vs t2).
- Streaming vs. in-memory routing for DoD, C2C, M3C2 (monkeypatch detector calls).

### Export-parity tests

- Lock down current file naming patterns for all methods.
- Verify current per-method output roots (flat vs. area-scoped).
- Verify single-run CRS caching (detect once, reuse everywhere).
- Run-input manifest contents and source ordering.

### End-to-end smoke tests

```bash
# Subprocess: shim still works
uv run scripts/run_workflow.py --show-config

# Full synthetic pipeline
uv run scripts/generate_synthetic_laz.py
uv run scripts/run_workflow.py --config config/profiles/synthetic.yaml

# Lightweight smoke: discovery + manifest only, no heavy compute
uv run scripts/run_workflow.py --config config/profiles/synthetic.yaml \
  --set alignment.enabled=false \
  --set detection.dod.enabled=false \
  --set detection.c2c.enabled=false \
  --set detection.m3c2.enabled=false
```

---

## Repo-Specific Risks and Mitigations

| Risk | Mitigation |
|:---|:---|
| `__init__.py` eagerly imports all subpackages — adding `workflow` there would trigger import side effects | **Do not** add `workflow` to `__init__.py` or `__all__`. Import submodules directly. |
| Tests import `resolve_subsample_count` from `scripts/run_workflow.py` | Migrate imports to `workflow.alignment`; optionally keep a thin re-export in the shim. |
| Output-path inconsistencies between methods | Freeze and document, don't normalize. Add export-parity tests. |
| Scattered `return` statements as error handling | Replace with `WorkflowAbort`; catch once in `runner.py`. |
| Subtle state mutation bugs during extraction | Typed result objects at phase boundaries make contracts explicit. |
| `python scripts/run_workflow.py` requires `src` path bootstrap | Keep the `sys.path.append` in the shim. |

---

## Maintainer Note (To Include in Codebase)

After refactoring, add a brief note to `workflow/__init__.py`:

> **Boundary rule**: New workflow coordination logic belongs in
> `terrain_change_detection.workflow`, not in `scripts/run_workflow.py`.
> The script is a compatibility shim only.
