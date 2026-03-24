# Integrate M3C2-EP Via py4dgeo

## Summary

The repository already has a solid M3C2 integration surface built around `py4dgeo`, but it only exposes the vanilla algorithm. The goal is to add M3C2 with error propagation in a way that fits the existing detector/config/workflow architecture instead of creating a parallel implementation path.

The first shippable version should be **phased**:

1. Add **in-memory M3C2-EP** as a first-class M3C2 variant.
2. Reuse the current alignment, export, and visualization pipeline.
3. Explicitly defer streaming/out-of-core and repository-level parallel EP until the missing metadata plumbing is in place.

This is the right initial scope because `py4dgeo.M3C2EP` requires inputs that the current codebase does not yet model end-to-end:

- per-point scan position IDs
- scan-position uncertainty metadata (`scanpos_info`)
- a 12x12 registration covariance matrix

In addition, `py4dgeo.M3C2EP` already uses its own multiprocessing internally, so we should not stack the repository's tile-level multiprocessing around it until compatibility is proven.

For practical ALS/Hoydedata usability, v1 should not require manually authored sidecars as the only metadata path. The default should be:

- synthesize scan-position metadata from grouped `point_source_id` values for ALS datasets
- auto-estimate `Cxx` from the final ICP solution
- allow sidecars and file-based covariance as higher-fidelity overrides

## Current State

### What already exists

- `src/terrain_change_detection/detection/m3c2.py` contains:
  - `M3C2Params`
  - `M3C2Result`
  - `compute_m3c2_original(...)`
  - streaming/tiled M3C2 execution paths that reuse the vanilla wrapper
- `scripts/run_workflow.py` already has a complete M3C2 workflow branch with:
  - core-point selection
  - autotune/fixed parameter selection
  - streaming vs in-memory dispatch
  - export to LAZ and GeoTIFF
- `src/terrain_change_detection/utils/export.py` already supports extra scalar fields such as:
  - `uncertainty`
  - `significant`
- `py4dgeo 0.7.0` is installed in the repo environment and exposes:
  - `M3C2`
  - `M3C2EP`

### What is missing

- No first-class M3C2 variant selector in config.
- No `compute_m3c2_ep(...)` detector method.
- No scan-position metadata ingestion.
- No registration covariance model or artifact.
- No EP-specific tests.
- No supported workflow path for significance flags or richer EP result fields.
- Streaming readers currently return only XYZ and drop attributes like `point_source_id`.

### Important backend facts discovered during planning

- `py4dgeo.M3C2EP` requires:
  - `tfM`: a 3x4 transform
  - `Cxx`: a 12x12 covariance matrix
  - `refPointMov`: reduction point
  - `scanpos_id` per point
  - `scanpos_info` per epoch
- `scanpos_info` entries must contain:
  - `origin`
  - `sigma_range`
  - `sigma_scan`
  - `sigma_yaw`
- `py4dgeo.M3C2EP` does **not** return a significance mask directly.
- `py4dgeo.M3C2EP` computes a LoD-like threshold internally at **95% confidence**.
- `py4dgeo.M3C2EP` launches multiprocessing internally, which means EP is not a good candidate for the repo's existing tile-parallel path in v1.

## Design Goals

- Keep M3C2 as one subsystem with multiple variants, not two separate detectors.
- Preserve current vanilla M3C2 behavior and defaults.
- Make EP outputs available through the same workflow/export surface as vanilla M3C2.
- Make EP usable on ALS/Hoydedata data without requiring manual metadata authoring.
- Fail clearly when EP-specific scientific inputs are missing.
- Avoid pretending unsupported combinations are available.

## Non-Goals For The First Delivery

- No full out-of-core M3C2-EP support.
- No tile-parallel M3C2-EP support.
- No custom reimplementation of EP math outside `py4dgeo`.
- No silent fallback to zero alignment covariance. Zero `Cxx` remains an explicit source choice.
- No user-facing config knobs for backend behavior that `py4dgeo 0.7.0` does not actually expose.

## Public Interface Changes

### 1. Config model changes

Add an explicit M3C2 variant selector:

```yaml
detection:
  m3c2:
    variant: original  # original | ep
```

Add an EP subsection:

```yaml
detection:
  m3c2:
    ep:
      scan_metadata_source: auto
      auto_discover_from_metadata_dir: true
      scan_positions_t1_path: null
      scan_positions_t2_path: null
      cxx_source: icp_estimate
      alignment_covariance_path: null
      export_scalar_fields: true
```

#### Meaning of the new fields

- `variant`
  - Selects vanilla M3C2 vs M3C2-EP.
  - Default remains `original` so existing workflows do not change.
- `scan_metadata_source`
  - Controls how EP scan-position metadata is obtained.
  - `auto`: prefer explicit sidecars, then metadata-dir discovery, then ALS synthesis from `point_source_id`.
  - `sidecar`: require explicit or discovered sidecar files.
  - `synthetic_from_point_source_id`: build pragmatic ALS metadata directly from filtered point groups.
- `auto_discover_from_metadata_dir`
  - When enabled, try to find scan-position sidecars in each dataset's discovered `metadata/` directory.
- `scan_positions_t1_path`, `scan_positions_t2_path`
  - Optional explicit sidecar files for scan-position uncertainty data.
  - These take precedence over auto-discovery.
- `cxx_source`
  - Controls how the 12x12 alignment covariance is obtained.
  - `icp_estimate`: estimate from the final ICP correspondences.
  - `file`: load from `alignment_covariance_path`.
  - `zero`: explicit no-alignment-uncertainty fallback.
- `alignment_covariance_path`
  - Optional path to a stored 12x12 registration covariance matrix.
- `export_scalar_fields`
  - Controls whether EP-only scalar outputs are included in LAZ export.

### 2. Detector API changes

Add a variant-dispatching entrypoint:

- `ChangeDetector.compute_m3c2(...)`
- `M3C2Detector.compute_m3c2(...)`

Keep the existing explicit methods:

- `compute_m3c2_original(...)`
- add `compute_m3c2_ep(...)`

The dispatcher should decide which implementation to run based on config or an explicit variant argument.

### 3. Result model changes

Keep `M3C2Result` as the common outward-facing result type, but extend it to support EP cleanly.

#### Preserve existing common fields

- `core_points`
- `distances`
- `uncertainty`
- `significant`
- `metadata`

#### Add EP detail payload

Add an EP-specific typed payload, for example `ep_details`, containing:

- `lodetection`
- `spread1`
- `spread2`
- `num_samples1`
- `num_samples2`
- `covariance1`
- `covariance2`

#### Compatibility rule

- `uncertainty` remains the primary generic scalar field.
- For EP, `uncertainty` should alias `lodetection`.
- For vanilla M3C2, keep the current behavior of extracting the best available scalar uncertainty from py4dgeo outputs.

## Data Model And Metadata Changes

### 1. Scan ID handling

Use LAS `point_source_id` as the raw scan identifier because:

- it is already present in sample LAZ files
- it survives typical LAS workflows better than inferred grouping logic
- it is semantically closer to acquisition-source identity

However, raw `point_source_id` values cannot be passed directly to `py4dgeo.M3C2EP`, because py4dgeo expects contiguous 1-based scan IDs that index into `scanpos_info`.

#### Required behavior

- Build a raw-to-normalized mapping per epoch:
  - raw IDs like `444`, `445`, `10`
  - normalized IDs like `1`, `2`, `3`
- Store normalized values in the `scanpos_id` additional dimension passed to `py4dgeo.Epoch`
- Keep the raw IDs in metadata for traceability and debug logging

### 2. Scan-position sidecar format

Add a sidecar format for scan-position uncertainty metadata keyed by raw scan/source IDs.

Recommended schema:

```yaml
444:
  origin: [x, y, z]
  sigma_range: 0.01
  sigma_scan: 0.01
  sigma_yaw: 0.01
445:
  origin: [x, y, z]
  sigma_range: 0.01
  sigma_scan: 0.01
  sigma_yaw: 0.01
```

JSON with the same structure should also be accepted.

#### Loader responsibilities

- Read YAML or JSON.
- Validate required keys and numeric types.
- Validate that every scan ID used by the filtered epoch data is present.
- Normalize raw IDs to the contiguous 1-based list structure expected by py4dgeo.

### 3. Synthetic ALS scan metadata

Because Hoydedata-style ALS datasets do not typically ship with explicit per-scan uncertainty sidecars, v1 should support a pragmatic default synthesis path.

Recommended v1 behavior:

- group each epoch by raw `point_source_id`
- create one synthetic scanner entry per raw source ID, not one scanner for the whole epoch
- derive each synthetic scanner origin from that group's spatial footprint in the epoch
- set the scanner Z above the group's local maximum elevation using a configurable/default ALS altitude heuristic
- assign conservative default `sigma_range`, `sigma_scan`, and `sigma_yaw` values suitable for ALS
- normalize the raw source IDs to contiguous 1-based `scanpos_id` values before passing them to py4dgeo

This synthetic path should be the default ALS fallback when sidecars are not available. Sidecars remain the preferred override for TLS, drone, or higher-fidelity scan metadata.

### 4. Registration covariance source

Add support for both estimated and persisted covariance inputs for alignment uncertainty.

Recommended v1 behavior:

- default `cxx_source = "icp_estimate"`
- alternative `cxx_source = "file"` with plain text or NumPy 12x12 matrix input
- explicit `cxx_source = "zero"` fallback for experiments or no-alignment cases

The persisted covariance artifact should be independent from the current 4x4 transform file so the covariance model can evolve separately.

### 5. ICP covariance estimation

Add an `estimate_alignment_covariance(...)` helper to the ICP backend that derives a 12x12 covariance estimate from the final correspondences and residuals.

Expected implementation shape:

- use the final registration transform and matched source/target pairs
- accumulate `J^T J` incrementally instead of materializing a full dense Jacobian
- estimate residual variance from the final fit
- compute a symmetric 12x12 covariance with pseudo-inverse safeguards

This estimate should be the default EP covariance source, with file and zero as explicit alternatives.

## Detector Implementation Plan

### 1. Refactor common M3C2 epoch construction

Create shared internal helpers in `m3c2.py` for:

- constructing py4dgeo `Epoch` objects
- selecting normal radii
- populating metadata consistently
- packaging backend outputs into `M3C2Result`

This avoids duplicating the current py4dgeo setup between original and EP variants.

### 2. Implement `compute_m3c2_ep(...)`

The EP method should:

1. Validate non-empty inputs.
2. Resolve EP scan metadata:
   - explicit/discovered sidecars when configured
   - otherwise synthesize ALS metadata from grouped `point_source_id`
3. Resolve `Cxx`:
   - estimate from ICP by default
   - otherwise load from file or use explicit zero matrix
3. Build py4dgeo epochs with:
   - `cloud`
   - `additional_dimensions` containing normalized `scanpos_id`
   - `scanpos_info`
4. Initialize `py4dgeo.M3C2EP` with:
   - `epochs`
   - `corepoints`
   - `cyl_radius`
   - `max_distance`
   - `normal_radii`
   - `tfM`
   - `Cxx`
   - `refPointMov`
   - `perform_trans`
5. Convert backend outputs into `M3C2Result`.

### 3. EP transform strategy

The default EP execution path should use the raw, untransformed T2 epoch and let `py4dgeo.M3C2EP` apply the transform internally:

- `perform_trans=True`
- `tfM = transform_matrix[:3, :]`
- `cloud_t2 = raw epoch-2 points in the same local coordinate frame used during ICP`
- `refPointMov = centroid of raw epoch-2 points in local coordinates`

This is the cleanest v1 strategy because:

- the transform and the covariance describe the same alignment operation
- it matches the way `py4dgeo.M3C2EP` is designed to apply `tfM`
- it avoids baking the transform into the points before the EP backend sees it

Retain an internal comparison test that checks the `perform_trans=True` path against a pre-aligned `perform_trans=False` path on synthetic data, but do not make the latter the production default.

### 4. EP significance rule

Populate:

- `uncertainty = lodetection`
- `significant = abs(distances) > lodetection`

This significance mask is not returned by py4dgeo directly, so the repo must compute it after the backend call.

### 5. Metadata packaging

EP metadata should include at least:

- `variant: "ep"`
- `n_valid`
- summary stats for distances
- covariance source used (`icp_estimate`, `file`, or `zero`)
- scan metadata source used (`sidecar` or `synthetic_from_point_source_id`)
- scan ID coverage counts
- whether transform was applied externally or internally
- fixed 95% confidence note

## Workflow Integration Plan

### 1. Replace ad hoc M3C2 branching with variant dispatch

In `scripts/run_workflow.py`, the M3C2 branch should:

- keep existing core-point selection logic
- keep existing autotune/fixed-parameter logic
- keep current streaming/original dispatch for vanilla M3C2
- choose EP only when `cfg.detection.m3c2.variant == "ep"`

### 2. EP input assembly in workflow

For in-memory EP runs, assemble:

- `core_src`
- `points1`
- raw `points2` before applying the ICP transform
- `transform_matrix`
- dataset metadata directories from `DatasetInfo.metadata_dir`
- `point_source_id` arrays for both epochs
- explicit/discovered sidecars or synthesized ALS scan metadata
- covariance matrix from `icp_estimate`, `file`, or `zero`

### 3. Hard gating for unsupported combinations

When `variant == "ep"`, fail fast with clear errors for:

- `outofcore.enabled == true`
- `parallel.enabled == true`
- missing `point_source_id`
- `scan_metadata_source == "sidecar"` with no usable sidecars
- `cxx_source == "file"` with no usable covariance file
- `cxx_source == "icp_estimate"` when covariance estimation fails and no alternate source was selected

These should be explicit validation errors, not silent downgrades to vanilla M3C2.

### 4. No hidden fallback to original M3C2

If the user asked for EP and EP prerequisites are missing, the workflow should fail clearly instead of quietly computing vanilla M3C2.

That keeps scientific meaning explicit and avoids misleading outputs.

## Preprocessing And I/O Changes

### 1. Preserve EP-relevant attributes through clipping

`src/terrain_change_detection/preprocessing/clipping.py` currently preserves several attributes but not `point_source_id`.

Add preservation of:

- `point_source_id`

Without this, clipped datasets cannot support EP later.

### 2. Keep aligned-file attribute preservation

`src/terrain_change_detection/alignment/streaming_alignment.py` already copies arbitrary point attributes when transforming files.

No conceptual redesign is needed there, but EP validation should confirm that `point_source_id` survives transformed-file workflows as expected.

### 3. Defer streaming reader redesign to Phase 2

`LaspyStreamReader` currently yields only XYZ arrays.

That is acceptable for vanilla M3C2, but not for EP. The first delivery should avoid forcing a large reader redesign. Instead:

- keep EP in-memory only for v1
- plan a later extension where streaming readers can yield structured chunks with XYZ plus `point_source_id`

## Export And Visualization Changes

### 1. LAZ export

Continue using `export_points_to_laz(...)` and add EP scalar fields via `extra_dims`.

Export for EP:

- `distance`
- `uncertainty` (LoD)
- `significant`
- `spread1`
- `spread2`
- `num_samples1`
- `num_samples2`

Do **not** attempt to export full 3x3 covariance tensors into LAZ in the first delivery.

### 2. Raster export

Keep GeoTIFF export based on distances only in v1.

Do not add EP-specific raster products yet. If needed later, add a separate export option for LoD/significance rasters once semantics are clear.

### 3. Visualization

Keep the current M3C2 visualizations.

Optional low-cost additions:

- histogram of significant-only EP distances
- summary log line: significant count and percentage

These should only be added after the detector result model is stable.

## Phased Delivery

### Phase 1: First shippable M3C2-EP integration

Deliver:

- config model for M3C2 variant + EP subsection
- sidecar loading and validation
- synthetic ALS scan-metadata generation from grouped `point_source_id`
- ICP-based covariance estimation
- scan ID normalization
- in-memory `compute_m3c2_ep(...)`
- workflow integration
- scalar export fields
- docs updates
- tests

Do not deliver:

- streaming EP
- parallel EP
- tile-wise EP

### Phase 2: Sequential streaming EP

Only after Phase 1 is stable:

- extend stream readers to retain `point_source_id`
- define structured chunk contracts for EP
- add sequential tiled EP path
- verify equivalence with in-memory EP on shared core points

### Phase 3: Parallel/out-of-core EP reassessment

Revisit only after proving:

- metadata plumbing works in streaming mode
- `py4dgeo.M3C2EP` multiprocessing does not conflict with repo-level orchestration
- performance warrants the added complexity

If multiprocessing conflicts remain, keep EP sequential rather than forcing unstable nested parallelism.

## Testing Plan

### 1. Config tests

Add tests for:

- default `variant == "original"`
- EP config parsing
- explicit path overrides
- validation failures for malformed EP config

### 2. Sidecar metadata tests

Add unit tests for:

- YAML/JSON loading
- required key validation
- raw-to-normalized scan ID mapping
- missing scan ID coverage detection

### 3. Synthetic ALS metadata tests

Add unit tests for:

- grouping by raw `point_source_id`
- generation of one synthetic scanner entry per source ID
- contiguous 1-based `scanpos_id` normalization
- fallback behavior when sidecars are absent but ALS synthesis is allowed

### 4. ICP covariance tests

Add unit tests for:

- identity/near-identity transform gives near-zero or small covariance
- known transform produces finite symmetric 12x12 covariance
- pseudo-inverse safeguards behave sensibly for weak correspondence sets

### 5. Detector tests

Add dedicated M3C2 tests, since the current suite has none.

#### Original M3C2 tests

- basic shape and metadata checks
- identical-cloud sanity case
- regression coverage for current uncertainty extraction

#### EP tests

- result shape and metadata checks
- `uncertainty` populated from `lodetection`
- `significant` computed from `abs(distance) > lodetection`
- `spread1`, `spread2`, `num_samples1`, `num_samples2` present
- raw T2 + `perform_trans=True` integration works with ICP-estimated `Cxx`

### 6. Workflow integration tests

Add tests for:

- successful in-memory EP run with synthesized ALS metadata and estimated covariance
- successful in-memory EP run with sidecars and file-based covariance
- failure when `point_source_id` is unavailable
- failure when `scan_metadata_source == "sidecar"` and sidecars are missing
- failure when `cxx_source == "file"` and covariance file is missing
- failure when EP is requested with `outofcore.enabled=true`
- failure when EP is requested with `parallel.enabled=true`

### 7. Export tests

Extend export coverage to verify EP scalar fields are written correctly to LAZ:

- `uncertainty`
- `significant`
- `spread1`
- `spread2`
- `num_samples1`
- `num_samples2`

### 8. Clipping regression tests

Add tests ensuring clipping preserves `point_source_id` so EP remains possible on clipped datasets.

### 9. Environment-sensitive smoke tests

Because `py4dgeo.M3C2EP` uses multiprocessing internally and may fail in restricted environments, backend smoke tests should be:

- isolated
- clearly marked
- skippable when the environment cannot support the backend runtime model

## Documentation Changes

Update:

- `README.md`
- `docs/CONFIGURATION_GUIDE.md`
- `docs/KNOWN_ISSUES.md`
- `docs/CHANGELOG.md`

### Documentation content to add

- M3C2 variant selection
- required EP inputs
- ALS default metadata synthesis strategy
- covariance source options (`icp_estimate`, `file`, `zero`)
- sidecar file format
- limitation that EP v1 is in-memory only
- explanation that significance is derived from `LoD`
- note that EP confidence is fixed by backend behavior in the first version

### Documentation cleanup

The changelog currently contains historical statements that imply M3C2-EP is already integrated. Once EP is actually implemented, reconcile those notes with the real feature state so the repository is internally consistent.

## Risks And Mitigations

### Risk 1: Missing scan-position metadata in real datasets

**Mitigation**

- support explicit sidecar files
- support metadata-dir auto-discovery
- support default ALS synthesis from grouped `point_source_id`
- fail clearly only when the selected metadata mode cannot be satisfied

### Risk 2: Raw scan identifiers are not contiguous

**Mitigation**

- normalize raw `point_source_id` values to contiguous 1-based `scanpos_id`
- keep a traceable mapping in metadata/logging

### Risk 3: ICP covariance estimation is unstable or unavailable

**Mitigation**

- make `icp_estimate` the default but keep `file` and `zero` as explicit alternatives
- use pseudo-inverse and residual-quality safeguards
- save estimated covariance artifacts for reproducibility and review

### Risk 4: Nested multiprocessing instability

**Mitigation**

- do not enable repo-level parallel EP in v1
- keep EP on the in-memory path first

### Risk 5: Silent semantic drift between original and EP outputs

**Mitigation**

- keep one shared result model
- make `variant` explicit in metadata
- avoid hidden fallback from EP to original

## Concrete Implementation Order

1. Add config types and defaults for `detection.m3c2.variant` and `detection.m3c2.ep`.
2. Add EP sidecar loader plus synthetic ALS metadata generation from grouped `point_source_id`.
3. Add scan ID normalization utilities based on `point_source_id`.
4. Add `estimate_alignment_covariance(...)` to the ICP backend.
5. Refactor shared py4dgeo epoch/result packaging helpers in `m3c2.py`.
6. Implement `compute_m3c2_ep(...)` using raw T2 + `perform_trans=True`.
7. Add `compute_m3c2(...)` dispatcher on `M3C2Detector` and `ChangeDetector`.
8. Integrate EP path into `scripts/run_workflow.py` with hard validation gates.
9. Extend LAZ export wiring for EP scalar fields.
10. Preserve `point_source_id` in clipping.
11. Add detector/config/workflow/export regression tests.
12. Update README/config/docs/changelog.

## Acceptance Criteria

The first delivery is complete when all of the following are true:

- Users can select `detection.m3c2.variant: ep`.
- In-memory workflows can run M3C2-EP through `py4dgeo`.
- ALS/Hoydedata workflows can run EP without manual sidecar creation when `point_source_id` is present.
- EP outputs include:
  - distances
  - LoD-style uncertainty
  - significance mask
  - spread/sample-count scalar fields
- LAZ exports include the EP scalar outputs.
- Unsupported EP modes fail explicitly and early.
- Vanilla M3C2 behavior is unchanged for existing configs.
- Tests cover config parsing, detector behavior, workflow integration, exports, and clipping preservation of `point_source_id`.

## Assumptions

- `point_source_id` is available in the main target datasets and is the best raw scan identifier to build on.
- Sidecar scan-position metadata remains an override path, but the default ALS path synthesizes scan metadata from grouped `point_source_id` values.
- The current workflow's local coordinate frame is acceptable for EP v1 when paired with raw T2 + `perform_trans=True`.
- The repository will implement ICP-based covariance estimation rather than requiring users to prepare a 12x12 matrix manually.
- Confidence for EP significance remains fixed at the backend's current 95% behavior in the first version.
