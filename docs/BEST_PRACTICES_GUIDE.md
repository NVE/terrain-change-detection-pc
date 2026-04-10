# Terrain Change Detection Best Practice Guide

This guide helps you choose good starting settings, decide when to override them, and judge whether an alignment or change-detection run is trustworthy.

This is not the full parameter reference. For the YAML details, see [Configuration Guide](CONFIGURATION_GUIDE.md). For the repo runs that support the recommendations here, see [Best-Practice Evidence](BEST_PRACTICES_EVIDENCE.md).

When this guide talks about areas you expect to be unchanged, it means surfaces such as roads, bedrock, or undisturbed terrain that should line up in both epochs.

## How To Use This Guide

1. Pick the dataset type that matches your project.
2. Run one pilot area before scaling up.
3. Change one setting family at a time.
4. Keep the seed fixed once a pilot behaves the way you want.
5. Save the exact config used for the accepted run.

## Quick Start By Dataset Type

### Dense Drone Data Like `Jeksla`

Start with:

- `config/profiles/drone.yaml`
- For the repo datasets, add `--set paths.base_dir=data`
- Keep ICP enabled
- Start with `alignment.max_correspondence_distance=2.0`
- Keep M3C2 on header autotune
- Start with `detection.m3c2.core_points_percent=10.0`

Looks good when:

- Most of the overlap is retained
- The autotuned M3C2 radius is roughly `0.5` to `2.0` m, matching the high point density
- The M3C2 point output covers areas you expect to be unchanged without large gaps

Change first if needed:

- If the output is speckled or patchy, increase `radius` and `normal_scale` together
- If the output is too smooth, lower `radius` and `normal_scale` together
- Leave the core-point percentage alone until the radius and depth produce a clean output

### Larger Survey Data Like Døli Area Data from hoydedata.no

Start with:

- `config/profiles/large_scale.yaml`
- For a single-area pilot that fits in memory, temporarily add `--set outofcore.enabled=false --set parallel.enabled=false`
- Keep M3C2 on header autotune
- Start with `detection.m3c2.core_points_percent=5.0`

Looks good when:

- The chosen core-point count is much lower than the full ground count
- The autotuned M3C2 radius is roughly `2` to `5` m, matching the lower point density
- At least 95% of core points produce valid M3C2 distances

Change first if needed:

- If the pilot is still too slow, reduce `core_points_percent` first
- If narrow banks, scarps, or edges blur together, lower `radius` and `normal_scale`

### Synthetic Or Controlled Validation Data

Start with:

- `config/profiles/synthetic.yaml`
- Use `alignment.subsample_size=30000` for ICP
- Use M3C2 header autotune
- Use `core_points_percent=100.0` while tuning
- Drop to `10.0` for faster check runs once the settings are stable

Looks good when:

- Repeated runs with the same seed match
- Header and sample autotune choose almost the same radius
- Reducing core density to `10%` keeps similar mean and std values

Change first if needed:

- If a lower ICP sample changes the alignment noticeably, increase the ICP sample
- If a smaller fixed M3C2 radius makes the result much noisier, go back to header autotune

## Understanding The M3C2 Distance Output

M3C2 distances are signed values measured along the local surface normal at each core point:

- **Positive distance**: the T2 surface is above the T1 surface (deposition, heave, or accumulation)
- **Negative distance**: the T2 surface is below the T1 surface (erosion, subsidence, or removal)
- **NaN / invalid**: not enough neighbors were found in one or both epochs to compute a reliable distance

The output also includes an `uncertainty` field per core point. A distance is statistically significant when its absolute value exceeds the uncertainty at the configured confidence level (default 95%).

## Understanding The Alignment RMSE In The Log

The workflow log reports RMSE values at different stages. These values depend on the target point density and are **not directly comparable across stages or runs with different subsample sizes**.

- **Pre-ICP RMSE** (shown when coarse alignment is enabled): nearest-neighbor RMSE against the full reference cloud.
- **Post-ICP RMSE**: nearest-neighbor RMSE against the subsampled reference cloud used for ICP.

A denser target produces smaller nearest-neighbor distances regardless of alignment quality. For example, on the Jeksla drone data, the pre-ICP RMSE of `0.324` m (against 2.7M T1 points) and the post-ICP RMSE of `1.014` m (against 50K T1 points) reflect different target densities, not a degradation from ICP.

To judge whether ICP actually improved the alignment, compare M3C2 outputs with ICP enabled vs disabled — not the RMSE values from different pipeline stages.

## ICP Alignment Best Practices

### When ICP Should Run

Use ICP whenever the two epochs come from different flights, sensors, or processing chains. Keep `alignment.enabled=true` and keep `overlap_filter=true` unless you have a very specific reason not to.

Only skip ICP after a pilot shows that:

- The transform is negligible
- Areas you expect to be unchanged look the same with and without ICP
- The overall change map (spatial pattern and magnitude) does not change when ICP is disabled

If disabling ICP introduces visible shifts or non-zero mean distances on stable areas, the epochs are not well enough aligned to skip it.

### How To Choose The ICP Sample Size

For real projects, start with:

- `subsample_mode=count`
- `subsample_size=50000`

This is the safer starting point for final production runs. In the synthetic verification, a much smaller 10% sample (`6,250` points) ran 5x faster but produced a slightly higher validation RMSE (`0.711` vs `0.693`). On Jeksla drone data, reducing from `50,000` to `25,000` increased the M3C2 distance std by about 5%.

Use a smaller sample only for quick screening runs. If reducing the sample noticeably changes the downstream M3C2 output (e.g., higher std or different spatial patterns), the smaller sample is not reliable enough for a final run.

Change first if needed:

- If the result changes across seeds or sample sizes, increase the sample count
- If the pilot is too slow, try one lower step such as `25000`
- Avoid changing sample size and correspondence distance at the same time

### When To Use Coarse Alignment

Turn on coarse alignment when you can see a clear XY shift, partial overlap, or uncertain starting placement. For most of those cases, start with:

- `alignment.coarse.enabled=true`
- `alignment.coarse.method=phase`

Use `centroid` when the shift looks simple and mainly translational. Treat `pca` and `open3d_fpfh` as recovery options after simpler starts fail, not as the first choice.

Coarse alignment is helping when ICP converges faster or the M3C2 output improves afterward. If it adds no visible improvement or makes the overlap worse, disable it.

### How To Choose `max_correspondence_distance`

Use a small but realistic search distance:

- Dense drone data: start at `2.0`
- Larger survey data: start at `1.0`

These are the values used in the `drone.yaml` and default profiles respectively, and they worked well in the verification runs.

The distance is too tight when ICP finds too few correspondences or converges after only a few iterations without improving. It is too loose when stable areas still show a visible systematic offset in the M3C2 output even though ICP reported convergence.

Change first if needed:

- Move one step only, for example from `1.0` to `1.5`
- Do not change this together with coarse alignment and sample size in the same run

### When The Reference Epoch Matters

Keep the default `reference=t1` for normal monitoring work. That keeps the earlier epoch fixed and aligns the later epoch to it.

Use `reference=t2` only when the workflow requires the later epoch to stay fixed. In the synthetic verification, both directions produced nearly identical alignment quality, so this is usually an operational choice rather than a quality fix.

If swapping the reference direction noticeably changes the M3C2 output, investigate the alignment stability (sample size, overlap) before interpreting the change map.

### What "Good Enough" Looks Like After ICP

- The same seed and same inputs reproduce the same result (deterministic)
- The overlap filter retains most of the point cloud (above 95% for well-overlapping datasets)
- Changing the ICP sample size by a factor of two does not noticeably change the M3C2 output
- Stable areas (roads, bedrock) show near-zero M3C2 distances
- Switching the reference epoch produces a similar M3C2 distance distribution

## M3C2 Best Practices

A key concept in M3C2 is the **feature scale** — the spatial size of the terrain changes you want to detect. A `radius` of `1.0` m averages over a 1 m neighborhood, so it can resolve changes at roughly that scale. Smaller radii detect finer features but are noisier; larger radii are smoother but blur small features.

### When To Trust Autotune

Use header autotune first:

- `detection.m3c2.use_autotune=true`
- `detection.m3c2.autotune.source=header`

That is the best starting point for a new site. In the synthetic verification, header autotune and sample autotune chose almost identical radii and produced the same summary statistics.

Header autotune is especially useful when:

- You are running a new site for the first time
- You want the same scale choice across in-memory and streaming runs
- You want a stable first answer before freezing manual settings

### Autotune Radius Bounds

Autotune clamps the computed radius to the `min_radius` and `max_radius` bounds set in the config profile. If the computed radius falls outside these bounds, the clamped value is used instead.

For example, the `large_scale.yaml` profile sets `min_radius=2.0`. If autotune computes a radius of `1.5` m, it will be clamped up to `2.0` m. Check the autotuned values in the log (visible at DEBUG level) to see whether clamping occurred. If the clamped value does not match the spatial scale of the changes you want to detect, adjust the bounds or switch to fixed parameters.

### When To Override Autotune

Switch to fixed settings only after a pilot run shows what radius produces good results. Manual settings make the most sense when:

- You are repeating the same site over time and want strict comparability
- You know the spatial scale of the changes you want to detect
- You need to lock the neighborhood definition for reporting consistency

When you freeze the values, set:

- `use_autotune=false`
- `fixed.radius`
- `fixed.normal_scale`
- `fixed.depth_factor`

If you keep changing the fixed values from run to run, you lose the main benefit of fixing them.

### How To Choose Core-Point Density

Use core-point density mainly as a runtime and output-density control.

Start with:

- `100%` for parameter studies or small validation areas
- `10%` for dense drone operational runs
- `5%` for larger survey operational runs

Those starting points were confirmed in the verification runs:

- `Jeksla`: `10%` produced `272,225` core points and a usable output
- Larger survey pilot: `5%` produced `296,866` core points while keeping broad coverage
- Synthetic sweep: lowering from `100%` to `10%` kept very similar summary statistics while greatly reducing output size

Increase the percentage when the output becomes too sparse to interpret. Decrease it when the pilot is too slow and the broad pattern is already clear.

### How To Think About `radius`, `normal_scale`, And `depth_factor`

When you manually override M3C2, start from the radius that autotune chose in your pilot and keep the scales tied together:

- Set `normal_scale` equal to `radius` (this is also the default when `normal_scale` is left as `null`)
- Keep `depth_factor` close to the value used in your pilot

The synthetic sweep is the clearest example:

- Header autotune picked `4.5 m` and produced a std of `0.071 m` with nearly 100% valid points
- Forcing the radius down to `2.0 m` tripled the std to `0.266 m` and dropped valid coverage to 97.5%
- Forcing the radius up to `8.0 m` reduced the std slightly to `0.069 m` but began to blur fine detail

Practical rule:

- If the output is speckled or many points become invalid, increase `radius` and `normal_scale` together
- If the output is too smooth and small features blur away, decrease them together
- Adjust by roughly 50% at a time (e.g., from `1.0` to `1.5`, not from `1.0` to `4.0`)

Only override when you have a specific reason — for example, you know the terrain features you care about are roughly 1 m wide, so you want a radius near `1.0` m. Do not override just because the autotuned value "looks large."

When using `--m3c2-radius` from the CLI without specifying `--m3c2-depth-factor`, the depth factor falls back to `autotune.max_depth_factor` from the active profile. For example, the `drone.yaml` profile sets `max_depth_factor=0.6`, so `--m3c2-radius 2.0` would produce `max_depth=1.2` m (2.0 × 0.6). Check the log to confirm the resolved parameters match your intent.

## Inspecting Results

The workflow exports M3C2 results as LAZ point clouds and optionally as GeoTIFF rasters.

### In A GIS Tool (QGIS, ArcGIS)

- Load the M3C2 `.laz` file and color by the `distance` field. Use a diverging color ramp (e.g., red–white–blue) centered on zero.
- Look for systematic patterns in areas you expect to be unchanged — a uniform non-zero shift suggests an alignment issue, not real change.
- Load the GeoTIFF raster for a gridded overview. Check that it covers the expected spatial extent.

### Using The Built-In Visualization

- Add `--show-plots` to display interactive Plotly plots during the run. These show histograms and 3D scatter plots colored by distance.
- The histogram should be centered near zero for stable terrain. A bimodal or shifted histogram suggests mixed stable and changed areas.

### Quick Statistics From The Log

Run with `--set logging.level=DEBUG` to see the M3C2 summary line in the log:

```
M3C2 completed: n=272225 (valid=252722), mean=-0.0064 m, median=-0.0065 m, std=0.0595 m
```

The `valid` count tells you how many core points had enough neighbors in both epochs. A low valid percentage (below ~80%) suggests the radius or depth is too small, or the point clouds have poor overlap.

## Using Clipping To Focus The Analysis

When working with large areas, clip the analysis to a polygon of interest (e.g., a river bank, a construction zone) to reduce runtime and focus the output.

```yaml
clipping:
  enabled: true
  boundary_file: "path/to/boundary.geojson"
```

Or from the CLI:

```bash
--set clipping.enabled=true --set clipping.boundary_file=path/to/boundary.geojson
```

Clipping is applied before alignment and detection. This means:

- The ICP alignment uses only points within the clipped region
- M3C2 core points are selected only from the clipped area
- Outputs cover only the clipped extent

Use clipping when the area of interest is much smaller than the full survey extent, or when edge effects at the survey boundary are causing problems.

## Validation Checklist Before Accepting A Run

- Confirm the workflow selected the correct area and time periods
- Confirm the overlap and alignment sample sizes make sense for the dataset
- Record the chosen M3C2 radius, max depth, and core-point count from the log
- Open the M3C2 point output or raster and inspect both areas you expect to be unchanged and known change zones
- Repeat one pilot run with the same seed if the work needs a fixed baseline
- Save the exact config and seed used for the accepted run

## Scaling Up To Large Datasets

### When To Enable Out-Of-Core Processing

Keep `outofcore.enabled=false` (in-memory) for:

- Pilot runs on single areas or small datasets
- Datasets that fit comfortably in RAM (typically under ~5M ground points per epoch)

Switch to `outofcore.enabled=true` when:

- The dataset exceeds available RAM
- You are processing multi-tile areas (e.g., the full Ristvassdrag site)
- The workflow crashes with memory errors

### Tile Size And Halo

- `tile_size_m=500.0` is the default and works well for most datasets.
- The `halo_m` parameter adds a buffer around each tile to prevent edge artifacts. It defaults to the M3C2 radius. Only increase it if you see discontinuities at tile boundaries.
- Smaller tiles use less memory per tile but increase I/O and duplicate computation in the halo overlap zones.

### Parallel Processing

Enable `parallel.enabled=true` to process tiles concurrently. Start with the default worker count (all cores minus one). If memory becomes an issue, reduce `n_workers` or set `memory_limit_gb`.

GPU mode and parallel mode are mutually exclusive. For multi-tile GPU workflows, process tiles sequentially with GPU acceleration instead.

## Common Failure Modes

### No Drone Data Are Discovered

This usually means `base_dir` points to an area folder instead of the parent folder, or the source type is wrong.

Change first:

- For drone-style repo data, set `paths.base_dir` to the parent folder such as `data`
- Keep `discovery.source_type=drone`

### ICP Reports Too Few Correspondences Or Stops Too Early

This usually means the correspondence distance is too tight, the overlap is too small, or the starting offset is too large.

Change first:

- Increase `max_correspondence_distance` one step
- If that is not enough, enable coarse alignment

### Alignment Looks Unstable Across Seeds Or Sample Sizes

This usually means the ICP sample is too small or the stable overlap is too weak.

Change first:

- Increase the alignment sample size
- Keep `overlap_filter=true`

### M3C2 Output Is Speckled Or Full Of Holes

This usually means the neighborhood is too small or the core-point density is too sparse.

Change first:

- Return to header autotune
- Or increase `radius` and `normal_scale` together

### M3C2 Output Is Too Smooth

This usually means the M3C2 radius is too large relative to the changes you want to detect.

Change first:

- Lower `radius` and `normal_scale` together

### Large Pilot Runs Are Too Slow

This usually means the pilot is still evaluating too many core points or generating too many outputs.

Change first:

- Lower `core_points_percent`
- Turn off extra outputs for the pilot
- Re-enable them for the final production run

## Example Commands For This Repo

### Dense Drone Pilot (`Jeksla`)

```bash
uv run scripts/run_workflow.py \
  --config config/profiles/drone.yaml \
  --set paths.base_dir=data \
  --set detection.dod.enabled=false \
  --set detection.c2c.enabled=false \
  --set detection.m3c2.export_raster=false \
  --area-name Jeksla \
  --years 2024 2025
```

### Larger Survey Pilot (`eksport_1225654_20250602`)

```bash
uv run scripts/run_workflow.py \
  --config config/profiles/large_scale.yaml \
  --set outofcore.enabled=false \
  --set parallel.enabled=false \
  --set detection.dod.enabled=false \
  --set detection.c2c.enabled=false \
  --set detection.m3c2.export_raster=false \
  --area-name eksport_1225654_20250602 \
  --years 2015 2020
```

If the project no longer fits comfortably in memory, remove the `outofcore.enabled=false` and `parallel.enabled=false` overrides so the same profile can switch back to tiled processing.

### Controlled Synthetic Validation

```bash
uv run scripts/run_workflow.py \
  --config config/profiles/synthetic.yaml \
  --set detection.dod.enabled=false \
  --set detection.c2c.enabled=false \
  --set detection.m3c2.export_raster=false \
  --area-name synthetic_area
```
