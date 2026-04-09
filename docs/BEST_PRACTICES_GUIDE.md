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
- The chosen M3C2 radius stays in the sub-meter to low-meter range
- The M3C2 point output covers areas you expect to be unchanged continuously

Change first if needed:

- If the output is speckled or patchy, increase `radius` and `normal_scale` together
- If the output is too smooth, lower `radius` and `normal_scale` together
- Leave the core-point percentage alone until the neighborhood size looks sensible

### Larger Survey Data Like Døli Area Data from hoydedata.no

Start with:

- `config/profiles/large_scale.yaml`
- For a single-area pilot that fits in memory, temporarily add `--set outofcore.enabled=false --set parallel.enabled=false`
- Keep M3C2 on header autotune
- Start with `detection.m3c2.core_points_percent=5.0`

Looks good when:

- The chosen core-point count is much lower than the full ground count
- The chosen M3C2 radius lands around a few meters
- Valid core-point coverage stays high

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
- Lower core density keeps the same broad pattern

Change first if needed:

- If a lower ICP sample changes the alignment noticeably, increase the ICP sample
- If a smaller fixed M3C2 radius makes the result much noisier, go back to header autotune

## ICP Alignment Best Practices

### When ICP Should Run

Use ICP whenever the two epochs come from different flights, sensors, or processing chains. Keep `alignment.enabled=true` and keep `overlap_filter=true` unless you have a very specific reason not to.

Only skip ICP after a pilot shows that:

- The transform is negligible
- Areas you expect to be unchanged look the same with and without ICP
- The change map does not materially change when ICP is disabled

If disabling ICP changes areas you expect to be unchanged or known features, the data were not aligned well enough to skip it.

### How To Choose The ICP Sample Size

For real projects, start with:

- `subsample_mode=count`
- `subsample_size=50000`

This is the safer starting point for production-style work. In the synthetic verification, a much smaller 10% sample ran faster, but the alignment check was slightly worse than the larger sample.

Use a smaller sample only for quick pilots. If a smaller sample changes the alignment result in a meaningful way, it was too small for a decision run.

Change first if needed:

- If the result changes across seeds or sample sizes, increase the sample count
- If the pilot is too slow, try one lower step such as `25000`
- Avoid changing sample size and correspondence distance at the same time

### When To Use Coarse Alignment

Turn on coarse alignment when you can see a clear XY shift, partial overlap, or uncertain starting placement. For most of those cases, start with:

- `alignment.coarse.enabled=true`
- `alignment.coarse.method=phase`

Use `centroid` when the shift looks simple and mainly translational. Treat `pca` and `open3d_fpfh` as recovery options after simpler starts fail, not as the first choice.

Coarse alignment is helping when ICP starts from a sensible position and converges cleanly. It is not helping when it adds no visible value or makes overlap worse.

### How To Choose `max_correspondence_distance`

Use a small but realistic search distance:

- Dense drone data: start at `2.0`
- Larger survey data: start at `1.0`

These were the working starting points in the repo-backed runs.

The distance is too tight when ICP cannot find enough correspondences or stops too early. It is too loose when the alignment still looks smeared even though the run converged numerically.

Change first if needed:

- Move one step only, for example from `1.0` to `1.5`
- Do not change this together with coarse alignment and sample size in the same run

### When The Reference Epoch Matters

Keep the default `reference=t1` for normal monitoring work. That keeps the earlier epoch fixed and aligns the later epoch to it.

Use `reference=t2` only when the workflow requires the later epoch to stay fixed. In the synthetic verification, both directions produced nearly identical alignment quality, so this is usually an operational choice rather than a quality fix.

If changing the reference direction materially changes the answer, treat that as an alignment stability problem first.

### What "Good Enough" Looks Like After ICP

- The same seed and same inputs reproduce the same result
- The overlap filter does not discard an unexpectedly large share of areas you expect to be unchanged
- The alignment check stays broadly stable when you make a modest change to sample size
- Areas you expect to be unchanged look aligned before you interpret terrain change
- Switching the reference epoch does not materially change a pilot result

## M3C2 Best Practices

### When To Trust Autotune

Use header autotune first:

- `detection.m3c2.use_autotune=true`
- `detection.m3c2.autotune.source=header`

That is the best starting point for a new site. In the synthetic verification, header autotune and sample autotune chose almost identical radii and produced the same summary statistics.

Header autotune is especially useful when:

- You are running a new site for the first time
- You want the same scale choice across in-memory and streaming runs
- You want a stable first answer before freezing manual settings

### When To Override Autotune

Switch to fixed settings only after a pilot tells you roughly what scales are sensible. Manual settings make the most sense when:

- You are repeating the same site over time and want strict comparability
- You know the feature scale you care about
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

Those starting points were supported by the repo runs:

- `Jeksla`: `10%` produced `272,225` core points and a usable output
- Larger survey pilot: `5%` produced `296,866` core points while keeping broad coverage
- Synthetic sweep: lowering from `100%` to `10%` kept very similar summary statistics while greatly reducing output size

Increase the percentage when the output becomes too sparse to interpret. Decrease it when the pilot is too slow and the broad pattern is already clear.

### How To Think About `radius`, `normal_scale`, And `depth_factor`

When you manually override M3C2, start from the autotuned radius and keep the scales tied together:

- Keep `normal_scale=radius`
- Keep `depth_factor` close to the validated pilot value

The synthetic sweep is the clearest example:

- Header autotune picked about `4.5 m` and stayed stable
- Forcing the radius down to `2.0 m` made the result much noisier and reduced valid coverage
- Forcing the radius up to `8.0 m` kept the result stable but smoothed the response slightly

Practical rule:

- If the output is speckled or many points become invalid, increase `radius` and `normal_scale` together
- If the output is too smooth and small features blur away, decrease them together
- Change one scale step at a time

Use manual overrides to match a known feature scale, not as the first thing you change.

## Validation Checklist Before Accepting A Run

- Confirm the workflow selected the correct area and time periods
- Confirm the overlap and alignment sample sizes make sense for the dataset
- Record the chosen M3C2 radius, max depth, and core-point count from the log
- Open the M3C2 point output or raster and inspect both areas you expect to be unchanged and known change zones
- Repeat one pilot run with the same seed if the work needs a fixed baseline
- Save the exact config and seed used for the accepted run

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

This usually means the neighborhood is too large for the feature scale you care about.

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
