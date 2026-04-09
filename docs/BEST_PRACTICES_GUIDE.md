# Terrain Change Detection Best Practice Guide

This guide is for operators who need to make good configuration choices quickly. Use it to decide where to start, when to override the defaults, and how to judge whether an alignment or change-detection run is trustworthy.

This is not the field-by-field parameter reference. For the full YAML reference, see [Configuration Guide](CONFIGURATION_GUIDE.md). For the run evidence behind the recommendations below, see [Best-Practice Evidence](BEST_PRACTICES_EVIDENCE.md).

## How To Use This Guide

1. Pick the quick-start row that matches your dataset.
2. Run one pilot area with ICP and M3C2 before scaling up.
3. Check the validation list before accepting the outputs.
4. Change one setting family at a time: first alignment sampling, then M3C2 core density, then M3C2 neighborhood size.
5. Once a pilot works, freeze the config and keep the same seed for repeat work.

## Quick Start By Dataset Type

| Situation | Recommended starting settings | Signs they are working | Signs they are too aggressive or too loose | First adjustment to try |
| :--- | :--- | :--- | :--- | :--- |
| Dense drone data similar to `Jeksla` | Start from `config/profiles/drone.yaml`. For the repo datasets, override `paths.base_dir=data`. Keep ICP on, start with `max_correspondence_distance=2.0`, keep M3C2 on header autotune, and start with `core_points_percent=10.0`. | Most overlap is retained, the chosen M3C2 radius stays in the sub-meter to low-meter range, and the point output shows continuous coverage over the stable ground. | Speckled output, patchy coverage, or many invalid core points usually means the neighborhood is too small. Broad, soft-edged change patches usually means it is too large. | Leave the core-point percentage alone first and adjust `radius` and `normal_scale` together in small steps. |
| Larger survey data similar to `data/raw/.../2015` and `data/raw/.../2020` | Start from `config/profiles/large_scale.yaml`. For a single-area pilot that fits in memory, temporarily set `outofcore.enabled=false` and `parallel.enabled=false`. Keep M3C2 on header autotune and start with `core_points_percent=5.0`. | The selected core-point count is much lower than the full ground count, the chosen radius lands around a few meters, and valid core coverage stays high. | If the output is still too slow, you are sampling too many core points for a pilot. If banks, scarps, or narrow change bands blur together, the radius is too large. | Reduce `core_points_percent` before changing the neighborhood size. If feature edges are being smoothed, lower `radius` and `normal_scale` together. |
| Synthetic or controlled validation data | Start from `config/profiles/synthetic.yaml`. Use `subsample_size=30000` for ICP. Use M3C2 header autotune with `core_points_percent=100.0` while tuning parameters, then drop to `10.0` for quick checks. | Repeated runs with the same seed match, header and sample autotune pick almost the same radius, and the summary statistics stay stable when you reduce core-point density for a quick pass. | If lower ICP sampling changes the alignment noticeably, the alignment sample is too small. If a smaller fixed M3C2 radius suddenly makes the result much noisier, the neighborhood is too small. | Return to header autotune before making manual M3C2 overrides. Raise the ICP sample count before changing multiple alignment settings. |

## ICP Alignment Best Practices

### When ICP Should Run

| Situation | Recommended starting settings | Signs they are working | Signs they are too aggressive or too loose | First adjustment to try |
| :--- | :--- | :--- | :--- | :--- |
| Two epochs from different flights, sensors, or processing chains | Keep `alignment.enabled=true`. Keep `overlap_filter=true`. | Stable ground looks coincident after alignment and repeated runs with the same seed return the same result. | If the result changes meaningfully when you repeat the run or change the sample slightly, the setup is too fragile. | Increase the ICP sample count before changing anything else. |
| Products that are already known to be tightly aligned | Still do one pilot run with ICP on. Only disable ICP after a pilot shows that the transform is negligible and the change result is unaffected. | The pilot run produces the same change pattern with or without ICP. | If disabling ICP changes stable areas or known features, the data were not aligned enough to skip it. | Re-enable ICP and keep the default reference epoch. |

### How To Choose The ICP Sample Size

| Situation | Recommended starting settings | Signs they are working | Signs they are too aggressive or too loose | First adjustment to try |
| :--- | :--- | :--- | :--- | :--- |
| Production alignment on real data | Use `subsample_mode=count` and start with `subsample_size=50000`. | Final alignment metrics stay stable when you rerun the same site, and the run time is still acceptable. | If the alignment changes when you rerun with a different seed, the sample is too small. If it is too slow for a pilot, it is larger than it needs to be. | Move down or up in one clear step, for example 25k or 75k. |
| Fast pilot or small synthetic validation | Use `subsample_mode=percent` or a smaller fixed count. In the synthetic verification, 10% of points ran much faster than 30k points but gave a slightly worse validation error. | The faster run still lands close to the same result as the larger sample. | If the faster run changes the transform or validation error more than you are comfortable with, it is too small for decision-making. | Go back to a larger fixed count for the decision run. |

### When To Use Coarse Alignment

| Situation | Recommended starting settings | Signs they are working | Signs they are too aggressive or too loose | First adjustment to try |
| :--- | :--- | :--- | :--- | :--- |
| Visible XY shift, partial overlap, or drone data with uncertain starting placement | Turn on `alignment.coarse.enabled=true`. Prefer `method=phase` first for translation-dominated cases. | ICP starts from a sensible position and converges cleanly. | If coarse alignment adds no visible value or makes overlap worse, it is unnecessary or the method is wrong. | Fall back to `centroid`, or disable coarse alignment if the datasets already start close together. |
| Datasets that already share the same footprint | Leave coarse alignment off for the pilot. | ICP converges without needing a stronger initial nudge. | If ICP struggles to find enough correspondences or the initial overlap is poor, the start was too weak. | Enable `phase` first, then try `centroid` if the shift is simple. |
| Clear orientation mismatch and simpler starts fail | Treat `pca` or `open3d_fpfh` as recovery options, not as the first choice. | The pilot result becomes stable after simpler methods failed. | If they do not improve stability, they are adding complexity without benefit. | Return to the simpler method that gave the most consistent result. |

### How To Choose `max_correspondence_distance`

| Situation | Recommended starting settings | Signs they are working | Signs they are too aggressive or too loose | First adjustment to try |
| :--- | :--- | :--- | :--- | :--- |
| Dense drone data | Start at `2.0` meters. This worked on the repo's `Jeksla` example. | ICP finds enough correspondences and converges without obvious drift. | Too tight: ICP reports too few correspondences or stops early. Too loose: alignment looks plausible numerically but stable surfaces still appear smeared. | Change the distance by one step only, not together with other ICP settings. |
| Larger survey data | Start at `1.0` meter. This worked on the repo's larger survey example. | Stable ground lines up and the run does not need a larger search radius. | Too tight: convergence fails. Too loose: the search pulls in wrong neighbors and the final alignment becomes less trustworthy. | Raise to `1.5` or `2.0` only if the pilot clearly needs it. |

### When The Reference Epoch Matters

| Situation | Recommended starting settings | Signs they are working | Signs they are too aggressive or too loose | First adjustment to try |
| :--- | :--- | :--- | :--- | :--- |
| Normal monitoring workflow | Keep the default `reference=t1`. | The later epoch aligns cleanly to the earlier baseline. | If a change in reference direction materially changes the answer, the alignment is not robust enough yet. | Improve sampling or correspondence settings before changing the reference direction. |
| You must keep the later epoch fixed | Use `reference=t2` and spot-check the result on one pilot area. The synthetic verification showed nearly identical RMSE in both directions, so this is usually a workflow choice rather than a quality fix. | Stable ground still aligns and the result stays close to the default direction. | If direction changes the answer materially, treat that as an alignment problem, not a reference preference. | Return to `t1`, stabilize the alignment, then try `t2` again if needed. |

### What "Good Enough" Looks Like After ICP

- The same seed and same inputs reproduce the same result.
- The overlap filter does not discard an unexpectedly large share of stable ground.
- The final alignment metric stays broadly stable when you rerun with a modestly different sample size.
- Stable ground looks aligned before you interpret any detected terrain change.
- Switching the reference epoch does not materially change the result on a pilot area.

## M3C2 Best Practices

### When To Trust Autotune And When To Override It

| Situation | Recommended starting settings | Signs they are working | Signs they are too aggressive or too loose | First adjustment to try |
| :--- | :--- | :--- | :--- | :--- |
| First pass on a new site | Use `use_autotune=true` with `autotune.source=header`. In the synthetic verification, header and sample autotune chose almost identical radii and produced the same summary statistics. | The chosen radius matches the point density: sub-meter to low-meter on dense drone data, around a few meters on sparser survey data. | If the output is obviously noisy or obviously over-smoothed, the autotuned neighborhood does not match the feature size you care about. | Freeze a manual `radius` and `normal_scale` only after one good pilot run tells you roughly where they should land. |
| Repeated monitoring on the same site | After one successful pilot, freeze the values with `use_autotune=false` and set fixed `radius`, `normal_scale`, and `depth_factor`. | Repeated runs stay comparable because the neighborhood definition no longer changes from project to project. | If you keep changing fixed values from run to run, you lose the main benefit of freezing them. | Pick one validated setting set and reuse it. |
| Need mode consistency across in-memory and streaming work | Prefer header autotune or fixed settings. | Different execution modes choose the same neighborhood size. | If sample-based autotune changes when the sampled cloud changes, the result is less repeatable. | Switch to header autotune before moving to fixed settings. |

### How To Choose Core-Point Density

| Situation | Recommended starting settings | Signs they are working | Signs they are too aggressive or too loose | First adjustment to try |
| :--- | :--- | :--- | :--- | :--- |
| Parameter study or small validation area | Start with `core_points_percent=100.0`. | You get the fullest possible picture while comparing settings. | Runtime and file size can become unnecessarily large for routine work. | Once the settings are stable, lower the percentage for operational runs. |
| Dense drone operational run | Start with `core_points_percent=10.0`. This produced a usable `Jeksla` output with 272,225 core points. | Coverage stays continuous and the summary statistics remain close to the denser pilot. | If changes disappear because the output is too sparse, the percentage is too low. | Raise to `15` or `20` before changing the neighborhood size. |
| Larger survey operational run | Start with `core_points_percent=5.0`. This reduced the repo's large survey case to 296,866 core points while preserving broad coverage. | Runtime drops sharply while valid coverage stays high. | If the output becomes too sparse for interpretation, the core-point percentage is too low. | Raise the core-point percentage before touching the M3C2 radius. |
| Fast first-pass screening | Use the lowest density that still shows the main patterns, typically `5` to `10` percent. | Major change zones still appear in the right places. | If only isolated patches survive or narrow features disappear, the percentage is too low for that area. | Increase the percentage one step and rerun. |

### How To Think About `radius`, `normal_scale`, and `depth_factor`

| Situation | Recommended starting settings | Signs they are working | Signs they are too aggressive or too loose | First adjustment to try |
| :--- | :--- | :--- | :--- | :--- |
| Manual override after a good pilot | Start from the autotuned radius. Keep `normal_scale=radius`. Keep `depth_factor` close to the validated pilot value. | The manual result looks like a controlled refinement of the pilot, not a completely different answer. | If you change several scale parameters at once, it becomes hard to see which one caused the shift. | Keep `normal_scale` tied to `radius` and change only one scale step at a time. |
| Output is too speckled or too many core points become invalid | Increase `radius` and `normal_scale` together. Keep `depth_factor` conservative at first. | Coverage becomes more continuous and the distance spread settles down. | If the result starts to wash out narrow features, you went too far. | Step the radius back down slightly. |
| Output is too smooth and small features blur away | Decrease `radius` and `normal_scale` together. | Edges sharpen without the stable ground turning into noise. | If the distance spread suddenly explodes or valid coverage drops, the radius is now too small. | Return to the previous radius and lower the core-point spacing instead. |

The synthetic sweep is the clearest example of why this matters:

- Header autotune picked a radius of about `4.5` m and stayed stable.
- Forcing the radius down to `2.0` m made the output much noisier and reduced valid coverage.
- Forcing the radius up to `8.0` m kept the result stable but smoothed the response slightly.

Use manual overrides to match a known feature scale, not as the first thing you change.

## Validation Checklist Before You Accept A Run

- Confirm the workflow selected the correct area and time periods.
- Confirm the overlap and alignment sample sizes make sense for the dataset.
- Record the chosen M3C2 radius, max depth, and core-point count from the log.
- Open the M3C2 point output or raster and inspect both stable ground and known change zones.
- Repeat one pilot run with the same seed if the analysis is important enough to need a fixed baseline.
- If you freeze manual M3C2 settings, store the exact config and seed with the project record.

## Common Failure Modes

| Symptom | What it usually means | What to change first |
| :--- | :--- | :--- |
| No drone data are discovered | `base_dir` points to an area folder instead of the parent folder, or the source type is wrong. | For drone-style repo data, set `paths.base_dir` to the parent folder such as `data` and keep `discovery.source_type=drone`. |
| ICP reports too few correspondences or stops too early | The correspondence distance is too tight, the overlap is too small, or the starting offset is too large. | Increase `max_correspondence_distance` one step or enable coarse alignment. |
| Alignment looks unstable across seeds or sample sizes | The ICP sample is too small or the stable overlap is too weak. | Increase the alignment sample size and keep `overlap_filter=true`. |
| M3C2 output is speckled or has many holes | The neighborhood is too small or the core-point density is too sparse for the terrain and point density. | Return to header autotune or increase `radius` and `normal_scale` together. |
| M3C2 output is too smooth | The neighborhood is too large for the feature scale you care about. | Lower `radius` and `normal_scale` together. |
| Large pilot runs are too slow | The pilot is still evaluating too many core points or generating too many outputs. | Lower `core_points_percent`, turn off extra outputs for the pilot, then re-enable them for the final production run. |

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
