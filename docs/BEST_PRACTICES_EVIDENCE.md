# Best-Practice Evidence

This note records the repo-backed observations used to write the operator guide in [Best Practice Guide](BEST_PRACTICES_GUIDE.md). It is intentionally short. The goal is to show which datasets were used, which runs were checked, and which results drove the recommendations.

## Datasets Used

| Dataset | Role in the guide | Notes |
| :--- | :--- | :--- |
| `data/Jeksla` | Dense drone example | Real drone-style layout with partial overlap between epochs. |
| `data/raw/eksport_1225654_20250602` | Larger survey example | Real larger survey tile pair from 2015 and 2020. |
| `data/synthetic/synthetic_area` | Controlled parameter sweep | Small, fast dataset with known injected changes and known misalignment. |

## Supporting ICP Evidence

These recommendations were based on two sources:

- Existing verification report: [ICP Alignment Fixes — Verification Report](ICP_FIX_VERIFICATION_REPORT.md)
- Fresh best-practice runs under `verification_runs/best_practice/`

### ICP observations used in the guide

| Observation | Evidence | Why it matters |
| :--- | :--- | :--- |
| Repeated runs with the same seed are deterministic. | `docs/ICP_FIX_VERIFICATION_REPORT.md` shows identical synthetic and `Jeksla` results for repeated seed-42 runs. | This supports using a fixed seed for repeatable production work. |
| Lower ICP sampling is faster but slightly less stable. | Synthetic verification in `docs/ICP_FIX_VERIFICATION_REPORT.md`: `30,000` alignment points gave validation RMSE `0.692617`, while `10%` sampling (`6,250` points) gave `0.711072` but ran much faster. | This supports using a larger fixed count for final runs and smaller samples only for quick pilots. |
| Reference direction is usually a workflow choice, not a quality fix. | Synthetic verification in `docs/ICP_FIX_VERIFICATION_REPORT.md`: `reference=t1` and `reference=t2` produced near-identical validation RMSE. | This supports keeping `t1` as the default unless operations require the later epoch to stay fixed. |
| Overlap filtering is useful on the drone example. | `Jeksla` verification retained `99.9%` of T1 points and `97.1%` of T2 points in the overlap region. | This supports leaving `overlap_filter=true` on real work. |

## How Statistics Were Collected

Run logs captured at INFO level record alignment RMSE, core point count, and runtime. Detailed M3C2 statistics (autotuned radius, distance std, p95, valid point count) come from two sources:

- **Autotuned parameters and summary statistics**: logged by the `terrain_change_detection.detection.m3c2` logger, visible at DEBUG level or above.
- **Output LAZ analysis**: the `distance` extra dimension in the exported M3C2 point cloud was read with `laspy` and summarized with NumPy to compute std, percentiles, and valid counts.

To reproduce any statistic, re-run the command with `--set logging.level=DEBUG` or load the output LAZ file and inspect the `distance` field.

## Fresh M3C2 Quick-Start Runs

### Dense drone quick start

Command style used:

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

Result summary:

| Metric | Value |
| :--- | :--- |
| Alignment sample | `50,000` per epoch |
| ICP RMSE (on subsample) | `1.015225` |
| Validation RMSE | `1.014352` |
| Core points | `272,225` (`10.0%`) |
| Autotuned radius | `0.90` m |
| Autotuned max depth | `0.54` m |
| Valid M3C2 points | `252,759 / 272,225` (`92.85%`) |
| M3C2 distance std | `0.0601` m |
| Absolute p95 distance | `0.1150` m |
| Workflow runtime | `37.34` s |

Supporting files:

- Log: `verification_runs/best_practice/jeksla_quickstart/run.log`
- Output: `verification_runs/best_practice/jeksla_quickstart/m3c2_Jeksla_2024-12-03_2025-11-14.laz`

What this supports:

- Header autotune is a sensible first choice on dense drone data.
- `10%` core points is a practical starting density for operator runs.
- A sub-meter neighborhood is plausible on dense ground-only drone data.

### Larger survey quick start

Command style used:

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

Result summary:

| Metric | Value |
| :--- | :--- |
| Alignment sample | `50,000` per epoch |
| ICP RMSE (on subsample) | `0.697938` |
| Validation RMSE | `0.700875` |
| Core points | `296,866` (`5.0%`) |
| Autotuned radius | `2.00` m |
| Autotuned max depth | `2.00` m |
| Valid M3C2 points | `295,953 / 296,866` (`99.69%`) |
| M3C2 distance std | `0.0861` m |
| Absolute p95 distance | `0.1506` m |
| Workflow runtime | `67.25` s |

Supporting files:

- Log: `verification_runs/best_practice/raw_quickstart/run.log`
- Output: `verification_runs/best_practice/raw_quickstart/m3c2_eksport_1225654_20250602_2015_2020.laz`

What this supports:

- `5%` core points is a useful starting point on a much larger survey tile.
- Header autotune picked a broader neighborhood than the drone case, which matches the lower effective density.
- A single-area pilot can be run in memory first, then scaled out later if needed.

## Synthetic M3C2 Parameter Sweep

All synthetic runs used the same dataset and seed, disabled DoD and C2C, and exported only the M3C2 point output.

| Scenario | Main settings | Valid points | Distance std | Absolute p95 | What it shows |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Header autotune | `radius≈4.52 m`, `100%` core points | `62499 / 62500` (`100.0%`) | `0.0713` m | `0.0924` m | Good stable baseline. |
| Sample autotune | `radius≈4.50 m`, `100%` core points | `62499 / 62500` (`100.0%`) | `0.0713` m | `0.0924` m | Essentially identical to header autotune on this dataset. |
| Fixed small radius | `radius=2.0 m`, `normal_scale=2.0`, `depth_factor=1.0` | `60967 / 62500` (`97.55%`) | `0.2656` m | `0.6133` m | Much noisier and less stable. Good example of an override that is too small. |
| Fixed large radius | `radius=8.0 m`, `normal_scale=8.0`, `depth_factor=1.0` | `62500 / 62500` (`100.0%`) | `0.0693` m | `0.0884` m | Slightly smoother than autotune. Good example of an override that begins to blur the response. |
| Reduced core density | Header autotune with `10%` core points | `6250 / 6250` (`100.0%`) | `0.0704` m | `0.0895` m | Similar summary statistics with a much smaller output set. Good for quick screening. |

Supporting files:

- Logs: `verification_runs/best_practice/synth_*/run.log`
- Outputs: `verification_runs/best_practice/synth_*/m3c2_synthetic_area_2015_2020.laz`

What this supports:

- Header autotune is a strong default.
- Manual overrides should be based on a clear feature-scale reason, not on guesswork.
- Core-point density can often be reduced for faster pilots without changing the broad answer.

## Additional Verification Experiments

These experiments were run independently to verify the recommendations in the guide.

### ICP Subsample Size Effect on Jeksla

All runs used `config/profiles/drone.yaml` with `--set paths.base_dir=data`, M3C2 autotune enabled, and 10% core points.

| ICP subsample | ICP iterations | ICP time | M3C2 valid | M3C2 std | M3C2 abs p95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `25,000` | 50 | `2.0` s | `252,828` | `0.0623` m | — |
| `50,000` (default) | 67 | `5.5` s | `252,367` | `0.0594` m | — |
| `100,000` | 25 | `5.6` s | `252,552` | `0.0542` m | — |

What this supports:

- The default `50,000` is a reasonable balance. Increasing to `100,000` reduces M3C2 noise slightly but roughly doubles alignment time.
- Reducing to `25,000` is acceptable for quick pilots; the M3C2 std only increases by ~5%.

### M3C2 Radius Sweep on Jeksla (Drone Data)

All runs used ICP with 50K subsample, autotune radius was `0.90` m.

| Radius | Valid | Std | Comment |
| :--- | :--- | :--- | :--- |
| `0.50` m (manual) | `243,567 / 272,225` (`89.5%`) | `0.0639` m | Noisier, fewer valid points |
| `0.90` m (autotune) | `252,722 / 272,225` (`92.8%`) | `0.0595` m | Good baseline |
| `2.00` m (manual) | `255,558 / 272,225` (`93.9%`) | `0.0566` m | Smoother, more valid |

What this supports:

- Autotune picks a sensible middle ground between noise and smoothing on real drone data.
- The guide's advice to increase radius when output is speckled, and decrease it when output is too smooth, holds on real data.

### M3C2 Radius Sweep on Larger Survey Data

All runs used ICP with 50K subsample, autotune radius was `2.00` m (clamped to `min_radius=2.0`).

| Radius | Valid | Std |
| :--- | :--- | :--- |
| `1.00` m (manual) | `294,815 / 296,866` (`99.31%`) | `0.0909` m |
| `2.00` m (autotune) | `295,953 / 296,866` (`99.69%`) | `0.0861` m |
| `4.00` m (manual) | `296,445 / 296,866` (`99.86%`) | `0.0806` m |

What this supports:

- Same pattern: smaller radius is noisier, larger is smoother.
- The autotuned radius was clamped to `min_radius=2.0` from the `large_scale.yaml` profile, not freely chosen. Users should be aware of the clamping bounds.

### ICP-On vs ICP-Off Comparison on Jeksla

| Setting | M3C2 valid | M3C2 mean | M3C2 std |
| :--- | :--- | :--- | :--- |
| ICP enabled (default) | `252,722` | `-0.0064` m | `0.0595` m |
| ICP disabled | `252,507` | `0.0126` m | `0.0590` m |

What this supports:

- For this well-aligned drone dataset, ICP has minimal effect on M3C2 statistics. The mean shifted closer to zero with ICP, and valid count increased slightly.
- This confirms the guide's advice to always run ICP and check whether it makes a material difference.

### Note on RMSE Metric Comparability

The Jeksla runs show pre-ICP RMSE `0.324` m and post-ICP RMSE `1.014` m. These values are **not comparable** because they use different target point densities:

- Pre-ICP: computes nearest-neighbor distance against the full T1 cloud (~2.7M points)
- Post-ICP: computes against the 50K subsampled T1 used for ICP

With sparser targets, nearest-neighbor distances are naturally larger. A test on the raw (unaligned) Jeksla data confirmed: RMSE against full T1 is `0.324`, RMSE against 50K T1 is `1.014` — identical to the logged values. The ICP transform is negligible here, not harmful.

Users should compare alignment quality via M3C2 outputs (ICP-on vs ICP-off), not via the log RMSE values which depend on target density.

## Evidence Summary

The operator guide rests on these repo-backed takeaways:

1. Use a fixed seed and a larger ICP sample for the decision run.
2. Leave header autotune on for the first M3C2 pilot.
3. Reduce core-point density before hand-tuning M3C2 neighborhoods for runtime.
4. When you do override M3C2 scales, move `radius` and `normal_scale` together.
5. Freeze manual M3C2 settings only after one pilot run proves they match the feature scale you care about.
6. The autotune radius-vs-noise tradeoff observed on synthetic data holds on real drone and survey data.
7. ICP subsample size has a modest effect on downstream M3C2 quality — 50K is a reasonable default.
8. Compare alignment quality through M3C2 outputs, not through the log RMSE values which depend on target point density.
