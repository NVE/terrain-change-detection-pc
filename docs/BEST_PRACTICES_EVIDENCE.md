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
| Final alignment RMSE | `1.015225` |
| Validation RMSE | `1.014352` |
| Core points | `272,225` (`10.0%`) |
| Autotuned radius | `0.90` m |
| Autotuned max depth | `0.54` m |
| Valid M3C2 points | `252,759 / 272,225` (`92.85%`) |
| M3C2 distance std | `0.0601` m |
| Absolute p95 distance | `0.1150` m |
| Elapsed wall time | `40.31` s |

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
| Final alignment RMSE | `0.697938` |
| Validation RMSE | `0.700875` |
| Core points | `296,866` (`5.0%`) |
| Autotuned radius | `2.00` m |
| Autotuned max depth | `2.00` m |
| Valid M3C2 points | `295,953 / 296,866` (`99.69%`) |
| M3C2 distance std | `0.0861` m |
| Absolute p95 distance | `0.1506` m |
| Elapsed wall time | `71.19` s |

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

## Evidence Summary

The operator guide rests on five repo-backed takeaways:

1. Use a fixed seed and a larger ICP sample for the decision run.
2. Leave header autotune on for the first M3C2 pilot.
3. Reduce core-point density before hand-tuning M3C2 neighborhoods for runtime.
4. When you do override M3C2 scales, move `radius` and `normal_scale` together.
5. Freeze manual M3C2 settings only after one pilot run proves they match the feature scale you care about.
