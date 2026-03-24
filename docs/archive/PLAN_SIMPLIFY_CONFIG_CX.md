# Simplify Configuration Around One Canonical Default

## Summary

- Make `config/default.yaml` the single canonical runtime config and the main documented entrypoint.
- Keep YAML-based flexibility, but stop treating extra YAMLs as full copied profiles; instead, treat them as small override files layered on top of `default.yaml`.
- Add a generic dot-path CLI override mechanism so users can work with new datasets without copying YAML files.
- Keep migration soft: existing `--config` usage and current profile paths continue to work during transition.

### Acceptance criteria

- A user can run a new dataset with `default.yaml` plus CLI overrides only.
- Repeated datasets can use a tiny override YAML instead of a copied full config.
- Committed override files contain only differences from `default.yaml`.
- Missing keys in override YAMLs inherit from `default.yaml`, not from drifting code defaults.

## Public Interfaces

### CLI

- Keep `--config`, but redefine and document it as an optional override YAML merged on top of `config/default.yaml`.
- Make `--config` repeatable so users can combine a curated preset and a local dataset override.
- Add repeatable `--set KEY=VALUE` for generic runtime overrides, using dot paths like `paths.base_dir=data/foo` and `outofcore.enabled=true`.
- Keep dedicated convenience flags such as `--base-dir`, `--seed`, `--reference`, and M3C2-specific flags, but route them through the same override pipeline.

### Config loader

- Extend `load_config(...)` to build the final config from layered sources with explicit precedence.
- Precedence order: schema safety defaults < `config/default.yaml` < `--config` files in the order provided < dedicated CLI override flags < `--set`.

### Repo config conventions

- `config/default.yaml` remains the only full committed config file.
- Existing files under `config/profiles/` and `config/default_clipped.yaml` are retained as compatibility paths, but converted to minimal override files.
- Add a gitignored `config/local/` convention for user- or dataset-specific tiny override YAMLs.

## Implementation Changes

- Keep the current Pydantic + PyYAML stack; do not introduce Hydra or OmegaConf.
- In the config utilities:
  - Add deep-merge support for layered YAML dictionaries before validation.
  - Add parsing for `--set KEY=VALUE` entries, with YAML-style value coercion for booleans, numbers, lists, and `null`.
  - Reject unknown override paths and invalid types with source-aware errors that name the failing key and whether it came from `default.yaml`, an override file, or CLI.
  - Reconcile today’s drift between `AppConfig` defaults and `config/default.yaml` so runtime defaults are consistent.
- In the workflow CLI:
  - Parse repeated `--config` and repeated `--set`.
  - Stop applying some overrides ad hoc after config load; instead, construct one final resolved config once and use it everywhere.
  - Update run-input logging so it captures:
    - ordered override files,
    - raw `--set` entries,
    - the final resolved config snapshot.
- In repo config files and docs:
  - Slim committed override YAMLs down to changed keys only.
  - Rewrite README and configuration docs around `default + overrides` rather than `copy a profile`.
  - Fix stale examples that reference nonexistent profile files.
  - Document three standard workflows:
    - default only,
    - default plus one-off CLI overrides,
    - default plus preset override plus local dataset override.

## Test Plan

- Add unit tests for deep-merge behavior so omitted keys in override YAMLs inherit from `config/default.yaml`.
- Add unit tests for `--set` parsing and typing for `true/false`, integers, floats, `null`, lists, and plain strings.
- Add precedence tests covering:
  - default only,
  - one override file,
  - multiple override files,
  - dedicated CLI flags,
  - `--set` winning over all earlier sources.
- Add error-path tests for:
  - unknown dot paths,
  - type-invalid override values,
  - malformed `KEY=VALUE` entries.
- Add regression tests confirming current profile paths still load successfully after migration.
- Add a parity test that guards against drift between `AppConfig` defaults and `config/default.yaml` for all user-facing fields.
- Add a workflow integration test that verifies the persisted run-input manifest contains the resolved merged config plus the override provenance.

## Assumptions And Defaults

- The simplified system should remain incremental and low-risk; no config framework replacement is planned.
- The primary UX is `config/default.yaml` plus CLI overrides.
- Tiny override YAMLs are still supported for repeated datasets, but they are optional and should stay minimal.
- Existing profile file paths remain valid for at least one transition cycle, even if their contents become minimal override patches.
