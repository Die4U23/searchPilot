# CTR inference artifact contract

New exports declare artifact_schema=1 and model_type=mlp. manifest.json must contain exactly meta.json, lr.json, mlp.json, temperature.json and feature_tables.json, with a full SHA-256 digest of each file's actual bytes. The loader parses the verified bytes without rereading inference files. Missing, modified or mixed-generation files fail with CtrNotReadyError (not ready), rather than silently switching to LR.

The model identity hashes all LR and MLP arrays, feature names, standardization mean/std, temperature and all feature tables. The calibrator identity additionally binds temperature to that model identity. Changing any of these inputs changes both identities. Hashes are integrity/version checks, not signatures, provenance proof or evidence of model quality. Offline calibrators.json, metrics and test scores are not online inference inputs and are not part of this manifest.

Publication atomically replaces individual files and writes the complete manifest last. On initial publication a not-ready marker precedes all file replacements. Readers may see not-ready during replacement or after a failed write; already loaded runtimes retain their in-memory snapshot. This is not a zero-downtime deployment protocol or a power-loss durability guarantee. Use a separately staged deployment directory when availability during replacement matters. The publishing process and serving process need appropriate read access; temporary-file permissions are private by default. Do not concurrently publish from multiple writers to one directory.

Legacy compatibility:
- An explicit mlp type requires mlp.json. Known old train_ctr exports (both epochs and learning_rate in metadata) also require it.
- Explicit lr packages use LR even when an unrelated mlp.json is present.
- Old untyped packages without these training markers retain the historical LR-only compatibility path. Their intended model cannot always be reconstructed from ambiguous metadata; migrate them via the exporter for full integrity and identity checks.
- Versioned metadata without a manifest is not a legacy package and is rejected.

The training script now calls the bundle exporter. This change does not regenerate historical model artifacts, metrics or reports and makes no claim of improved AUC/ECE. Focused tests use tiny synthetic weights, not a full training run. CI must run tests/test_ctr_bundle.py alongside existing runtime/MLP regression tests.
