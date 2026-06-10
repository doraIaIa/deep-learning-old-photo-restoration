# Scripts Layout

Root scripts provide stable user-facing commands for pipeline execution, demo, evaluation, artifact verification, and reproducibility checks.

Additional subfolders document the intended long-term layout:

- `scripts/train/`: training documentation and future training entrypoints when external data paths are configured
- `scripts/data/`: dataset manifest and split-preparation helpers
- `scripts/artifacts/`: checkpoint and artifact inspection helpers

The repository keeps the current root commands stable while documenting a cleaner layout for future growth.
