# Training snapshot

This directory contains the training implementation and configurations used by
the model families in the final Team TUM inference graph.

## Scope

- `source/champion/train.py` trains the projection-conditioned base networks.
- `source/champion/train_refiner.py` trains MRI-conditioned residual refiners.
- `configs/` records the configuration stored next to each final checkpoint.
- `folds.json` records the fixed five-fold subject assignment.
- `run_config.py` converts a recorded JSON configuration into a command line.

The training source is a snapshot of the research trainer at the end of the
challenge. It supports the options used by the final families as well as later
ablation options. The inference implementation under `../src/champion/` is a
separate, exact extraction from the submitted image.

## Data cache

The trainer consumes the preprocessed cache produced by the challenge code.
Each subject needs the NAC-PET channels, projected topogram, target mu-map,
body/tissue masks, and metadata. Refiner training additionally needs Dixon MRI
channels and a cached base prediction. Paths in the configurations are relative
to the repository and should be adjusted for the local dataset location.

## Base networks

Inspect a recorded command without launching it:

```bash
PYTHONPATH=training/source python training/run_config.py training/configs/e06.json
```

Run it after preparing the cache and initialization checkpoint:

```bash
PYTHONPATH=training/source python training/run_config.py \
  training/configs/e06.json --execute
```

The final base families form a continuation chain. E06 starts from the selected
auxiliary model, E08 starts from E06, E13 starts from E08, and E24 starts from
E13. The five E13 fold models start from their corresponding E08 fold models.
The E04 auxiliary initializer and five E08 fold initializers are precursor
artifacts and are not part of the final inference package; their expected paths
are explicit in the JSON files.

## MRI refiners

E40 and E43 use the same architecture and data, with independent random seeds:

```bash
PYTHONPATH=training/source python training/run_config.py \
  training/configs/e40.json --refiner --execute
```

Before running, set `base_prediction_dir` in the JSON file to the cached E35
base predictions used to condition the refiner.

## Reproducibility boundary

The Hugging Face package contains all 11 checkpoints required for exact final
inference. It does not contain every precursor checkpoint in the multi-stage
training chain, optimizer state from every experiment, or the BIC-MAC dataset.
Consequently, this directory documents and reruns the method, while the released
weights are the reference for reproducing the submitted predictions.
