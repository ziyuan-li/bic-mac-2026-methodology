---
library_name: pytorch
pipeline_tag: image-to-image
tags:
  - medical-imaging
  - pet
  - ct
  - attenuation-correction
  - bic-mac
---

# Team TUM BIC-MAC 2026 model

This repository contains the 11 checkpoints and configurations used by Team
TUM's final BIC-MAC 2026 submission. The weights are extracted from the exact
organizer-tested image `bicmac-e582-e612:dry-run-20260810`.

The model predicts a CT attenuation map from NAC-PET, an X-ray topogram, and
Dixon MRI. It does not generate PET. In the challenge, PET was reconstructed by
the organizers from the submitted CT using the official reconstruction pipeline.

## Files

```text
runtime-closure.json
deployment.json
e06/last.pt
e06/config.json
e08/last.pt
e08/config.json
e13-fold0/best-objective.pt
e13-fold0/config.json
...
e13-fold4/best-objective.pt
e13-fold4/config.json
e13-full75/last.pt
e13-full75/config.json
e24/last.pt
e24/config.json
e40/last.pt
e40/config.json
e43/last.pt
e43/config.json
```

`runtime-closure.json` records the byte count and SHA-256 digest of every model
file. The checkpoints total approximately 1.61 GB uncompressed.

## Use

Inference code and instructions are available at
[`ziyuan-li/bic-mac-2026-methodology`](https://github.com/ziyuan-li/bic-mac-2026-methodology).

```bash
hf download ziyuan-li/bic-mac-tum-e612 --local-dir model
python scripts/verify_weights.py --model-dir model
PYTHONPATH=src python -m champion.predict \
  --features-dir /path/to/features \
  --model-dir model \
  --output-ct output/ct.nii.gz \
  --report output/runtime-report.json
```

The final environment used Python 3.12, PyTorch 2.10.0, CUDA 12.6.3, MONAI
1.5.2, NiBabel 5.3.3, and SciPy 1.18.0. An NVIDIA GPU with substantial memory
is required; the official evaluation used one NVIDIA A40.

## Results

On the hidden BIC-MAC test set, the submission achieved 0.00591 CT mu-map MAE,
0.0284 whole-body SUV MAE, 1.95% organ bias, and 0.0249 brain outlier score.
Its mean rank was 6.25 (joint sixth, Tier 1).

## Limitations

The model was developed for the BIC-MAC data geometry and input modalities. It
has not been validated for other scanners, acquisition protocols, populations,
or clinical use. This is research software and is not a medical device.
