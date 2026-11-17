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

The 11 checkpoints used by Team TUM's final BIC-MAC 2026 submission. The model
predicts a CT attenuation map from NAC-PET, an X-ray topogram, and Dixon MRI.

## Download

```bash
hf download ziyuan-li/bic-mac-tum-e612 --local-dir model
```

The package contains E06, E08, five E13 fold models, E13-full75, E24, E40, and
E43. `runtime-closure.json` records the expected file sizes and SHA-256 hashes.

## Inference

Code and setup instructions are available in
[`ziyuan-li/bic-mac-2026-methodology`](https://github.com/ziyuan-li/bic-mac-2026-methodology).

```bash
python scripts/verify_weights.py --model-dir model
PYTHONPATH=src python -m champion.predict \
  --features-dir /path/to/features \
  --model-dir model \
  --output-ct output/ct.nii.gz \
  --report output/runtime-report.json \
  --device cuda:0
```

## Results

| Metric | Hidden test result |
|---|---:|
| CT mu-map MAE | 0.00591 |
| Whole-body SUV MAE | 0.0284 |
| Organ bias | 1.95% |
| Brain outlier score | 0.0249 |
| Mean rank | 6.25 |

Python 3.12, PyTorch 2.10.0, and an NVIDIA GPU are required. The model was
developed for the BIC-MAC inputs and geometry and is not a medical device.
