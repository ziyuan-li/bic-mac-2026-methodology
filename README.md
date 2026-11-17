# Physics-Aware Branched Mu-Map Ensembles

Code for Team TUM's BIC-MAC 2026 submission. The method predicts CT
attenuation maps from NAC-PET, an X-ray topogram, and Dixon MRI. The method is
described in [`Physics_Aware_Branched.pdf`](Physics_Aware_Branched.pdf).

## Contents

```text
src/champion/              inference code
configs/                   deployment and checkpoint manifests
training/source/champion/  training code
training/configs/          final training configurations
docker/                    final container definition
scripts/                   weight verification utilities
tests/                     release tests
```

## Installation

Python 3.12 and an NVIDIA GPU are required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0
python -m pip install -r requirements.txt
```

## Weights

The 11 checkpoints are hosted on
[Hugging Face](https://huggingface.co/ziyuan-li/bic-mac-tum-e612).

```bash
hf download ziyuan-li/bic-mac-tum-e612 --local-dir model
python scripts/verify_weights.py --model-dir model
```

## Inference

The feature directory must contain:

```text
nacpet.nii.gz
topogram.nii.gz
mri_combined_in_phase.nii.gz
mri_combined_out_phase.nii.gz
```

Run:

```bash
PYTHONPATH=src python -m champion.predict \
  --features-dir /path/to/features \
  --model-dir model \
  --output-ct output/ct.nii.gz \
  --report output/runtime-report.json \
  --device cuda:0
```

The output is a NIfTI CT volume in Hounsfield units on the NAC-PET grid.

## Docker

Download the weights into `model/`, then run:

```bash
docker build -f docker/Dockerfile -t bicmac-tum:e612 .
docker run --rm --gpus all --network none \
  -v /path/to/features:/data/features:ro \
  -v /path/to/output:/data/output \
  bicmac-tum:e612
```

The organizer-tested image is also available as a
[Docker archive](https://drive.google.com/file/d/1BuqtBtCMoaA3ddaMx0z7TSynmvNHZATj/view?usp=sharing):

```text
e612-e582-e577-runlog-compatible-dry-run.tar.gz
SHA256 67d3c45b66c04720c46e0407d0536b678788b691e2f6445b850ac2d0d0e83f42
```

## Training

Training instructions and configurations are in
[`training/README.md`](training/README.md).

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

This repository is for research use only and is not a medical device.
