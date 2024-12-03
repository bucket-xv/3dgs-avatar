# Preparation for experiments

## Environment

run `env.sh`

## Model

Too many manual operations to be written here

## Dataset

### Overall illustration & ZJUMoCap

- https://github.com/taconite/arah-release/blob/main/DATASET.md

### People Snapshot

- Source dataset lies at: https://graphics.tu-bs.de/people-snapshot

## Train && eval

### Train

- Take care of the available GPUs, run `nvidia-smi` beforehand to watch GPU utilization.
- Valid choices of `dataset.test_mode`: view, video, all

```bash
conda activate 3dgs-avatar
CUDA_VISIBLE_DEVICES=1 python render.py mode=test dataset.test_mode=video dataset=zjumocap_377_mono
```


### Eval

Take care of the available GPUs, run `nvidia-smi` beforehand to watch GPU utilization.

```bash
CUDA_VISIBLE_DEVICES=1 python render.py mode=test dataset.test_mode=pose pose_correction=none dataset=zjumocap_377_mono
```

### Predict

```bash
CUDA_VISIBLE_DEVICES=1 python render.py mode=predict dataset.predict_seq=0 dataset=zjumocap_377_mono
```

## Visualize

```bash