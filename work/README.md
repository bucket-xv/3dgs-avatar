# Preparation for experiments

## Environment

run `env.sh`

## Model

Too many manual operations to be written here.

## Dataset

### Overall illustration & ZJUMoCap

- https://github.com/taconite/arah-release/blob/main/DATASET.md

### People Snapshot

- Source dataset lies at: https://graphics.tu-bs.de/people-snapshot

## Train && eval

- Results in this section should be seen under `exp/`
- You need to run the scripts under the project root directory.
- You need to activate the right environment (run `conda activate 3dgs-avatar`)

### Train

- Take care of the available GPUs, run `nvidia-smi` beforehand to watch GPU utilization.
- Valid choices of `dataset.test_mode`: view, video, all

```bash
CUDA_VISIBLE_DEVICES=1 python render.py mode=test dataset.test_mode=video dataset=zjumocap_377_mono
```

### Eval

Take care of the available GPUs, run `nvidia-smi` beforehand to watch GPU utilization.

```bash
CUDA_VISIBLE_DEVICES=1 python render.py mode=test dataset.test_mode=pose pose_correction=none dataset=zjumocap_377_mono
```

### Predict

Predict is used to predict novel poses from the given monocular video.

```bash
CUDA_VISIBLE_DEVICES=1 python render.py mode=predict dataset.predict_seq=0 dataset=zjumocap_377_mono
```

## Visualize

`video.py` is a simple script to convert images to video. To use it, first switch to `tool` environment.

```bash
conda activate tool
python video.py -h

# Example command:
# python video.py -i /users/xch/multimodal/3dgs-avatar/exp/zju_377_mono-direct-mlp_field-ingp-shallow_mlp-default/predict-dance0/renders -o output.mp4 -d png -f 30
```