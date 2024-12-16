# Guide

## Structure

- `exp/` stores all the experiment results
- `work/` stores all the scripts and documents that we have added
- `tmp/` stores the raw downloaded dataset for backup in case the processed data are polluted

The rest are within the original 3DGS-Avatar project, only important ones are listed here:
- `configs/` stores all the configuration files
- `dataset/` stores all the dataset scripts
- `models/` stores all the model scripts

## Environment

<!-- run `env.sh` -->

TO BE DONE

## Model

Too many manual operations to be written here.

## Dataset

Currently, all zju_mocap datasets and its relevant data have been downloaded at `~/data`. People snapshot haven't been processed due to lack of processing scripts.

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


```bash
CUDA_VISIBLE_DEVICES=1 python train.py dataset=zjumocap_377_mono
```

### Eval

Take care of the available GPUs, run `nvidia-smi` beforehand to watch GPU utilization.
- Valid choices of `dataset.test_mode`: view, video, all

```bash
CUDA_VISIBLE_DEVICES=1 python render.py mode=test dataset.test_mode=view dataset=zjumocap_377_mono
```

### Predict

Predict is used to predict novel poses from the given monocular video.

```bash
CUDA_VISIBLE_DEVICES=1 python render.py mode=predict dataset.predict_seq=0 dataset=zjumocap_377_mono
```

## Visualize

`video.py` is a simple script to convert images to video. To use it, first switch to `tool` environment. Then give the right arguments(See them using `python video.py -h`).

```bash
conda activate tool
python video.py -h

# Example command:
# python work/video.py -i /users/xch/multimodal/3dgs-avatar/exp/zju_377_mono-direct-mlp_field-ingp-shallow_mlp-default/predict-dance0/renders -o videos/output.mp4 -d png -f 30
```