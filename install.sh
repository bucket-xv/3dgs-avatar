git clone --recursive https://github.com/mikeqzy/3dgs-avatar-release.git
cd 3dgs-avatar-release
conda env create -f environment.yml
conda activate 3dgs-avatar
conda install nvidia/label/cuda-11.6.2::cuda-toolkit
# install tinycudann
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch