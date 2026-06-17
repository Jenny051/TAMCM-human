### Environment
# conda create -n bodyfit python=3.8.13
# conda activate bodyfit

### Core runtime
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

### External source packages
# Install PyTorch3D, a compatible body-model prior package, and voxel preprocessing
# extensions from your approved mirrors or internal package cache.

pip install cython
pip install hydra-core==1.2.0 pytorch-lightning==1.5.10 open3d==0.15.2 trimesh==3.13.0 opencv-python==4.6.0.66 scikit-image==0.19.3 robust-laplacian==0.2.4 plotly==5.10.0
pip install scikit-learn==1.1.2
pip install -e .
