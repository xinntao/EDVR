# EDVR 
#### [Paper](https://arxiv.org/abs/1905.02716) | [Project Page](https://xinntao.github.io/projects/EDVR) | [Open VideoRestoration Doc (under construction)](https://xinntao.github.io/open-videorestoration/) 
### Video Restoration with Enhanced Deformable Convolutional Networks
By [Xintao Wang](https://xinntao.github.io/), Kelvin C.K. Chan, [Ke Yu](https://yuke93.github.io/), [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=en), [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/)

EDVR won all four tracks in [NTIRE 2019 Challenges on **Video Restoration and Enhancement**](http://www.vision.ee.ethz.ch/ntire19/) (CVPR19 Workshops). 


Testing codes have been released. 

### Dependencies

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch = 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- [Deformable Convolution](https://arxiv.org/abs/1703.06211). We use [Charles Shang](https://github.com/CharlesShang)'s [DCNv2](https://github.com/CharlesShang/DCNv2) implementation. Please first compile it. 
  ```
  cd ./codes/models/modules/DCNv2
  bash make.sh
  ```
- Python packages: `pip install numpy opencv-python lmdb`

### How to test

1. Download the [pretrained models](https://drive.google.com/open?id=1pFMrZQaqSeBJqGHSjzAlHvJ4jzHnKleE) and [testing datasts](https://drive.google.com/open?id=10-gUO6zBeOpWEamrWKCtSkkUFukB9W5m).

2. Run `test_Vid4_REDS4_with_GT.py`


Test lines.
