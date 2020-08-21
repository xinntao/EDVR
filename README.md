This is the old version of EDVR. It is out-of-date.

Please use the newest [BasicSR](https://github.com/xinntao/BasicSR).

# EDVR [[BasicSR]](https://github.com/xinntao/BasicSR)

#### [Paper](https://arxiv.org/abs/1905.02716) | [Project Page](https://xinntao.github.io/projects/EDVR)

### Video Restoration with Enhanced Deformable Convolutional Networks
By [Xintao Wang](https://xinntao.github.io/), Kelvin C.K. Chan, [Ke Yu](https://yuke93.github.io/), [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=en), [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/)

EDVR won all four tracks in [NTIRE 2019 Challenges on **Video Restoration and Enhancement**](http://www.vision.ee.ethz.ch/ntire19/) (CVPR19 Workshops).

### Highlights
- **A unified framework** suitable for various video restoration tasks, *e.g.*, super-resolution, deblurring, denoising, *etc*
- **State of the art**: Winners in NTIRE 2019 Challenges on Video Restoration and Enhancement
- **Multi-GPU (distributed) training**

### Updates
[2019-06-28] Provide training logs and pretrained model for EDVR-M. Check [here](https://github.com/xinntao/EDVR/wiki/Testing-and-Training). <br/>
[2019-06-28] Support [TOFlow testing (SR)](http://toflow.csail.mit.edu/) (converted from [officially released models](https://github.com/anchen1011/toflow)). <br/>
[2019-06-12] Add training codes.<br/>
[2019-06-11] Add data preparation in [wiki](https://github.com/xinntao/EDVR/wiki/Prepare-datasets-in-LMDB-format).<br/>
[2019-06-07] Support [DUF testing](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.pdf) (converted from [officially released models](https://github.com/yhjo09/VSR-DUF)). <br/>
[2019-05-28] Release testing codes.

## Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- [Deformable Convolution](https://arxiv.org/abs/1703.06211). We use [Charles Shang](https://github.com/CharlesShang)'s [DCNv2](https://github.com/CharlesShang/DCNv2) implementation. Please first compile it.
  ```
  cd ./codes/models/modules/DCNv2
  bash make.sh
  ```
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard:
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`

## Dataset Preparation
We use datasets in LDMB format for faster IO speed. Please refer to [wiki](https://github.com/xinntao/EDVR/wiki/Prepare-datasets-in-LMDB-format) for more details.

## Get Started
Please see [wiki](https://github.com/xinntao/EDVR/wiki/Testing-and-Training) for the basic usage, *i.e.,* training and testing.
## Model Zoo and Baselines
Results and pre-trained models are available in the [wiki-Model zoo](https://github.com/xinntao/EDVR/wiki/Model-Zoo).

## Contributing
We appreciate all contributions. Please refer to [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/CONTRIBUTING.md) for contributing guideline.

**Python code style**<br/>
We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style. We use [flake8](http://flake8.pycqa.org/en/latest/) as the linter and [yapf](https://github.com/google/yapf) as the formatter. Please upgrade to the latest yapf (>=0.27.0) and refer to the [yapf configuration](https://github.com/xinntao/EDVR/blob/master/.style.yapf) and [flake8 configuration](https://github.com/xinntao/EDVR/blob/master/.flake8).

> Before you create a PR, make sure that your code lints and is formatted by yapf.

## Citation
```
@InProceedings{wang2019edvr,
  author    = {Wang, Xintao and Chan, Kelvin C.K. and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title     = {EDVR: Video restoration with enhanced deformable convolutional networks},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  month     = {June},
  year      = {2019},
}
@Article{tian2018tdan,
  author    = {Tian, Yapeng and Zhang, Yulun and Fu, Yun and Xu, Chenliang},
  title     = {TDAN: Temporally deformable alignment network for video super-resolution},
  journal   = {arXiv preprint arXiv:1812.02898},
  year      = {2018},
}
```
