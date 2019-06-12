# EDVR 
#### [Paper](https://arxiv.org/abs/1905.02716) | [Project Page](https://xinntao.github.io/projects/EDVR) | [Open VideoRestoration Doc (under construction)](https://xinntao.github.io/open-videorestoration/) 
### Video Restoration with Enhanced Deformable Convolutional Networks
By [Xintao Wang](https://xinntao.github.io/), Kelvin C.K. Chan, [Ke Yu](https://yuke93.github.io/), [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=en), [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/)

EDVR won all four tracks in [NTIRE 2019 Challenges on **Video Restoration and Enhancement**](http://www.vision.ee.ethz.ch/ntire19/) (CVPR19 Workshops). 

### Highlights
- **A unified framework** suitable for various video restoration tasks, *e.g.*, super-resolution, deblurring, denoising, *etc*
- **State of the art**: Winners in NTIRE 2019 Challenges on Video Restoration and Enhancement
- **Multi-GPU (distributed) training**

### Updates
[2019-06-11] Add data preparation in [wiki](https://github.com/xinntao/EDVR/wiki/Prepare-datasets-in-LMDB-format).<br/>
[2019-06-07] Support [DUF testing](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.pdf) (converted from [officially released models](https://github.com/yhjo09/VSR-DUF)). <br/>
[2019-05-28] Release testing codes.

## Dependencies and Installation
Please refer to [wiki](https://github.com/xinntao/EDVR/wiki/Dependencies-and-installation) for dependencies and installation.

## Dataset Preparation
We use datasets in LDMB format for faster IO speed. Please refer to [wiki](https://github.com/xinntao/EDVR/wiki/Prepare-datasets-in-LMDB-format) for more details.

## Get Started
Please see [wiki]() (TODO) for the basic usage, *i.e.,* training and testing.
## Model Zoo and Baselines
Results and pre-trained models are available in the [wiki-Model zoo]() (TODO).

## Contributing
We appreciate all contributions. Please refer to [wiki]() (TODO) for the contributing guideline.

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
