# iMAP SLAM in PyTorch

This is an implementation of [iMAP SLAM](https://edgarsucar.github.io/iMAP/) in PyTorch, which is a cleaned up version of [ueda0319's iMAP](https://github.com/ueda0319/iMAP_pytorch) implementation.  Thanks to ueda0319 for the original implementation!  I mostly went through refactoring, renaming for clarity, and added lots of comments as I learned through reading his code.  Hopefully this can help others to learn NERF and NERF based SLAM algorithms.

![tum_test](https://github.com/ueda0319/iMAP_pytorch/wiki/imgs/teddy.gif)

## dependency
- PyTorch 1.8
- opencv-python
- numpy

## Quick start
Download RGBD dataset from TUM(https://vision.in.tum.de/data/datasets/rgbd-dataset/download)

run with following commands
```
python run_slam_tum.py path/to/rgbd_dataset_freiburg1_teddy/
```
