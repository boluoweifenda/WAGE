# Training and Inference with Integers in Deep Neural Networks

Code example for the ICLR 2018 oral paper

## Prerequisites
- NVIDIA GPU + CUDA + CuDNN
- Tensorflow (GPU version)
- python2.7
- tqdm


## Data
Download and generate CIFAR10 dataset: 
```bash
cd dataSet/
python CIFAR10.py
```

## Config
Change your configurations in the file
```bash
gedit source/Option.py
```
## Train
Start training:
```bash
cd source/
python Top.py
```
## Citation
If you find this paper or this repository helpful, please cite it:
```bash
@inproceedings{
wu2018training,
title={Training and Inference with Integers in Deep Neural Networks},
author={Shuang Wu and Guoqi Li and Feng Chen and Luping Shi},
booktitle={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=HJGXzmspb},
} 
```

