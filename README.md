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


