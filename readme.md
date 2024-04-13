# CIFNET Residual Neural Networks for CIFAR-10
This repository contains the code and models for the CIFNET series, a set of modified residual neural networks (ResNets) designed for high-performance image classification on the CIFAR-10 dataset. The models range from CIFNET-18 to CIFNET-98, with detailed augmentations and training strategies to maximize accuracy. This project was developed as part of a mini-project at NYU Tandon School of Engineering.
## Installation

To set up this project, clone the repository to your local machine:

```bash
git clone https://github.com/ermu2001/CIF.git
cd CIF
```

Regularly install torch, transformers (and accelerate for training).


## Downloading Model Weights
To download the pre-trained weights for the CIFNET models along with our training logs, run the following command:
```bash
python3 python_scripts/hf.py
```

## Running Training
To train the models from scratch or fine-tune them, use the provided training script:
```bash
chmod +x scripts/train.sh
./scripts/train.sh
```

## Running Evaluation
Reference `explore/test.ipynb`
```bash
cd explore
jupyter notebook test.ipynb
```

## Inference
To run inference on a trained model, use the following command:
```bash
jupyter notebook demo.ipynb
```

## Monitor Training with Tensorboard
To monitor the training process using Tensorboard, execute the following command and visit http://localhost:6006:
```bash
tensorboard --logdir ./OUTPUTS
```

The files in our huggingface repo also contains every tensorboard training log of our trained models.

## Model Details
The CIFNET models implemented in this repository are designed to balance the depth and complexity of the network to ensure high accuracy without excessive computational cost. Here are some highlights:

CIFNET-22 (OUTPUTS/cifnet-18-cucumber--lr0.001--d4_256) and CIFNET-24 (OUTPUTS/cifnet-18-durian--lr0.001--d4_256): Basic ResNet architectures with moderate depth.
CIFNET-98 (OUTPUTS/best_0409_2--lr0.001--BetterAug): Includes deeper layers and employs data augmentation techniques to achieve the highest performance.
For detailed model architectures and training processes, please refer to the models directory and the training scripts.

## Acknowledgements
Thanks to the NYU Tandon School of Engineering for supporting this project.
Special thanks to all contributors and researchers who have shared their insights into deep learning and residual networks.
