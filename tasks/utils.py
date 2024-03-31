

import torch
import torchvision

from transformers import AutoImageProcessor
from datasets import load_dataset

from models.cifnet import (
    CifNetForImageClassification,
    CifNetConfig,
)

def setup_model(model_name_or_path, ):
    model_config = CifNetConfig.from_pretrained(model_name_or_path)
    model = CifNetForImageClassification(model_config)
    # model = CifNetForImageClassification.from_pretrained(model_name_or_path, config=model_config)
    processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    return model, processor


def setup_dataset():
    # dataset = torchvision.datasets.CIFAR10("DATAS/CIFAR10", download=True)
    dataset = load_dataset("cifar10", )
    return dataset

