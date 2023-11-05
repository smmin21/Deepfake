import os #, sys
import logging
from omegaconf import OmegaConf
from argparse import ArgumentParser

import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

from model.model import CNNModel, DANN_InceptionV3
from engine.tester import Tester
from engine.trainer import Trainer
from dataloader.FFImageDataset import get_image_dataset

parser = ArgumentParser('Deepface Training Script')
parser.add_argument('config', type=str, help='config file path')

if __name__ == '__main__':
    args = parser.parse_args()
    opt = OmegaConf.load(args.config)
    
    # Set random seeds
    ramdom_seed = 1234
    deterministic = True
    
    random.seed(ramdom_seed)
    np.random.seed(ramdom_seed)
    torch.manual_seed(ramdom_seed)
    torch.cuda.manual_seed_all(ramdom_seed)
    
    # Load Dataloader
    dataloader = get_image_dataset(opt.DATA)
    
    # Model
    model = DANN_InceptionV3()
    
    
    # Logger
    # log train/val loss, acc, etc.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG) 
    console_logging_format = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_logging_format)
    logger.addHandler(console_handler)

    # BANMo System    
    trainer = Trainer(opt, dataloader, model, logger)

    # train
    trainer.train()
    
    # test
    tester = Tester(opt, dataloader, model, logger)
    tester.test("test")