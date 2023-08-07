import os #, sys
import logging
from omegaconf import OmegaConf
from argparse import ArgumentParser

import torch.nn as nn
from torchvision import models

from experiment.engine.tester import Tester
from experiment.engine.trainer import Trainer
from experiment.dataloader.ImageDataset import get_image_dataset
from experiment.dataloader.VideoDataset import get_video_dataset
from experiment.model.model import InceptionNet

parser = ArgumentParser('Deepface Training Script')
parser.add_argument('config', type=str, help='config file path')

if __name__ == '__main__':
    args = parser.parse_args()
    opt = OmegaConf.load(args.config)

    # Load Dataloader
    if opt.DATA.is_image:
        dataloader = get_image_dataset(opt.DATA)
    else:
        dataloader = get_video_dataset(opt.DATA)
    
    # Model
    # model = InceptionNet(num_classes = opt.MODEL.num_classes)
    model = models.inception_v3(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

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
    tester.test("train")
    tester.test("val")
    