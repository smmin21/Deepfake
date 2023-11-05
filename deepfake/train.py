import os #, sys
import logging
from omegaconf import OmegaConf
from argparse import ArgumentParser

import torch.nn as nn
from torchvision import models

from experiment.engine.tester import Tester
from experiment.engine.tester_video import VideoTester
from experiment.engine.trainer import Trainer
from experiment.dataloader.FFImageDataset import get_image_dataset
from experiment.dataloader.VideoDataset import get_video_dataset
from experiment.dataloader.VideoTestDataset import get_video_test_dataset
from experiment.model.model import InceptionNet
from torchsummary import summary
import random
import numpy as np
import torch

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
    if opt.DATA.is_image:
        dataloader = get_image_dataset(opt.DATA)
    else:
        dataloader = get_video_dataset(opt.DATA)
    if opt.TEST.is_image:
        test_dataloader = dataloader
    else:
        test_dataloader = get_video_test_dataset(opt.DATA)
    
    # Model
    # model = models.inception_v3(pretrained=True)
    model = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    summary(model, (3, opt.DATA.image_size, opt.DATA.image_size), device='cpu')

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
    if opt.TEST.is_image:
        tester = Tester(opt, test_dataloader, model, logger)
    else:
        tester = VideoTester(opt, test_dataloader, model, logger)
    tester.test("test")
    # tester.test("train") # test 데이터셋의 train 영상들
    # tester.test("val")   # test 데이터셋의 val 영상들