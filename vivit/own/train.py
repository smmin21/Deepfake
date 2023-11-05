import numpy as np
import torch
import logging
from omegaconf import OmegaConf
from argparse import ArgumentParser


from transformers import VivitConfig, VivitForVideoClassification
from dataloader.VideoDataset import get_video_dataset
from engine.trainer import Trainer


parser = ArgumentParser('Deepface Training Script')
parser.add_argument('config', type=str, help='config file path')


if __name__ == '__main__':
    np.random.seed(0)
    args = parser.parse_args()
    opt = OmegaConf.load(args.config)
    
    # load dataloader
    dataloader = get_video_dataset(opt.DATA)
    # model
    configuration = VivitConfig(num_labels=2, num_frames=opt.DATA.num_frames)
    model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400", config=configuration, ignore_mismatched_sizes=True)
    
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


