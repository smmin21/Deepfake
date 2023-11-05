import os
import h5py
from PIL import Image
import pandas as pd
import numpy as np
import pdb
import torch
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image


def get_video_dataset(opt):

    augmentation = T.Compose([
        T.Resize((opt.image_size, opt.image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    
    # For cross evaluation
    # test_data_path에 경로 있으면 cross evaluation
    if opt.test_data_path == 'None':
        test_data_name = opt.train_data_name
        test_data_path = opt.train_data_path
    else:
        test_data_name = opt.test_data_name
        test_data_path = opt.test_data_path

    # train dataset
    train_dataset = CelebDF(opt.train_data_path, split='train', image_size=128, transform=augmentation, num_frames=opt.num_frames)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=opt.batch_size, 
                                  shuffle=True, 
                                  num_workers=opt.num_workers)
    # val dataset
    val_dataset = CelebDF(opt.train_data_path, split='val', image_size=128, transform=augmentation, num_frames=opt.num_frames)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers) 
    
    # test dataset
    test_dataset = CelebDF(test_data_path, split='test', image_size=128, transform=augmentation, num_frames=opt.num_frames)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)
    
    dataset = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    return dataset

def load_video_frames(video_dir, transform):
    frame_keys = sorted(os.listdir(video_dir))
    frames = []
    for frame_key in frame_keys:
        frame = Image.open(os.path.join(video_dir, frame_key))
        frame = transform(frame)
        frames.append(frame)
    return frames

    
class CelebDF(Dataset):
    def __init__(self, root_path, split='train', image_size=128, transform=None, num_frames=16):
        mode_dict = {'train': 'train',
                     'meta-train': 'train',
                     'val': 'val',
                     'meta-val': 'val',
                     'test': 'test',
                     'meta-test': 'test',
                     }
        self.path = root_path
        self.mode = mode_dict[split]

        self.transform = transform
        self.num_frames = num_frames

        self.videos = []
        self.labels = []
        self.n_classes = 2
        self.test_list = None
        
        # set iteration path
        iter_path = [os.path.join(self.path, 'Celeb-real', 'crop_jpg'),
                     os.path.join(self.path, 'Celeb-synthesis', 'crop_jpg'),
                     os.path.join(self.path, 'YouTube-real', 'crop_jpg')]
        with open(self.path + "/List_of_testing_videos.txt", "r") as f:
            self.test_list = f.readlines()
            self.test_list = [x.split("/")[-1].split(".mp4")[0] for x in self.test_list]
        
        # iteration
        for i, each_path in enumerate(iter_path):
            video_keys = sorted(os.listdir(each_path))
            # test 데이터셋이 정해져 있는 경우 (Celeb-DF)
            if self.mode == 'test': 
                video_keys = [x for x in self.test_list if x in video_keys]
            elif self.mode == 'train':
                video_keys = [x for x in video_keys if x not in self.test_list]
                video_keys = video_keys[:int(len(video_keys)*0.8)]
            elif self.mode == 'val':
                video_keys = [x for x in video_keys if x not in self.test_list]
                video_keys = video_keys[int(len(video_keys)*0.8):]
            
            video_dirs = [os.path.join(each_path, video_key) for video_key in video_keys]
            self.videos += video_dirs
            self.labels += [0 if each_path.find('real') >= 0 else 1 for _ in range(len(video_keys))] # 0: real, 1: fake

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        video_dir = self.videos[index]
        frames = load_video_frames(video_dir, self.transform)
        
        if len(frames) > self.num_frames:
            frame_indices = torch.linspace(0, len(frames)-1, self.num_frames, dtype=torch.int64)
            frames = [frames[i] for i in frame_indices]
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
            
        frames = torch.stack(frames)
        data = {'frame': frames, 'label': self.labels[index]}
        return data
    
if __name__ == "__main__":
    data_path = '/root/result/h5_result'
    bbox_path = '/root/deepfakedatas'
    dataset = CelebDF(data_path, bbox_path, crop_ratio=1.7, mode='train')
    data = dataset[1500]
