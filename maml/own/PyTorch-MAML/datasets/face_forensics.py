import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import numpy as np
import torch

from .datasets import register
from .transforms import get_transform

@register('face-forensics')
class FaceForensics(Dataset):
    def __init__(self, root_path, split='train', image_size=84,
                 normalization=True, transform=None):
        mode_dict = {'train': 'train',
                     'meta-train': 'train',
                     'val': 'val',
                     'meta-val': 'val',
                     'test': 'test',
                     'meta-test': 'test',
                     }
        self.path = root_path
        self.mode = mode_dict[split]
        
        if normalization:
          self.norm_params = {'mean': [0.485, 0.456, 0.406],
                              'std':  [0.229, 0.224, 0.225]}   # ImageNet statistics
        else:
          self.norm_params = {'mean': [0., 0., 0.],
                              'std':  [1., 1., 1.]}

        self.transform = get_transform(transform, image_size, self.norm_params)

        self.videos = []
        self.labels = []
        self.n_classes = 2
        # self.test_list = None
        
        # set iteration path
        iter_path = [os.path.join(self.path, 'original_sequences', 'raw', 'crop_jpg'), 
                     os.path.join(self.path, 'manipulated_sequences', 'Deepfakes', 'raw', 'crop_jpg'),
                     os.path.join(self.path, 'manipulated_sequences', 'Face2Face', 'raw', 'crop_jpg'),
                     os.path.join(self.path, 'manipulated_sequences', 'FaceSwap', 'raw', 'crop_jpg'),
                     os.path.join(self.path, 'manipulated_sequences', 'NeuralTextures', 'raw', 'crop_jpg')]
        
        # iteration
        for i, each_path in enumerate(iter_path):
            video_keys = sorted(os.listdir(each_path))
            # test 데이터셋이 정해져있지 않은 경우 (FF++ & DFDC)
            if self.mode == 'train':
                video_keys = video_keys[:int(len(video_keys)*0.8)]
            elif self.mode == 'val':
                video_keys = video_keys[int(len(video_keys)*0.8):int(len(video_keys)*0.9)]
            elif self.mode == 'test':
                video_keys = video_keys[int(len(video_keys)*0.9):]
            
            video_dirs = [os.path.join(each_path, video_key) for video_key in video_keys]
            self.videos += video_dirs
            self.labels += [0 if each_path.find('original') >= 0 else 1 for _ in range(len(video_keys))] # 0: real, 1: fake

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        video_dir = self.videos[index]
        frame_keys = sorted(os.listdir(video_dir))
        frame_key = random.choice(frame_keys)
        frame = Image.open(os.path.join(video_dir, frame_key))
        frame = self.transform(frame)
        data = {'frame': frame, 'label': self.labels[index]}
        return data


@register('meta-face-forensics')
class MetaFaceForensics(FaceForensics):
    def __init__(self, root_path, split='train', image_size=84,
                 normalization=True, transform=None, val_transform=None, 
                 n_batch=200, n_episode=4, n_way=2, n_shot=1, n_query=15):
        super(MetaFaceForensics, self).__init__(root_path, split, image_size,
                                                normalization, transform)
        
        self.n_batch = n_batch
        self.n_episode = n_episode
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        
        self.catlocs = tuple()
        for cat in range(self.n_classes):
            self.catlocs += (np.argwhere(np.array(self.labels) == cat).reshape(-1),)
        
        self.val_transform = get_transform(
            val_transform, image_size, self.norm_params)
        
    def __len__(self):
        return self.n_batch * self.n_episode
    
    def __getitem__(self, index):
        shot, query = [], []
        cats = np.random.choice(self.n_classes, self.n_way, replace=False)
        for c in cats:
            c_shot, c_query = [], []
            idx_list = np.random.choice(
                self.catlocs[c], self.n_shot + self.n_query, replace=False)
            shot_idx, query_idx = idx_list[:self.n_shot], idx_list[-self.n_query:]
            for idx in shot_idx:
                video_dir = self.videos[idx]
                frame_keys = sorted(os.listdir(video_dir))
                frame_key = random.choice(frame_keys)
                frame = Image.open(os.path.join(video_dir, frame_key))
                frame = self.transform(frame)
                c_shot.append(frame)
            for idx in query_idx:
                video_dir = self.videos[idx]
                frame_keys = sorted(os.listdir(video_dir))
                frame_key = random.choice(frame_keys)
                frame = Image.open(os.path.join(video_dir, frame_key))
                frame = self.val_transform(frame) if self.val_transform is not None else self.transform(frame)
                c_query.append(frame)
            shot.append(torch.stack(c_shot))
            query.append(torch.stack(c_query))
        
        shot = torch.cat(shot, dim=0)             # [n_way * n_shot, C, H, W]
        query = torch.cat(query, dim=0)           # [n_way * n_query, C, H, W]
        
        cls = torch.arange(self.n_way)[:, None]
        shot_labels = cls.repeat(1, self.n_shot).flatten()    # [n_way * n_shot]
        query_labels = cls.repeat(1, self.n_query).flatten()  # [n_way * n_query]
        
        return shot, query, shot_labels, query_labels
    
    
    
        
        
        