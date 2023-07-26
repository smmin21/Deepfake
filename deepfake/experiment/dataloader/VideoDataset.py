import os
import h5py
from PIL import Image
import pandas as pd
import numpy as np
import pdb
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image


def get_video_dataset(opt):

    augmentation = T.Compose([
        T.Resize((opt.image_size, opt.image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    # train dataset
    train_dataset = VideoDataset(opt.data_path, opt.bbox_path, 
                                 crop_ratio=opt.crop_ratio, mode='train', transforms=augmentation, frame_num=opt.frame_num)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=opt.batch_size, 
                                  shuffle=True, 
                                  num_workers=opt.num_workers)
    # val dataset
    val_dataset = VideoDataset(opt.data_path, opt.bbox_path, 
                                crop_ratio=opt.crop_ratio, mode='val', transforms=augmentation, frame_num=opt.frame_num)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers)
    
    # test dataset
    test_dataset = VideoDataset(opt.data_path, opt.bbox_path, 
                                crop_ratio=opt.crop_ratio, mode='test', transforms=augmentation, frame_num=opt.frame_num)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)
    
    dataset = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    return dataset


# TODO : VideoDataset from h5 file
class VideoDataset(Dataset):
    def __init__(self, path, bbox_path, crop_ratio=1.2, mode='train', transforms=None, frame_num=10):
        self.path = path
        self.bbox_path = bbox_path ##
        self.crop_ratio = crop_ratio ##
        self.mode = mode
        self.frame_num = frame_num
        self.mtype = ['Original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        if transforms is None:
           self.transforms = T.ToTensor()
        else:
           self.transforms = transforms

        self.videos = []
        self.labels = []
        self.mtype_index = []
        self.data_list = None
        for i, m in enumerate(self.mtype):
            path = os.path.join(self.path, f'{m}.h5')
            with h5py.File(path, 'r') as f:
                video_keys = sorted(list(f.keys()))
                print(f.keys())
                if self.mode == 'train':
                    video_keys = video_keys[:int(len(video_keys)*0.8)]
                elif self.mode == 'val':
                    video_keys = video_keys[int(len(video_keys)*0.8):int(len(video_keys)*0.9)]
                elif self.mode == 'test':
                    video_keys = video_keys[int(len(video_keys)*0.9):]

                self.videos += video_keys
                self.labels += [0 if path.find('Original') >= 0 else 1 for _ in range(len(video_keys))] # 0: real, 1: fake
                self.mtype_index += [i for _ in range(len(video_keys))]

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        if self.data_list is None:
            self.data_list = []
            for m in self.mtype:
                self.data_list.append(h5py.File(os.path.join(self.path, f'{m}.h5'), 'r', swmr=True))

        data_file = self.data_list[self.mtype_index[index]] # Original file
        clip = data_file[self.videos[index]] # 특정 비디오
        start_frame_idx = np.random.randint(0, len(clip)-self.frame_num)
        frames = [clip[start_frame_idx + i] for i in range(self.frame_num)]
        frames = torch.stack([self.transforms(Image.fromarray(frame[...,::-1])) for frame in frames])
        data = {'frame': frames, 'label': self.labels[index]}  # t x c x (h x w)
        return data
    
    
if __name__ == "__main__":
    data_path = '/root/result/h5_result'
    bbox_path = '/root/deepfakedatas'
    dataset = VideoDataset(data_path, bbox_path, crop_ratio=1.7, mode='train')
    data = dataset[1500]