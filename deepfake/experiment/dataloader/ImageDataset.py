import os
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import pdb
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image



def get_dataset(opt):

    augmentation = T.Compose([
        T.Resize((opt.image_size, opt.image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    # train dataset
    train_dataset = ImageDataset(opt.data_path, opt.bbox_path, 
                                 crop_ratio=opt.crop_ratio, mode='train', transforms=augmentation)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=opt.batch_size, 
                                  shuffle=True, 
                                  num_workers=opt.num_workers)
    # val dataset
    val_dataset = ImageDataset(opt.data_path, opt.bbox_path, 
                                crop_ratio=opt.crop_ratio, mode='val', transforms=augmentation)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers) 
    
    # test dataset
    test_dataset = ImageDataset(opt.data_path, opt.bbox_path, 
                                crop_ratio=opt.crop_ratio, mode='test', transforms=augmentation)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)
    
    dataset = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    return dataset


# TODO : ImageDataset from h5 file
class ImageDataset(Dataset):
    def __init__(self, path, bbox_path, crop_ratio=1.2, mode='train', transforms=None):
        self.path = path
        self.bbox_path = bbox_path
        self.crop_ratio = crop_ratio
        self.mode = mode
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
                if self.mode == 'train':
                    video_keys = video_keys[:int(len(video_keys)*0.8)]
                elif self.mode == 'val':
                    video_keys = video_keys[int(len(video_keys)*0.8):int(len(video_keys)*0.9)]
                elif self.mode == 'test':
                    video_keys = video_keys[int(len(video_keys)*0.9):]

                self.videos += video_keys
                self.labels += [0 if path.find('Original') >= 0 else 1 for _ in range(len(video_keys))] # 0: real, 1: fake
                self.mtype_index += [i for _ in range(len(video_keys))]
        
        self.bbox_original = pd.read_json(os.path.join(self.bbox_path, 'Original_bbox.json'))
        self.bbox_deepfakes = pd.read_json(os.path.join(self.bbox_path, 'Deepfakes_bbox.json'))
        self.bbox_face2face = pd.read_json(os.path.join(self.bbox_path, 'Face2Face_bbox.json'))
        self.bbox_faceswap = pd.read_json(os.path.join(self.bbox_path, 'FaceSwap_bbox.json'))
        self.bbox_neuraltextures = pd.read_json(os.path.join(self.bbox_path, 'NeuralTextures_bbox.json'))

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        if self.data_list is None:
            self.data_list = []
            for m in self.mtype:
                self.data_list.append(h5py.File(os.path.join(self.path, f'{m}.h5'), 'r', swmr=True))
        data_file = self.data_list[self.mtype_index[index]] # Original file
        clip = data_file[self.videos[index]]
        frame_idx = np.random.randint(0, len(clip))
        frame = clip[frame_idx]
        frame = Image.fromarray(frame[...,::-1])    # BGR2RGB

        # TODO : bbox face crop
        # bbox path
        bbox_dfs = [
            self.bbox_original,
            self.bbox_deepfakes,
            self.bbox_face2face,
            self.bbox_faceswap,
            self.bbox_neuraltextures,
        ]
        bbox_df = bbox_dfs[self.mtype_index[index]]
        try:
            bbox = bbox_df.loc[frame_idx, int(self.videos[index])]["bbox"]
        except:
            bbox = bbox_df.loc[frame_idx, str(self.videos[index])]["bbox"]
        width, height = frame.size
        if bbox :
            bbox = [bbox[0]+bbox[2]*(1-self.crop_ratio)/2, bbox[1]+bbox[3]*(1-self.crop_ratio)/2, bbox[0]+bbox[2]*(1-self.crop_ratio)/2+bbox[2]*self.crop_ratio, bbox[1]+bbox[3]*(1-self.crop_ratio)/2+bbox[3]*self.crop_ratio]
        else :
            bbox = [0, 0, width, height]          
        frame_cropped = frame.crop(bbox)  
        
        # frame_cropped.save(f'/workspace/result/cropped_img/{frame_idx}.jpg', format='JPEG')
        
        frame = self.transforms(frame_cropped)
        data = {'frame': frame, 'label': self.labels[index]}
        return data
    
if __name__ == "__main__":
    data_path = '/workspace/datasets'
    bbox_path = '/workspace/deepfakedatas'
    dataset = ImageDataset(data_path, bbox_path, crop_ratio=1, mode='train')
    data = dataset[2000]