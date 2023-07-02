import os
import h5py
from tqdm import tqdm
import argparse

import cv2
import numpy as np


def extract_frames(video_path, group):
    video_list = sorted(os.listdir(video_path))
    print(video_path)
    print('num_videos: ', len(video_list))
    for j, video in enumerate(video_list):
        frame_list = sorted(os.listdir(os.path.join(video_path, video)))
        data = []
        for i, frame in enumerate(frame_list):
            frame_path = os.path.join(video_path, video, frame)
            frame = cv2.imread(frame_path)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            data.append(frame)     
        data = np.stack(data, axis=0)
        group.create_dataset(f'{video}', data=data, dtype=np.uint8)
        # group.create_dataset(f'{video}', data=data, dtype=np.uint8, compression='gzip',compression_opts=9)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', 
                        type=str, 
                        default='Original', 
                        choices = ['Original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
                        help='dataset type')
    args = parser.parse_args()
    
    if args.dtype == 'Original':
        video_path = '/root/deepfakedatas/original_sequences/raw/png'
    else:
        video_path = '/root/deepfakedatas/manipulated_sequences/{}/raw/png'.format(args.dtype)

    # if args.dtype == 'Original':
    #     video_path = '/media/NAS2/CIPLAB/dataset/FaceForensics-master/download/original_sequences/raw/png'
    # else:
    #     video_path = '/media/NAS2/CIPLAB/dataset/FaceForensics-master/download/manipulated_sequences/{}/raw/png'.format(args.dtype)

    output_path =  f'{args.dtype}.h5'
    db = h5py.File(output_path, 'w')
    extract_frames(video_path, db)
