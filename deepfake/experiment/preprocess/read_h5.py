import h5py
import numpy as np

if __name__ == '__main__':
    f = h5py.File('/root/code/deepfake/experiment/preprocess/YouTube-real.h5', 'r')
    # f = h5py.File('/root/workspace/deepfake/faceforensics/dataset/image/Deepfakes.h5', 'r')
    print(f.keys())
    print("num of videos: ", len(f.keys()))
    
    # for celeb dataset
    # for key in f.keys():
    #     print(key, f[key]) # frame_num x 256 x 256 x 3
        
        
    # f = h5py.File('/workspace/deepfake/face_forensics.h5', 'r')
    # print(f.keys())
    # print(f['Original'].keys())
    # videos = list(f.keys())
    # print(f['Original'][videos[0]].shape)
    # print(f['Original'][videos[1]].shape)
    # print(f['Original'][videos[2]].shape)