import os
import h5py
import argparse
from PIL import Image
import cv2
import numpy as np
import pandas as pd


def extract_frames_ff(video_path, group, dtype, crop_ratio):
    # save_path = f'/root/result/h5_test/ff/{dtype}'
    save_path = f'{video_path}/../crop_jpg' ##########
    os.makedirs(save_path, exist_ok=True)
    
    bbox_df = pd.read_json(f'/root/deepfakedatas/{dtype}_bbox.json')
    video_list = sorted(os.listdir(video_path))
    print(video_path)
    print('num_videos: ', len(video_list))
    for j, video in enumerate(video_list):
        frame_list = sorted(os.listdir(os.path.join(video_path, video))) # frame_list: 0001.png, 0002.png, ...
        
        data = []
        for i, frame_id in enumerate(frame_list):
            frame_path = os.path.join(video_path, video, frame_id)                
            frame = cv2.imread(frame_path)
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # print(i, f'{video}', frame_id.split('.png')[0][-4:])
            
            # find bbox
            frame_id = frame_id.split('.png')[0][-4:] # 000_0001.png -> 0001
            try:
                bbox = bbox_df.loc[int(f'{frame_id}'), int(f'{video}')]["bbox"] 
            except:
                try:
                    bbox = bbox_df.loc[f'{frame_id}', f'{video}']["bbox"]
                except:
                    print(f"find error : video-{video}, frame-{frame_id}")
                    bbox = prev_bbox
            try:
                bbox = [bbox[0]+bbox[2]/2-bbox[3]*crop_ratio/2, 
                        bbox[1]+bbox[3]*(1-crop_ratio)/2, 
                        bbox[0]+bbox[2]/2+bbox[3]*crop_ratio/2, 
                        bbox[1]+bbox[3]*(1+crop_ratio)/2]
            except:
                print(f"full size bbox : video-{video}, frame-{frame_id}")
                bbox = prev_bbox
            prev_bbox = bbox
            frame = frame.crop(bbox)
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            
            try:
                frame = cv2.resize(frame, (256, 256))
            except:
                print(bbox)
                print(frame)
            # if int(frame_id) % 100 == 0:
            #     cv2.imwrite(os.path.join(save_path, f'{video}_{frame_id}.png'), frame)
            tmp_save_path = os.path.join(save_path, video)
            os.makedirs(tmp_save_path, exist_ok=True)
            cv2.imwrite(os.path.join(tmp_save_path, f'{frame_id}.jpg'), frame)
            #################################################################
            data.append(frame)     
        data = np.stack(data, axis=0)
        # group.create_dataset(f'{video}', data=data, dtype=np.uint8)
                
def extract_frames_celeb(video_path, group, dtype, crop_ratio):
    # save_path = f'/root/result/h5_test/celeb/{dtype}'
    save_path = f'/root/celebdatas/preprocess/{dtype}/crop_jpg' #######
    os.makedirs(save_path, exist_ok=True)
    
    bbox_path = f'/root/celebdatas/preprocess/{dtype}/bbox'
    video_list = sorted(os.listdir(video_path)) # video_list: id0_0000, id0_0001, ...
    print(video_path)
    print('num_videos: ', len(video_list))
    for j, video in enumerate(video_list):
        frame_list = sorted(os.listdir(os.path.join(video_path, video))) # frame_list: 0001.png, 0002.png, ...
        bbox_df = pd.read_json(os.path.join(bbox_path, video+'.json'))
        
        data = []
        for i, frame_id in enumerate(frame_list):
            frame_path = os.path.join(video_path, video, frame_id)
            # find bbox
            frame_id = frame_id.split('.png')[0]
            try:
                bbox = bbox_df.loc["bbox", int(f'{frame_id}')]
            except:
                bbox = bbox_df.loc["bbox", f'{video}']
            try:
                bbox_size = bbox[3] - bbox[1] if bbox[2] - bbox[0] < bbox[3] - bbox[1] else bbox[2] - bbox[0]
                bbox = [((bbox[0]+bbox[2])-(bbox_size)*crop_ratio)/2,
                        ((bbox[1]+bbox[3])-(bbox_size)*crop_ratio)/2,
                        ((bbox[0]+bbox[2])+(bbox_size)*crop_ratio)/2,
                        ((bbox[1]+bbox[3])+(bbox_size)*crop_ratio)/2]
            except KeyError:
                print(bbox)
            frame = cv2.imread(frame_path)
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # print(i, f'{video}', frame_id[-4:])
            frame = frame.crop(bbox)
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            
            try:
                frame = cv2.resize(frame, (256, 256))
            except:
                print(bbox)
                print(frame)
            # if int(frame_id[-4:]) % 100 == 0:
            #     cv2.imwrite(os.path.join(save_path, f'{video}_{frame_id}.png'), frame)
            tmp_save_path = os.path.join(save_path, video)
            os.makedirs(tmp_save_path, exist_ok=True)
            cv2.imwrite(os.path.join(tmp_save_path, f'{frame_id}.jpg'), frame)
            #######################################################################
            data.append(frame)     
        data = np.stack(data, axis=0)
        # group.create_dataset(f'{video}', data=data, dtype=np.uint8)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', 
                        type=str, 
                        default='Original', 
                        choices = ['Original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures',
                                   'Celeb-real', 'Celeb-synthesis', 'YouTube-real'],
                        help='dataset type')
    parser.add_argument('--crop_ratio',
                        type=float,
                        default=1.7,
                        help='crop ratio')
    args = parser.parse_args()
    
    # get video png path
    if args.dtype == 'Original':
        video_path = '/root/nasdatas/original_sequences/raw/png'
    elif args.dtype == 'Deepfakes' or args.dtype == 'Face2Face' or args.dtype == 'FaceSwap' or args.dtype == 'NeuralTextures':
        video_path = '/root/nasdatas/manipulated_sequences/{}/raw/png'.format(args.dtype)
    else:
        video_path = f'/root/celebdatas/preprocess/{args.dtype}/png'

    # make h5 file
    output_path =  f'{args.dtype}.h5'
    db = h5py.File(output_path, 'w')
    # extract frames
    if args.dtype == 'Original' or args.dtype == 'Deepfakes' or args.dtype == 'Face2Face' or args.dtype == 'FaceSwap' or args.dtype == 'NeuralTextures':
        extract_frames_ff(video_path, db, args.dtype, args.crop_ratio)
    else:
        extract_frames_celeb(video_path, db, args.dtype, args.crop_ratio)
