import os
import cv2
from PIL import Image
import argparse
import pandas as pd
import numpy as np
import json
from collections import defaultdict

def find_wrong_ff(bbox_df):
    crop_ratio = 1.7
    out_of_bound_frames_num = 0
    out_of_bound_video = {}
    bbox_error_video = {}
    result = {}
    total_frames_num = 0
    for i, video_id in enumerate(bbox_df):
        try:
            first_bbox = bbox_df.loc[0, int(f'{video_id}')]["bbox"]
        except:
            first_bbox = bbox_df.loc[0, f'{video_id}']["bbox"]
        for j, frame_id in enumerate(bbox_df[video_id]):
            if not pd.isna(frame_id):
                total_frames_num += 1
                try:
                    bbox = frame_id['bbox']
                    if (first_bbox[0]+first_bbox[2]/2-first_bbox[3]*crop_ratio/2 <= bbox[0]) and \
                       (first_bbox[1]+first_bbox[3]*(1-crop_ratio)/2 <= bbox[1]) and \
                       (first_bbox[0]+first_bbox[2]/2+first_bbox[3]*crop_ratio/2 >= bbox[0] + bbox[2]) and \
                       (first_bbox[1]+first_bbox[3]*(1+crop_ratio)/2 >= bbox[1] + bbox[3]):
                           pass
                    else:
                        if video_id not in out_of_bound_video:
                            out_of_bound_video[video_id] = {'frame_ids': []}
                        out_of_bound_video[video_id]['frame_ids'].append(j)
                        out_of_bound_video[video_id]['out_of_bound_frames_num_per_video'] = out_of_bound_video[video_id].get('out_of_bound_frames_num_per_video', 0) + 1
                        out_of_bound_video[video_id]['total_frames_per_video'] = len(bbox_df[video_id])
                        out_of_bound_frames_num+=1
                except:
                    if video_id not in bbox_error_video:
                        bbox_error_video[video_id] = {'frame_ids': []}
                    bbox_error_video[video_id]['frame_ids'].append(j)
                    # return
                    
    print(out_of_bound_video)
    print(f'Out of bound Total : {len(out_of_bound_video)} videos')
    print(bbox_error_video)
    print(f'Bbox error Total : {len(bbox_error_video)} videos')
    result['total_frame_num'] = total_frames_num
    result['out_of_bound_videos_num'] = len(out_of_bound_video)
    result['out_of_bound_frames_num'] = out_of_bound_frames_num
    result['out_of_bound_videos'] = out_of_bound_video
    result['bbox_error_videos'] = bbox_error_video
    return result

def find_wrong_celeb(bbox_path, dtype, crop_ratio = 1.7, ver = 'first', is_save = False):
    video_list = sorted(os.listdir(bbox_path))
    
    result = defaultdict(lambda: [[], [], []])
    bbox_sizes = []
    
    for video_id in video_list:
        bbox_df = pd.read_json(os.path.join(bbox_path, video_id))
        video_id = video_id[:-5]
        ########################################################
        # get bbox
        if ver == 'first':
            bbox = get_first_bbox(bbox_df)
        elif ver == 'avg':
            bbox = get_avg_bbox(bbox_df)
        elif ver == 'union':
            bbox = get_union_bbox(bbox_df)
        ########################################################
        try:
            bbox_size = bbox[3] - bbox[1] if bbox[2] - bbox[0] < bbox[3] - bbox[1] else bbox[2] - bbox[0]
            bbox_sizes.append(bbox_size*crop_ratio)
            bbox = [((bbox[0]+bbox[2])-(bbox_size)*crop_ratio)/2,
                    ((bbox[1]+bbox[3])-(bbox_size)*crop_ratio)/2,
                    ((bbox[0]+bbox[2])+(bbox_size)*crop_ratio)/2,
                    ((bbox[1]+bbox[3])+(bbox_size)*crop_ratio)/2]
        except KeyError:
            print("Cannot find bbox!")
        compare_to_all_frames(bbox_path, video_id, bbox, bbox_df, result, dtype, ver, is_save)
    
    # print bbox size frequency distribution
    freq, bins = np.histogram(bbox_sizes, bins=10, range=(0, 1000))
    freq_class = [f'{bins[i]}~{bins[i+1]}' for i in range(len(bins)-1)]
    freq_dist_df = pd.DataFrame({'freq': freq}, index=freq_class)
    print(freq_dist_df)
    return get_json_result(result)
            

def get_first_bbox(bbox_df):
    try:
        first_bbox = bbox_df.loc["bbox", 0]
    except KeyError:
        print("Cannot find bbox!")
    return first_bbox

def get_avg_bbox(bbox_df):
    bbox_df_values = bbox_df.loc['bbox'].values
    bbox_df_values = np.array(bbox_df_values.tolist())
    bbox_avg = np.mean(bbox_df_values, axis=0)
    return bbox_avg

def get_union_bbox(bbox_df):
    bbox_df_values = bbox_df.loc['bbox'].values
    bbox_df_values = np.array(bbox_df_values.tolist())
    bbox_min = np.min(bbox_df_values, axis=0)
    bbox_max = np.max(bbox_df_values, axis=0)
    bbox_union = np.concatenate([bbox_min[:2], bbox_max[2:]])
    return bbox_union
    
def compare_to_all_frames(bbox_path, video_id, pivot, bbox_df, result, dtype, pivot_type, is_save):
    for frame_id, value in bbox_df.iteritems():
        if not pd.isna(frame_id):
            try:
                bbox = value['bbox']
                if (pivot[0] <= bbox[0]) and \
                   (pivot[1] <= bbox[1]) and \
                   (pivot[2] >= bbox[2]) and \
                   (pivot[3] >= bbox[3]):
                       result[video_id][0].append(frame_id)
                else:
                    # save out of bound frame
                    if is_save:
                        out_of_bound_frame_path = os.path.join(bbox_path.replace('bbox', 'png'), video_id, '%04d.png' %frame_id)
                        out_of_bound_frame = cv2.imread(out_of_bound_frame_path)
                        out_of_bound_frame = Image.fromarray(cv2.cvtColor(out_of_bound_frame, cv2.COLOR_BGR2RGB))
                        out_of_bound_frame_cropped = out_of_bound_frame.crop(pivot)
                        out_of_bound_frame_cropped = cv2.cvtColor(np.array(out_of_bound_frame_cropped), cv2.COLOR_RGB2BGR)
                        out_of_bound_frame_resized = cv2.resize(out_of_bound_frame_cropped, (256, 256))
                        cv2.imwrite(f'/root/result/celeb_bbox_test/{dtype}/oob_{pivot_type}/{video_id}_{frame_id}.png', out_of_bound_frame_resized)
                    result[video_id][1].append(frame_id)
            except:
                result[video_id][2].append(frame_id)
                
def get_json_result(result):
    result_json = {}
    result_json['total_frame_num'] = sum([len(result[vid][0]) + len(result[vid][1]) + len(result[vid][2]) for vid in result])
    result_json['out_of_bound_videos_num'] = len([vid for vid in result if len(result[vid][1]) > 0])
    result_json['out_of_bound_frames_num'] = sum([len(result[vid][1]) for vid in result])
    result_json['out_of_bound_videos'] = {vid: result[vid][1] for vid in result if len(result[vid][1]) > 0}
    result_json['bbox_error_videos'] = {vid: result[vid][2] for vid in result if len(result[vid][2]) > 0}
    return result_json
    
                       
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', 
                        type=str, 
                        default='Original', 
                        choices = ['Original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures',
                                   'Celeb-real', 'Celeb-synthesis', 'YouTube-real'],
                        help='dataset type')
    parser.add_argument('--version', 
                        type=str, 
                        default='ff++', 
                        choices = ['ff++', 'first', 'avg', 'union'],
                        help='ff++: faceforensics++ / first: celeb by first / avg: celeb by avg')
    parser.add_argument('--crop_ratio', 
                        type=float, 
                        default=1.7, 
                        help='crop ratio')
    parser.add_argument('--is_save',
                        type=bool,
                        default=False,
                        help='save out of bound frames')
    
    args = parser.parse_args()
    
    if args.dtype == 'Original':
        bbox_path = os.path.join('/root/deepfakedatas', 'Original_bbox.json')
    elif args.dtype == 'Deepfakes':
        bbox_path = os.path.join('/root/deepfakedatas', 'Deepfakes_bbox.json')
    elif args.dtype == 'Face2Face':
        bbox_path = os.path.join('/root/deepfakedatas', 'Face2Face_bbox.json')
    elif args.dtype == 'FaceSwap':
        bbox_path = os.path.join('/root/deepfakedatas', 'FaceSwap_bbox.json')
    elif args.dtype == 'NeuralTextures':
        bbox_path = os.path.join('/root/deepfakedatas', 'NeuralTextures_bbox.json')
    elif args.dtype == 'Celeb-real':
        bbox_path = '/root/celebdatas/preprocess/Celeb-real/bbox'
    elif args.dtype == 'Celeb-synthesis':
        bbox_path = '/root/celebdatas/preprocess/Celeb-synthesis/bbox'
    elif args.dtype == 'YouTube-real':
        bbox_path = '/root/celebdatas/preprocess/YouTube-real/bbox'
    else:
        pass
    
    if args.version == 'ff++':
        bbox_df = pd.read_json(bbox_path)
        result = find_wrong_ff(bbox_df)
    else:
        result = find_wrong_celeb(bbox_path, args.dtype, args.crop_ratio, args.version, args.is_save)
    with open(f'{args.dtype}_{args.version}.json','w') as outfile:
        json.dump(result, outfile, indent=2)