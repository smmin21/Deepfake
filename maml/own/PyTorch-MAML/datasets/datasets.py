import os

import torch
from PIL import Image



DEFAULT_ROOT = './materials'
datasets = {}

def register(name):
  def decorator(cls):
    datasets[name] = cls
    return cls
  return decorator


def make(name, **kwargs):
  if kwargs.get('root_path') is None:
    path_dict = {'meta-celeb-df': '/root/datasets/celeb',
                 'meta-face-forensics': '/root/datasets/ff',
                 'meta-vfhq': '/root/datasets/vfhq',
                 'meta-dfdc': '/root/volume3/dfdc_preprocessed',
                 'meta-dff': '/root/volume3/dff_preprocessed'
                 }
    # kwargs['root_path'] = os.path.join(DEFAULT_ROOT, name.replace('meta-', ''))
    kwargs['root_path'] =  path_dict[name]
  dataset = datasets[name](**kwargs)
  # dataset = datasets[name]()
  return dataset


def collate_fn(batch):
  shot, query, shot_label, query_label = [], [], [], []
  for s, q, sl, ql in batch:
    shot.append(s)
    query.append(q)
    shot_label.append(sl)
    query_label.append(ql)
  
  # frame_keys = torch.stack(frame_keys)    # [n_ep, n_way * (n_shot + n_query)]
  shot = torch.stack(shot)                # [n_ep, n_way * n_shot, C, H, W]
  query = torch.stack(query)              # [n_ep, n_way * n_query, C, H, W]
  shot_label = torch.stack(shot_label)    # [n_ep, n_way * n_shot]
  query_label = torch.stack(query_label)  # [n_ep, n_way * n_query]
  
  return shot, query, shot_label, query_label


def load_video_frames(video_dir, transform):
    frame_keys = sorted(os.listdir(video_dir))
    frames = []
    for frame_key in frame_keys:
        frame = Image.open(os.path.join(video_dir, frame_key))
        frame = transform(frame)
        frames.append(frame)
    return frames