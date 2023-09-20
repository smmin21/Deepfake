import torch
import torch.nn as nn
import torchvision.models as mds

import pdb
models = {}

def register(name):
  def decorator(cls):
    models[name] = cls
    return cls
  return decorator


def make(name, **kwargs):
  if name is None:
    return None
  model = models[name](**kwargs)
  if torch.cuda.is_available():
    model.cuda()
  return model


def load(ckpt):
  model = make(ckpt['encoder'], **ckpt['encoder_args'])
  if model is not None:
    model.load_state_dict(ckpt['encoder_state_dict'])
  return model

def pretrained_load(name, enc_args):
  assert name in ['resnet18', 'inception_v3']
  model = make(name, **enc_args)
  corresponding_pretrained_model = {'resnet18': mds.resnet18(weights='ResNet18_Weights.DEFAULT'),
                                    'inception_v3': mds.inception_v3(weights='Inception_V3_Weights.DEFAULT'),}
  pretrained_model = corresponding_pretrained_model[name]
  
  if model is not None:
    model_dict = model.state_dict()
  pretrained_dict = pretrained_model.state_dict()
  
  pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  
  model_dict.update(pretrained_dict)
  model.load_state_dict(model_dict)
  return model
  