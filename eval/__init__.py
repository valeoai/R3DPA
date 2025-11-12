"""
@Author: Haoxi Ran
@Date: 01/03/2024
@Citation: Towards Realistic Scene Generation with LiDAR Diffusion Models

"""

import os
import sys

import torch
import yaml
import argparse

from .models.rangevit.model import RangeViT as rangevit
from .models.liploc.model import Model as liploc

try:
    from .models.spvcnn.model import Model as spvcnn
    from .models.minkowskinet.model import Model as minkowskinet
except:
    print('To install torchsparse 1.4.0, please refer to https://github.com/mit-han-lab/torchsparse/tree/74099d10a51c71c14318bce63d6421f698b24f24')

# user settings
DEFAULT_ROOT = './pretrained_weights'
MODEL2BATCHSIZE = {'minkowskinet': 50, 'spvcnn': 25, 'rangevit_wo_i': 8, 'rangevit_w_i': 8, 'liploc':16}
OUTPUT_TEMPLATE = 50 * '-' + '\n|' + 16 * ' ' + '{}:{:.4E}' + 17 * ' ' + '|\n' + 50 * '-'

# eval settings (do not modify)
VOXEL_SIZE = 0.05
NUM_SECTORS = 16
AGG_TYPE = 'depth'
TYPE2DATASET = {'32': 'nuscenes', '64': 'kitti'}
DATA_CONFIG = {'64': {'x': [-50, 50], 'y': [-50, 50], 'z': [-3, 1]},
               '32': {'x': [-30, 30], 'y': [-30, 30], 'z': [-3, 6]}}
DATASET_CONFIG = {'kitti': {'size': [64, 1024], 'fov': [3, -25], 'depth_range': [1.0, 56.0], 'depth_scale': 56},
                  'nuscenes': {'size': [32, 1024], 'fov': [10, -30], 'depth_range': [1.0, 45.0]}}

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def build_model(dataset_name, model_name, device='cpu'):
    # config
    model_folder = os.path.join(DEFAULT_ROOT, dataset_name, model_name)

    if not os.path.isdir(model_folder):
        raise Exception('Not Available Pretrained Weights!')

    config = yaml.safe_load(open(os.path.join(model_folder, 'config.yaml'), 'r'))
    config = dict2namespace(config)

    if 'rangevit' in model_name:
        model = rangevit(config)
    else:
        model = eval(model_name)(config)

    # load checkpoint
    ckpt = torch.load(os.path.join(model_folder, 'model.ckpt'), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device)
    model.eval()

    return model
