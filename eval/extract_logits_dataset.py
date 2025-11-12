import os
import torch
import numpy as np
import math
from omegaconf import OmegaConf
import argparse
from tqdm import tqdm

from .metric_utils import compute_batch_logits, volume_sum_update, voxelize_pcd
from . import build_model, VOXEL_SIZE, MODEL2BATCHSIZE, DATASET_CONFIG, AGG_TYPE, NUM_SECTORS, \
    TYPE2DATASET, DATA_CONFIG
from dataset_kitti import KITTI360, range2pcd

def preprocess_range(x,i, depth_scale):
    depth = (x*.5+.5)*depth_scale
    mask = depth > 0
    range_img = np.where(mask, depth, -1)
    return np.concatenate([range_img,i],axis=0)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = ['minkowskinet', 'spvcnn','liploc', 'rangevit_w_i', 'rangevit_wo_i']

    parser = argparse.ArgumentParser(description="Extract logits from dataset")
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the logits statistics')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')

    args = parser.parse_args()  
    
    config = OmegaConf.load("configs/kitti.yaml")
    
    all_dataset = KITTI360(data_root=args.dataset_path, split='all', dataset_config=config)
    val_dataset = KITTI360(data_root=args.dataset_path, split='val', dataset_config=config)

    dict_logits_statistics = {"val":{},"all":{}}
    for m in models:

        is_voxel = m in ['minkowskinet', 'spvcnn']

        print(f"loading model {m}...")
        model = build_model('kitti',m, device)
        print(f"model {m} loaded.")

        print(f"Computing logits for model {m}...")
        for split in ['all','val']:
            dataloader = torch.utils.data.DataLoader(
                eval(f"{split}_dataset"),
                batch_size = MODEL2BATCHSIZE[m],
                num_workers = 4,
            )
            all_logits_list = []
            if is_voxel:
                for batch in tqdm(dataloader):
                    batch = [range2pcd(x.squeeze().numpy()* .5 + .5,**DATASET_CONFIG['kitti'])[0] for x in batch['image']]
                    logits = compute_batch_logits(batch, model, is_voxel=is_voxel, dataset_config=config)
                    all_logits_list.extend(logits)
                all_logits = np.stack(all_logits_list)
            else:
                for batch in tqdm(dataloader):
                    batch = [preprocess_range(x, i, config.depth_scale) for x,i in zip(batch['image'], batch['intensity'].numpy())]
                    print(batch[0].shape)
                    logits = compute_batch_logits(batch, model, is_voxel=is_voxel, dataset_config=config)
                    all_logits_list.append(logits)
                all_logits = np.vstack(all_logits_list)

            mu = np.mean(all_logits, axis=0)
            sigma = np.cov(all_logits, rowvar=False)
            dict_logits_statistics[split][m] = {"mu": mu, "sigma": sigma}
        print(f"Logits for model {m} computed.")

    x_range, y_range = DATA_CONFIG['64']['x'], DATA_CONFIG['64']['y']
    vol_shape = (math.ceil((x_range[1] - x_range[0]) / VOXEL_SIZE), math.ceil((y_range[1] - y_range[0]) / VOXEL_SIZE))
    min_bound = (math.ceil((x_range[0]) / VOXEL_SIZE), math.ceil((y_range[0]) / VOXEL_SIZE))
    
    #Compute validation set voxelized point cloud bev
    print("Computing validation set voxelized point cloud bev...")

    valset_pcd = KITTI360(data_root=args.dataset_path, split='val', dataset_config=config, return_pcd=True)

    data, fp = [], []
    for i, pcd in enumerate(valset_pcd):
        pcd_voxel = voxelize_pcd(pcd['reproj'], x_range, y_range, VOXEL_SIZE, min_bound, vol_shape)
        data.append(pcd_voxel)
        # Cut the first part of the file path to get the filename
        filename = valset_pcd.data[i].split('/')[-4:]
        fp.append(filename)

    perm = np.random.permutation(len(data))
    data = [data[i] for i in perm]
    fp = [fp[i] for i in perm]

    dict_logits_statistics['val']['valset_voxelized_bev'] = {}
    dict_logits_statistics['val']['valset_voxelized_bev']['points'] = data
    dict_logits_statistics['val']['valset_voxelized_bev']['file_paths'] = fp

    print("Validation set voxelized point cloud bev computed.")

    ## Compute dataset bev histogram ##
    print("Computing dataset bev histogram...")

    for split in ["all",'val']:
        volume_sum = np.zeros(vol_shape, np.float32)

        dataset_pcd = KITTI360(data_root=args.dataset_path, split=split, dataset_config=config, return_pcd=True)
        for pcd in dataset_pcd:
            volume_sum = volume_sum_update(volume_sum, pcd['reproj'], x_range, y_range, VOXEL_SIZE, min_bound)

        dict_logits_statistics[split]['dataset_bev_hist'] = volume_sum

        print("Dataset bev histogram computed.")

    # Save the statistics
    path = os.path.join(args.save_path, 'range_update_stats.npy')
    np.save(path, dict_logits_statistics, allow_pickle=True)    
    print(f"Logits statistics saved to {path}.")

        