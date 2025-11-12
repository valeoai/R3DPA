from omegaconf import OmegaConf
import os
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('./')
from dataset_kitti import KITTI360, point_to_token_index
from multiprocessing import Pool, cpu_count

def process_file(i):
    filename = dataset.data[i]
    pcd,_ = dataset.load_lidar_sweep(filename)
    create_grids_from_scalr_features(pcd, filename, data_root=data_dir, grid_sizes=grid_sizes)


def create_grid(pcd, feature_points, grid_size):
    h, w = grid_size
    feature_map = np.zeros((h, w, feature_points.shape[1]))
    coord = point_to_token_index(pcd,grid_size=grid_size)
    coord = coord[1] * w + coord[0]
    for i in range(h):
        for j in range(w):
            mask = coord == i * w + j
            feature_map[i, j, :] = feature_points[mask].mean(axis=0) if mask.any() else np.zeros(feature_points.shape[1])
    return feature_map.reshape(-1, feature_points.shape[1])


def create_grids_from_scalr_features(pcd,filename, data_root, grid_sizes):
    sequence = filename.split('/')[-4].split('_')[-2]
    name = filename.split('/')[-1].replace('.bin', '')
    feature_dir = os.path.join("/home/nsereyjo/iveco/nsereyjo/scalr_features", sequence, name)
    feature_points = np.load(os.path.join(feature_dir,"feature_points.npz"))['feats']

    for grid_size in grid_sizes:
        save_name = os.path.join(feature_dir, f"feature_grid_{grid_size[0]}x{grid_size[1]}")
        if os.path.exists(save_name+".npz"):
            # Skip if already computed
            print(f"[SKIP] Grid {grid_size} already exists for {filename}")
            continue
        grid = create_grid(pcd, feature_points, grid_size)
        np.savez(save_name, feats=grid)
    print(f'[DONE] Grids processed for {filename}')

def sanity_check(i):
    filename = dataset.data[i]
    sequence = filename.split('/')[-4].split('_')[-2]
    name = filename.split('/')[-1].replace('.bin', '')
    datasets_dir = dataset.data_root.split('/')[:-1]
    feature_dir = os.path.join('/',*datasets_dir,"scalr_features", sequence, name)
    try:
        np.load(os.path.join(feature_dir,f"feature_grid_{grid_size[0]}x{grid_size[1]}.npz"))['feats']
    except Exception as e:
        print(f"[ERROR] Grid not found for {filename}: {e}")
        print("Recomputing grid")
        os.remove(os.path.join(feature_dir,f"feature_grid_{grid_size[0]}x{grid_size[1]}.npz"))
        pcd = dataset.load_lidar_sweep(filename)
        create_grids_from_scalr_features(pcd, filename, dataset.data_root, [grid_size])
    


if __name__ == "__main__":
    dataset_config = OmegaConf.load("configs/kitti.yaml")
    data_dir = "/home/nsereyjo/workspace/REPA3D/dataset"
    grid_sizes = [(8,128)]
    grid_size = (8, 128)  # Default grid size for sanity check
    dataset = KITTI360(data_dir, split="val", dataset_config=dataset_config)
    print(len(dataset), "files to process")

    with Pool(processes=cpu_count()) as pool:
        # list(tqdm(pool.imap(sanity_check, range(len(dataset))), total=len(dataset)))
        list(tqdm(pool.imap(process_file, range(len(dataset))), total=len(dataset)))
