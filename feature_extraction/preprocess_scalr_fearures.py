import os
import argparse
import yaml

import numpy as np
import torch
from tqdm import tqdm

from waffleiron import Segmenter
from pc_dataset import Collate
from kitti360_for_scalr import KITTI360pc


def point_to_token_index(pcd, fov=[3, -25], grid_size=[8,64]):
    "Projection mask of the points onto a range grid"

    yaw = -np.arctan2(pcd[:, 1],pcd[:, 0])
    pitch = np.arcsin(pcd[:, 2] / np.linalg.norm(pcd[:, :3], axis=1))

    fov_up = fov[0] / 180.0 * np.pi
    fov_down = fov[1] / 180.0 * np.pi
    fov_range = abs(fov_down) + abs(fov_up)

    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov_range  # in [0.0, 1.0]

    proj_x *= grid_size[1]  # in [0.0, W]
    proj_y *= grid_size[0]  # in [0.0, H]
    
    proj_x = np.maximum(0, np.minimum(grid_size[1] - 1, np.floor(proj_x))).astype(np.int32)  # in [0,W-1]
    proj_y = np.maximum(0, np.minimum(grid_size[0] - 1, np.floor(proj_y))).astype(np.int32)  # in [0,H-1]
    return proj_x , proj_y


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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the dataset directory containing images.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the model file.")
    parser.add_argument("--grid-size", type=int, nargs=2, required=False, default=[8, 64],
                        help="Grid size as two integers [height, width] for feature extraction.")
    args = parser.parse_args()

    scalr_features_dir = os.path.join(args.dataset_path, "scalr_features")

    #load ScaLR model
    with open(os.path.join(args.model_path,"config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    data_cfg = {
        "rootdir": os.path.join(args.dataset_path, "KITTI-360"),
        "input_feat": config["point_backbone"]["input_features"],
        "voxel_size": config["point_backbone"]["voxel_size"],
        "num_neighbors": config["point_backbone"]["num_neighbors"],
        "dim_proj": config["point_backbone"]["dim_proj"],
        "grids_shape": config["point_backbone"]["grid_shape"],
        "fov_xyz": config["point_backbone"]["fov"],
    }
    dataset = KITTI360pc(phase="val", **data_cfg)

    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=Collate(),
    )


    model = Segmenter(
        input_channels=config["point_backbone"]["size_input"],
        feat_channels=config["point_backbone"]["nb_channels"],
        depth=config["point_backbone"]["depth"],
        grid_shape=config["point_backbone"]["grid_shape"],
        nb_class=config["point_backbone"]["nb_class"],
        layer_norm=config["point_backbone"]["layernorm"],
    )

    # Load pretrained model
    ckpt_path = os.path.join(args.model_path, "ckpt_last.pth")
    if os.path.isfile(ckpt_path):
        print(f"Loading checkpoint {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if ckpt.get("model_points") is not None:
            ckpt = ckpt["model_points"]
        else:
            ckpt = ckpt["model_point"]
        new_ckpt = {}
        for k in ckpt.keys():
            if k.startswith("module"):
                new_ckpt[k[len("module.") :]] = ckpt[k]
            else:
                new_ckpt[k] = ckpt[k]
        model.load_state_dict(new_ckpt)
    else:
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")
    model.classif = torch.nn.Identity()  # Replace classification layer with identity for feature extraction
    model.to(device)
    model.eval()

    def get_network_input(batch):
        feat = batch["feat"].cuda(non_blocking=True)
        batch["upsample"] = [
            up.cuda(non_blocking=True) for up in batch["upsample"]
        ]
        cell_ind = batch["cell_ind"].cuda(non_blocking=True)
        occupied_cell = batch["occupied_cells"].cuda(non_blocking=True)
        neighbors_emb = batch["neighbors_emb"].cuda(non_blocking=True)
        return (feat, cell_ind, occupied_cell, neighbors_emb), batch["upsample"][0], batch["filename"][0]

    ld = tqdm(iter(loader))

    os.makedirs(scalr_features_dir, exist_ok=True)

    for batch in ld:
        filename = batch['filename'][0]
        pc_orig = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)[:, :3]  # Load point cloud
        
        sequence = filename.split('/')[-4].split('_')[-2]  # Extract the correct sequence
        name = filename.split('/')[-1].replace('.bin', '')
        save_name = os.path.join(scalr_features_dir, sequence, name)

        if os.path.isdir(save_name) and len(os.listdir(save_name)) == 2:
            continue
        os.makedirs(save_name, exist_ok=True)

        # Extract features
        net_inputs, upsample, pc_filename = get_network_input(batch)
        with torch.autocast("cuda", enabled=True):
            with torch.inference_mode():
                feat = model(*net_inputs)
                feat = feat[0, :, upsample].T
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        
        feature_map = create_grid(pc_orig, feat.cpu().numpy(), grid_size=args.grid_size)

        np.savez(os.path.join(save_name,"feature_points"), feats=feat.cpu())
        np.savez(os.path.join(save_name,f"feature_grid_{args.grid_size[0]}x{args.grid_size[1]}.npz"), feats=feature_map)

        print(f'Features are extracted for {filename}')
        


    