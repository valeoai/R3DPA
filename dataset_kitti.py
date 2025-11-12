import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset

KITTI360_TRAIN_SET = ['03','04', '05', '06', '07', '09', '10']  # LiDARGEN/R2DM/R2FLOW
KITTI360_VAL_SET = ['00','02']


class KITTI360(Dataset):
    def __init__(self, data_root, split, dataset_config, grid_size=(8,64), return_pcd=False):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.data = []

        self.grid_size = grid_size
        self.img_size = dataset_config.size
        self.fov = dataset_config.fov
        self.depth_range = dataset_config.depth_range
        self.depth_scale = dataset_config.depth_scale
        self.log_scale = dataset_config.log_scale
        self.return_pcd = return_pcd

        if self.log_scale:
            self.depth_thresh = (np.log2(1./255. + 1) / self.depth_scale) * 2. - 1 + 1e-6
        else:
            self.depth_thresh = (1./255. / self.depth_scale) * 2. - 1 + 1e-6

        self.prepare_data()

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load_lidar_sweep(path):
        scan = np.fromfile(path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]  # get xyz
        intensity = scan[:, 3]  # get intensity
        return points, intensity
    
    def load_scalr_features(self, filename):
        sequence = filename.split('/')[-4].split('_')[-2]  
        name = filename.split('/')[-1].replace('.bin', '')
        feature_dir = os.path.join(self.data_root,"scalr_features", sequence, name)
        grid = np.load(os.path.join(feature_dir,f"feature_grid_{self.grid_size[0]}x{self.grid_size[1]}.npz"))['feats']
        return torch.from_numpy(grid)

    def process_scan(self, range_img):
        range_img = np.where(range_img < 0, 0, range_img)

        if self.log_scale:
            # log scale
            range_img = np.log2(range_img + 0.0001 + 1)

        range_img = range_img / self.depth_scale
        range_img = range_img * 2. - 1.

        range_img = np.clip(range_img, -1, 1)
        range_img = np.expand_dims(range_img, axis=0)

        # mask
        range_mask = np.ones_like(range_img)
        range_mask[range_img < self.depth_thresh] = -1

        return torch.from_numpy(range_img), torch.from_numpy(range_mask)

    def prepare_data(self):
        # read data paths
        self.data = []
        if self.split ==  'all':
            seq_list = KITTI360_TRAIN_SET + KITTI360_VAL_SET
        else:
            seq_list = eval('KITTI360_%s_SET' % self.split.upper())
        for seq_id in seq_list:
            self.data.extend(glob.glob(os.path.join(
                self.data_root, "KITTI-360",f'data_3d_raw/2013_05_28_drive_00{seq_id}_sync/velodyne_points/data/*.bin')))

    def __getitem__(self, idx):
        example = dict()
        data_path = self.data[idx]

        sweep, intensity = self.load_lidar_sweep(data_path)
        example['features'] = self.load_scalr_features(data_path)
       
        proj_range, proj_intensity = pcd2range(sweep, self.img_size, self.fov, self.depth_range, labels=intensity)
        proj_range, proj_mask = self.process_scan(proj_range)
        example['image'], example['mask'] = proj_range, proj_mask
        example['intensity'] = torch.from_numpy(proj_intensity).unsqueeze(0) 
        
        if self.return_pcd:
            reproj_sweep, _, _ = range2pcd(proj_range[0].numpy() * .5 + .5, self.fov, self.depth_range, self.depth_scale, self.log_scale)
            example['raw'] = sweep
            example['reproj'] = reproj_sweep.astype(np.float32)

        return example


def pcd2range(pcd, size, fov, depth_range, remission=None, labels=None, **kwargs):
    # laser parameters
    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
    fov_range = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth (distance) of all points
    depth = np.linalg.norm(pcd, 2, axis=1)

    # mask points out of range
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    depth, pcd = depth[mask], pcd[mask]

    # get scan components
    scan_x, scan_y, scan_z = pcd[:, 0], pcd[:, 1], pcd[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov_range  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= size[1]  # in [0.0, W]
    proj_y *= size[0]  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.maximum(0, np.minimum(size[1] - 1, np.floor(proj_x))).astype(np.int32)  # in [0,W-1]
    proj_y = np.maximum(0, np.minimum(size[0] - 1, np.floor(proj_y))).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    proj_x, proj_y = proj_x[order], proj_y[order]

    # project depth
    depth = depth[order]
    proj_range = np.full(size, -1, dtype=np.float32)
    proj_range[proj_y, proj_x] = depth

    # project point feature
    if remission is not None:
        remission = remission[mask][order]
        proj_feature = np.full(size, -1, dtype=np.float32)
        proj_feature[proj_y, proj_x] = remission
    elif labels is not None:
        labels = labels[mask][order]
        proj_feature = np.full(size, 0, dtype=np.float32)
        proj_feature[proj_y, proj_x] = labels
    else:
        proj_feature = None

    return proj_range, proj_feature

def range2pcd(range_img, fov, depth_range, depth_scale, log_scale=False, label=None, color=None, **kwargs):
    # laser parameters
    size = range_img.shape
    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
    fov_range = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # inverse transform from depth
    depth = (range_img * depth_scale).flatten()
    if log_scale:
        depth = np.exp2(depth) - 1

    scan_x, scan_y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    scan_x = scan_x.astype(np.float64) / size[1]
    scan_y = scan_y.astype(np.float64) / size[0]

    yaw = (np.pi * (scan_x * 2 - 1)).flatten()
    pitch = ((1.0 - scan_y) * fov_range - abs(fov_down)).flatten()

    pcd = np.zeros((len(yaw), 3))
    pcd[:, 0] = np.cos(yaw) * np.cos(pitch) * depth
    pcd[:, 1] = -np.sin(yaw) * np.cos(pitch) * depth
    pcd[:, 2] = np.sin(pitch) * depth

    # mask out invalid points
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    pcd = pcd[mask, :]

    # label
    if label is not None:
        label = label.flatten()[mask]

    # default point color
    if color is not None:
        color = color.reshape(-1, 3)[mask, :]
    else:
        color = np.ones((pcd.shape[0], 3)) * [0.7, 0.7, 1]

    return pcd, label, color

def range2xyz(range_img, fov, depth_range, depth_scale, log_scale=True, **kwargs):
    # laser parameters
    size = range_img.shape
    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
    fov_range = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # inverse transform from depth
    if log_scale:
        print("in_logscale")
        depth = (np.exp2(range_img * depth_scale) - 1)
    else:
        depth = range_img * depth_scale

    scan_x, scan_y = np.meshgrid(np.arange(size[-1]), np.arange(size[-2]))
    scan_x = scan_x[None,None].astype(np.float64) / size[-1]
    scan_y = scan_y[None,None].astype(np.float64) / size[-2]

    yaw = np.pi * (scan_x * 2 - 1)
    pitch = (1.0 - scan_y) * fov_range - abs(fov_down)

    xyz = -np.ones((3, *size))
    xyz[0] = np.cos(yaw) * np.cos(pitch) * depth
    xyz[1] = -np.sin(yaw) * np.cos(pitch) * depth
    xyz[2] = np.sin(pitch) * depth

    # mask out invalid points
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    xyz[:, ~mask] = -1

    return xyz, depth