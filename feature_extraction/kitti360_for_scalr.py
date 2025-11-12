import os
import copy
import yaml
import numpy as np
from PIL import Image
from glob import glob
from pc_dataset import PCDataset

# For normalizing intensities
MEAN_INT = 0.294584
STD_INT = 0.147190

SEQUENCES = ['00', '02', '03', '04', '05', '06', '07', '08', '09', '10'] 


class KITTI360pc(PCDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # For normalizing intensities
        self.mean_int = MEAN_INT
        self.std_int = STD_INT

        # Find all files
        self.im_idx = []
        self.file_path  = []
        for seq_id in SEQUENCES:
            self.im_idx.extend(glob(os.path.join(
                self.rootdir, f'data_3d_raw/2013_05_28_drive_00{seq_id}_sync/velodyne_points/data/*.bin')))
        self.im_idx = np.sort(self.im_idx)
        #assert len(self.im_idx) == 19130

    def __len__(self):
        return len(self.im_idx)

    def load_pc(self, index):
        pc = np.fromfile(self.im_idx[index], dtype=np.float32)
        return pc.reshape((-1, 4)), None, self.im_idx[index]
    