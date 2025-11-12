# Copyright 2024 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np


class Transformation:
    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, pcloud, labels):
        if labels is None:
            return (
                (pcloud, None) if self.inplace else (np.array(pcloud, copy=True), None)
            )

        out = (
            (pcloud, labels)
            if self.inplace
            else (np.array(pcloud, copy=True), np.array(labels, copy=True))
        )
        return out


class Crop(Transformation):
    def __init__(self, dims=(0, 1, 2), fov=((-5, -5, -5), (5, 5, 5)), eps=1e-4):
        super().__init__(inplace=True)
        self.dims = dims
        self.fov = fov
        self.eps = eps
        assert len(fov[0]) == len(fov[1]), "Min and Max FOV must have the same length."
        for i, (min, max) in enumerate(zip(*fov)):
            assert (
                min < max
            ), f"Field of view: min ({min}) < max ({max}) is expected on dimension {i}."

    def __call__(self, pcloud, labels, return_mask=False):
        pc, labels = super().__call__(pcloud, labels)

        where = None
        for i, d in enumerate(self.dims):
            temp = (pc[:, i] > self.fov[0][i] + self.eps) & (
                pc[:, i] < self.fov[1][i] - self.eps
            )
            where = temp if where is None else where & temp

        if return_mask:
            return pc[where], None if labels is None else labels[where],where
        else:
            return pc[where], None if labels is None else labels[where]


class Voxelize(Transformation):
    def __init__(self, dims=(0, 1, 2), voxel_size=0.1, random=False):
        super().__init__(inplace=True)
        self.dims = dims
        self.voxel_size = voxel_size
        self.random = random
        assert voxel_size >= 0

    def __call__(self, pcloud, labels):
        pc, labels = super().__call__(pcloud, labels)
        if self.voxel_size <= 0:
            return pc, labels

        if self.random:
            permute = torch.randperm(pc.shape[0])
            pc, labels = pc[permute], None if labels is None else labels[permute]

        pc_shift = pc[:, self.dims] - pc[:, self.dims].min(0, keepdims=True)

        _, ind = np.unique(
            (pc_shift / self.voxel_size).astype("int"), return_index=True, axis=0
        )

        return pc[ind, :], None if labels is None else labels[ind]