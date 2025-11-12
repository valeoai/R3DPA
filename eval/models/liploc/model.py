import timm
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch import nn
import torch
from torchvision.transforms import Resize, Normalize, ToTensor,  Compose
from ..range_utils import pcd2range

def createRangeImage(range, crop=True):
    depth = range[0]
    image = Image.fromarray(np.uint8(depth * 5.1) , 'L')
    image = np.stack((image,)*3, axis=-1)

    if crop: # overlap with the camera image
        clipStart = image.shape[1] // 3      
        image = image[0:32, clipStart:clipStart*2, :]
    return image

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name, pretrained, trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_lidar = ImageEncoder(model_name=cfg.trained_image_model_name, pretrained=cfg.pretrained, trainable=cfg.trainable)
        self.transform = Compose([
        ToTensor(),
        Resize((224, 224)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])

    def preprocess(self, batch):
        batch = [createRangeImage(pcd ,False) for pcd in batch]
        for i in range(len(batch)):
            batch[i] = self.transform(batch[i])
        batch = torch.stack(batch, dim=0).float()
        return batch

    def forward(self, batch):
        batch = self.preprocess(batch).to(self.device)
        return self.encoder_lidar(batch).cpu()