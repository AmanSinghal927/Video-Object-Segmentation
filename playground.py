import os
import numpy as np
from PIL import Image
import torch 
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.root_dir = root_dir
        self.mode = mode 
        self.transform = transform 
        
        self.mode_path = os.path.join(root_dir, mode)
        self.vedios = [os.path.join(root_dir, subfolder) for subfolder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subfolder))]