import os
import numpy as np
from PIL import Image
import torch 
from torch.utils.data import Dataset
from torchvision import transforms


class FrameDataset(Dataset):
    def __init__(self, root_dir, labeled=True, transform=None):
        """
        Args:
            root_dir (string): Dataset folder path
            labeled (bool, optional): If the input data is labeled. Defaults to True.
            transform (torchvision.transforms, optional): Data transforms. Defaults to None.
        """
        self.root_dir = root_dir
        self.labeled = labeled
        self.transform = transform
        self.video_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
    def __len__(self):
        return len(self.video_dirs)
    
    def __getitem__(self, idx):
        video_dir = os.path.join(self.root_dir, self.video_dirs[idx])
        frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.png')])
        
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(video_dir, frame_file)
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
            
        frames = torch.stack(frames)
        
        if self.labeled:
            mask_path = os.path.join(video_dir, 'mask.npy')
            mask = np.load(mask_path)
            mask = torch.from_numpy(mask).long()
            return frames, mask
        else:
            return frames, None


def dataset_test():
    train = "/scratch/jp4906/VideoMask/train"
    unlabeled = "/scratch/jp4906/VideoMask/unlabeled"
    val = "/scratch/jp4906/VideoMask/val"
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = FrameDataset(root_dir=train, labeled=True, transform=transform)
    print("Train dataset has {} length".format(len(train_dataset)))
    print("")
    video_frames, video_mask = train_dataset[0]
    print('video_frames for train size: {}'.format(video_frames.size()))
    print("")
    print('video_mask for train size: {}'.format(video_mask.shape))
    print("")
    print("====================")

    unlabeled_dataset = FrameDataset(root_dir=unlabeled, labeled=False, transform=transform)
    print("Unlabeled dataset has {} length".format(len(unlabeled_dataset)))
    print("")
    video_frames, video_mask = unlabeled_dataset[0]
    print('video_frames for unlabeled size: {}'.format(video_frames.size()))
    print("")
    print('video_mask for unlabeled should be NoneType: {}'.format(video_mask))
    print("")
    print("====================")
