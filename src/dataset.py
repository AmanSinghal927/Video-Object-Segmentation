import os
import numpy as np
from PIL import Image
import torch 
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


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
        self.video_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.video_dirs.sort(key=lambda x: int(x.split('/')[-1].split('_')[-1]))
        
    def __len__(self):
        return len(self.video_dirs)
    
    def __getitem__(self, idx):
        video_dir = os.path.join(self.root_dir, self.video_dirs[idx])
        frame_files = [f for f in os.listdir(video_dir) if f.endswith('.png')]
        frame_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))

        # metadata: 
        #   frame_file names in order
        #   video_dir name
        metadata = {
            "frames": frame_files,
            "video": video_dir
        }
        print("{}".format(metadata))
        
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
    train = "./data/train"
    unlabeled = "./data/unlabeled"
    val = "./data/val"
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

    
def show_img_and_mask(img, mask):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(mask)
    plt.show()

if __name__ == "__main__":
    dataset_test()

    # test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = FrameDataset(root_dir='./data/train', labeled=True, transform=transform)
    print('dataset size: {}'.format(len(dataset)))

    video_frames, video_mask = dataset[0]
    print('video_frames size: {}'.format(video_frames.size()))
    print('video_mask size: {}'.format(video_mask.shape))

    print (video_frames[0].shape)
    print(video_mask[0].shape)

    for i in range(0, 21):
        img = video_frames[i].numpy().transpose(1, 2, 0)
        mask = video_mask[i]
        show_img_and_mask(img, mask)

