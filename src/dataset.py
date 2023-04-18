import torch
import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class VideoFramesDataset(data.Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode # train, val, unlabeled
        self.transform = transform

        self.video_names = os.listdir(os.path.join(self.root, self.mode))
        # remove .DS_Store
        if '.DS_Store' in self.video_names:
            self.video_names.remove('.DS_Store')


        self.video_paths = [os.path.join(self.root, self.mode, video_name) for video_name in self.video_names]


        self.video_frame_paths = []
        for video_path in self.video_paths:
            # frame_paths: list of frame paths, image_0.png, image_1.png, ...
            # exclude mask.npy
            frame_paths = [os.path.join(video_path, frame_name) for frame_name in os.listdir(video_path) if frame_name != 'mask.npy']
            self.video_frame_paths.append(frame_paths)

        # sort frame paths by number in file name
        for frame_paths in self.video_frame_paths:
            frame_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))

            
        self.video_frame_masks = []
        if self.mode != 'unlabeled':
            # load mask, mask.npy contains mask for each frame
            for video_path in self.video_paths:
                mask_path = os.path.join(video_path, 'mask.npy')
                mask = np.load(mask_path)
                self.video_frame_masks.append(mask)


    def __getitem__(self, index):
        video_frames = []
        for frame_path in self.video_frame_paths[index]:
            frame = Image.open(frame_path)
            if self.transform is not None:
                frame = self.transform(frame)
            video_frames.append(frame)
        video_frames = torch.stack(video_frames, dim=0)

        video_mask = self.video_frame_masks[index]

        return video_frames, video_mask

    def __len__(self):
        return len(self.video_names)
    
def show_img_and_mask(img, mask):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(mask)
    plt.show()
    
if __name__ == '__main__':
    # test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = VideoFramesDataset(root='./data/', mode='train', transform=transform)
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

    




