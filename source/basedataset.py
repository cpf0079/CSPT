from torch.utils import data
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms


class MyDataset(data.Dataset):
    def __init__(self, txt_dir, root, num_types, num_frames, crop_size):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(txt_dir, 'r')
        imgs = list()
        self.num_types = num_types

        for line in fh:
            line = line.rstrip()
            words = line.split()
            vid_name = words[0]
            imgs.append(vid_name)
        self.num_frames = num_frames
        self.imgs = imgs
        self.root = root
        self.crop_size = crop_size
        # self.transform = transform
        # self.target_transform = target_transform
        # self.height = 540
        # self.width = 960
        # self.resized_height = 224
        # self.resized_width = 224
        #
        # self.transform = transforms.Compose([
        #     transforms.Resize((self.resized_height, self.resized_width)),
        #     # transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225]),
        #  ])

    def __getitem__(self, index):
        vid_name = self.imgs[index]
        IMG = np.empty((self.num_types, self.num_frames, self.crop_size[0], self.crop_size[1], 3), np.dtype('float32'))
        for i in range(self.num_types):
            for j in range(self.num_frames):
                img = Image.open(self.root + vid_name + '_' + str(i+1) + '_' + str(j+1) + '.png').convert('RGB')
                img = img.resize(self.crop_size, Image.BICUBIC)
                # img = self.transform(img)
                img = np.asarray(img, np.float32)
                img -= np.array((90.0, 98.0, 102.0), dtype=np.float32)
                IMG[i, j] = img
        # IMG = self.normalize(IMG)
        IMG = self.to_tensor(IMG)

        return torch.from_numpy(IMG)

    def __len__(self):
        return len(self.imgs)

    def to_tensor(self, buffer):
        return buffer.transpose((0, 4, 1, 2, 3))
