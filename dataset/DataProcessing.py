import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np

class RoadDataset(Dataset):
    def __init__(self, root_images, root_masks, h, w, augment=False):
        super().__init__()
        self.root_images = root_images
        self.root_masks = root_masks
        self.h = h
        self.w = w
        self.augment = augment
        self.images = []
        self.labels = []

        files = sorted(os.listdir(self.root_images))
        sfiles = sorted(os.listdir(self.root_masks))
        for i in range(len(sfiles)):
            img_file = os.path.join(self.root_images, files[i])
            mask_file = os.path.join(self.root_masks, sfiles[i])
            self.images.append(img_file)
            self.labels.append(mask_file)

        self.norm = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.labels[idx])

        if self.augment:
            tf = transforms.Compose([
                transforms.Resize((int(self.h * 1.25), int(self.w * 1.25))),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(16, fill=144),
                transforms.CenterCrop((self.h, self.w)),
                transforms.ToTensor()
            ])

            seed = np.random.randint(42)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            image = tf(image)

            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            mask = tf(mask)
        else:
            tf = transforms.Compose([
                transforms.Resize((self.h, self.w)),
                transforms.ToTensor()
            ])
            image = tf(image)
            mask = tf(mask)

        image = self.norm(image)
        mask[mask > 0] = 1.

        sample = {'image': image, 'mask': mask}
        return sample
