import torch
import os
import numpy as np
from skimage.io import imread

class BDD10K(torch.utils.data.Dataset):
    classes_map = {
        # very similar to KITTI with fewer classes.
        # unknown (255) is treated as void.
        0: (255, 255),  # void
        1: (0, 1),   # flat
        2: (2, 4),   # construction
        3: (5, 7),   # object
        4: (8, 9),   # vegetation
        5: (10, 10), # sky
        6: (11, 12), # human
        7: (13, 18), # vehicle
    }
    num_classes = 8

    def __init__(self, root, split, transform=None):
        assert split in ('train', 'test')
        split = 'val' if split == 'test' else split
        self.img_root = os.path.join(root, 'bdd100k', 'images', '10k', split)
        self.mask_root = os.path.join(root, 'bdd100k', 'labels', 'sem_seg', 'masks', split)
        self.images = sorted(os.listdir(self.img_root))
        self.masks = sorted(os.listdir(self.mask_root))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def map_classes(self, mask):
        new_mask = np.zeros_like(mask)
        for klass, (mincolor, maxcolor) in self.classes_map.items():
            cmp = (mask >= mincolor) & (mask <= maxcolor)
            new_mask[cmp] = klass
        return new_mask

    def __getitem__(self, i):
        image = imread(os.path.join(self.img_root, self.images[i]))
        mask = self.map_classes(imread(os.path.join(self.mask_root, self.masks[i]), True))
        d = {'image': image, 'mask': mask}
        if self.transform != None:
            d = self.transform(**d)
        return d

class BDD10K_Binary(BDD10K):
    num_classes = 1
    def __getitem__(self, i):
        d = super()[i]
        d['mask'] = d['mask'] >= 6
        return d

if __name__ == '__main__':  # debug
    import matplotlib.pyplot as plt
    ds = BDD10K_Binary('/data/auto', 'train')
    d = ds[0]
    plt.subplot(1, 2, 1)
    plt.imshow(d['image'])
    plt.subplot(1, 2, 2)
    plt.imshow(d['mask'])
    plt.show()