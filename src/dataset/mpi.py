import os
import copy
import json

import matplotlib.pyplot as plt
import torch.utils.data

from src.dataset.augment import *
from src.utils import *


class MPI(torch.utils.data.Dataset):
    def __init__(self, img_dir, anno_path, is_train):

        self.img_dir = img_dir
        self.is_train = is_train
        f = open(anno_path, 'r')
        self.anno = json.load(f)
        self.train_index, self.valid_index = [], []
        for i, v in enumerate(self.anno):
            if v['validation'] == 1.0:
                self.valid_index.append(i)
            else:
                self.train_index.append(i)

    def __getitem__(self, idx):
        if self.is_train:
            meta = copy.deepcopy(self.anno[self.train_index[idx]])
        else:
            meta = copy.deepcopy(self.anno[self.valid_index[idx]])
        meta['objpos'] = np.array(meta['objpos'])
        meta['joint_self'] = np.array(meta['joint_self'])
        if meta['numOtherPeople'] != 0.0:
            meta['objpos_other'] = np.array(meta['objpos_other'])
            meta['joint_others'] = np.array(meta['joint_others'])

        mpi2ours(meta)

        # Read image
        img = plt.imread(os.path.join(self.img_dir, meta['img_paths']))

        # Image augmentation
        if self.is_train:
            img_scale = scale(img, meta, self.is_train)
            img_rotate = rotate(img_scale, meta)
            img_croppad = croppad(img_rotate, meta, self.is_train)
            label_crop_joint(meta)
            img_dst = flip(img_croppad, meta)
        else:
            img_scale = scale(img, meta, self.is_train)
            img_dst = croppad(img_scale, meta, self.is_train)
            label_crop_joint(meta)

        # visualize(img_dst, meta)
        img_dst = img_dst / 255

        # Normalize
        img_dst = (img_dst - np.array([0.485, 0.456, 0.406])) \
                  / np.array([0.229, 0.224, 0.225])
        img_dst = torch.from_numpy(img_dst.transpose((2, 0, 1)).astype(np.float32))

        # Make label
        label = np.zeros((12, 12, 1311))

        # The first one represents the existence of that joint and
        # the second one represents which person the joint belongs to
        delta = np.zeros((20, 20, 16, 2))
        mask = np.zeros((12, 12, 1311))
        mask[:, :, 0:96:6] = 1
        # Joint's label and mask
        head_length = np.linalg.norm(meta['joint_self'][1, :2] -
                                      meta['joint_self'][2, :2])
        for i in range(16):
            if meta['joint_self'][i, -1] != 2.0:
                grid = (meta['joint_self'][i, :2] / cfg.CELL_SIZE).astype(int)
                x, y = grid[0], grid[1]
                delta[y+4, x+4, i, :] = np.array([1, 1])
                offset = i * 6
                label[y, x, offset] = 1
                label[y, x, offset+2 : offset+4] = meta['joint_self'][i, :2]
                if i == 0:  # person instance
                    label[y, x, offset+4 : offset+6] = 2 * head_length
                else:
                    label[y, x, offset+4 : offset+6] = 0.5 * head_length
                mask[y, x, offset + 1: offset + 6] = 1

        n = int(meta['numOtherPeople'])
        for i in range(n):
            head_length = np.linalg.norm(meta['joint_others'][i, 1, :2] -
                                  meta['joint_others'][i, 2, :2])
            for j in range(16):
                if meta['joint_others'][i, j, -1] != 2.0:
                    grid = (meta['joint_others'][i, j, :2] / cfg.CELL_SIZE).astype(int)
                    x, y = grid[0], grid[1]
                    delta[y+4, x+4, j, :] = np.array([1, i+2])
                    offset = j * 6
                    label[y, x, offset] = 1
                    label[y, x, offset + 2:offset + 4] = meta['joint_others'][i, j, :2]
                    if j == 0:  # person instance
                        label[y, x, offset + 4:offset + 6] = 2 * head_length
                    else:
                        label[y, x, offset + 4:offset + 6] = 0.5 * head_length
                    mask[y, x, offset + 1:offset + 6] = 1

        # Limb's label and mask
        for i in range(12):
            for j in range(12):
                start, end = delta[i+4, j+4, limbs_start, :], delta[i:i+9, j:j+9, limbs_end, :]
                mask[i, j, 96:] = np.reshape(np.maximum(start[:, 0], end[:, :, :, 0]), -1)
                condition = np.logical_and(start[:, 0] * end[:, :, :, 0] == 1, start[:, 1] == end[:, :, :, 1])
                label[i, j, 96:] = np.reshape(np.where(condition, 1, 0), -1)

        label = torch.from_numpy(label.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        return img_dst, label, mask

    def __len__(self):
        return len(self.train_index) if self.is_train else len(self.valid_index)


if __name__ == '__main__':
    dset = MPI('../../data/images', '../../data/mpi_annotations.json', True)
    for i, (img, label, mask) in enumerate(dset):
        print(label)

