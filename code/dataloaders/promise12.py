import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import SimpleITK as sitk
import random
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom



class Promise12(Dataset):
    """promise12 Dataset"""

    def __init__(self,base_dir=None,split='train',num=None,transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.split = split

        train_path = self._base_dir+'/train.txt'
        val_path = self._base_dir+'/val.txt'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'val':
            with open(val_path, 'r') as f:
                self.image_list = f.readlines()
        
        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        
        print("total {} samples".format(len(self.image_list)))


    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        label_name = image_name.replace('slice','seg_slice')
        if self.split == "train":
            image_path = self._base_dir + '/train/images/{}'.format(image_name)
            label_path = self._base_dir + '/train/masks/{}'.format(label_name)
        elif self.split == "val":
            image_path = self._base_dir + '/val/images/{}'.format(image_name)
            label_path = self._base_dir + '/val/masks/{}'.format(label_name)
        
        image = Image.open(image_path)
        image = np.array(image)
        label = Image.open(label_path)
        label = np.array(label)
        if label.max() > 1:
            label[label>50] = 255
            label = label / 255
        sample = {"image":image,"label":label}

        if self.split == "train":
            sample = self.transform(sample)
        elif self.split == "val":
            image,label = sample['image'],sample["label"]
            x , y =image.shape
            image = zoom(image,(224/x,224/y),order=0)
            label = zoom(label,(224/x,224/y),order=0)
            image = torch.from_numpy(image).unsqueeze(0)  
            label = torch.from_numpy(label)
            sample = {"image": image, "label": label}

        sample["idx"] = idx
        return sample
    
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        x, y = image.shape
        image = zoom(image, (self.output_size[0]/ x, self.output_size[1]/ y), order=0)
        label = zoom(label, (self.output_size[0]/ x, self.output_size[1]/ y), order=0)

        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image
    
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label