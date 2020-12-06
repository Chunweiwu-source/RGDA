from torchvision import datasets, transforms
import torch
import numpy as np
from torchvision import transforms
import os
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class txt_Dataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
	    fh = open(txt_path, 'r')
	    imgs = []
	    for line in fh:
		    line = line.rstrip()
		    words = line.split()
		    imgs.append((words[0], int(words[1])))
		    self.imgs = imgs 
		    self.transform = transform
		    self.target_transform = target_transform
    def __getitem__(self, index):
	    fn, label = self.imgs[index]
	    img = Image.open(fn).convert('RGB') 
	    if self.transform is not None:
		    img = self.transform(img) 
	    return img, label
    def __len__(self):
	    return len(self.imgs)

class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))

class PlaceCrop(object):

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

def load_training(root_path, dir, batch_size, txt_flag, kwargs):

    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    if txt_flag:
        data = txt_Dataset(txt_path=root_path + dir, transform=transform)
    else:
        data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, txt_flag, kwargs):
    start_center = (256 - 224 - 1) / 2
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         PlaceCrop(224, start_center, start_center),
         transforms.ToTensor()])
    if txt_flag:
        data = txt_Dataset(txt_path=root_path + dir, transform=transform)
    else:
        data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    return test_loader


