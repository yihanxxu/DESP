import os
import os.path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from .randaugment import RandAugmentMC


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])

def get_celeba(root, train_file_list, test_file_list, label_ratio):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size = (218, 178),
                                padding = (int(218 * 0.125), int(178 * 0.125)),
                                padding_mode = 'reflect'),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    train_labeled_idxs, train_unlabeled_idxs = data_split(train_file_list, label_ratio)
    
    train_labeled_dataset = TCelebA(
        root, train_file_list, train_labeled_idxs, 
        transform = transform_labeled)
    
    train_unlabeled_dataset = TCelebA(
        root,train_file_list, train_unlabeled_idxs, 
        transform = TransformFixMatch(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225]))
    
    test_dataset = CelebA(
        root, test_file_list, transform = transform_test)
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def target_read(path):
    file_list = []
    with open(path) as f:
        img_label_list = f.read().splitlines() 
    for info in img_label_list:
        label_list = info.split(' ')
        file_list.append(label_list)
    return file_list


def data_split(label_list, label_ratio):
    img_labels = target_read(label_list)
    img_labels = np.array(img_labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    idxs = np.array(range(len(img_labels)))    
    np.random.shuffle(idxs)
    train_labeled_size = int(len(img_labels) * label_ratio)
    train_labeled_idxs.extend(idxs[:train_labeled_size])
    train_unlabeled_idxs.extend(idxs[train_labeled_size:])
    return train_labeled_idxs, train_unlabeled_idxs


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size = (218, 178),
                                  padding = (int(218 * 0.125), int(178 * 0.125)),
                                  padding_mode = 'reflect')
                                  ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size = (218, 178),
                                  padding = (int(218 * 0.125), int(178 * 0.125)),
                                  padding_mode = 'reflect'),     
                                  RandAugmentMC(n = 5, m = 30),
                                  ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TCelebA(data.Dataset):
    def __init__(self, root, ann_file, indexs, transform=None, target_transform=None, loader=default_loader):
        images = []
        targets = []  
        for line in open(os.path.join(root, ann_file), 'r'):
            sample = line.split()
            if len(sample) != 41:
                raise(RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
            images.append(sample[0])
            targets.append([int(i) for i in sample[1:]])
        self.images = [os.path.join(root, 'img_align_celeba', img) for img in images]
        self.targets = targets
        if indexs is not None:
            self.images = np.array(self.images)[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
		
    def __getitem__(self, index):
        path = self.images[index]
        sample = self.loader(path)
        target = self.targets[index]
        target = torch.Tensor(target)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.images)


class CelebA(data.Dataset):
    def __init__(self, root, ann_file, transform=None, target_transform=None, loader=default_loader):
        images = []
        targets = []  
        for line in open(os.path.join(root, ann_file), 'r'):
            sample = line.split()
            if len(sample) != 41:
                raise(RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
            images.append(sample[0])
            targets.append([int(i) for i in sample[1:]])
        self.images = [os.path.join(root, 'img_align_celeba', img) for img in images]
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
		
    def __getitem__(self, index):
        path = self.images[index]
        sample = self.loader(path)
        target = self.targets[index]
        target = torch.Tensor(target)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.images)
