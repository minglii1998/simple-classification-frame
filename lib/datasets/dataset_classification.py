from __future__ import absolute_import

# import sys
# sys.path.append('./')

import os
import os.path as osp
# import moxing as mox

import pickle
from tqdm import tqdm
from PIL import Image, ImageFile
import numpy as np
import random
import cv2
import lmdb
import sys
import six

import torch
from torch.utils import data
from torch.utils.data import sampler
from torchvision import transforms

from lib.utils import fileutils

from imgaug import augmenters as iaa
import imageio


ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(data.Dataset):
  def __init__(self, root='/home/liming/chenhan/project/pipe/result/DL-based-findcenter/data_txt',
                   num_samples=500, height=1280, width=1960, is_train=False, transform=None):
    super(Dataset, self).__init__()

    self.transform = transform
    self.is_train = is_train
    self.width = width
    self.height = height

    self.data_root = root

    num_txt_p = osp.join(root,'num.txt')
    with open(num_txt_p,'r') as f:
      num = f.readline()
    num = num.strip()

    self.nSamples = int(num)
    self.nSamples = min(self.nSamples, num_samples)

    if self.is_train:
      self.augment = iaa.Sequential([
        iaa.MotionBlur(k=5, angle=45),
        iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
        # iaa.SaltAndPepper(0.2,per_channel=True), 
        # iaa.AddToHueAndSaturation((-60, 60)),
        iaa.MultiplyBrightness((0.5, 1.5)),
        iaa.Affine(rotate=(-180, 180)),
        iaa.pillike.EnhanceSharpness(),
        # iaa.pillike.EnhanceColor(),
        iaa.Dropout(p=(0, 0.1))
        ], random_order=True)

  def __len__(self):
    return self.nSamples

  def __getitem__(self, index):
    assert index <= self.nSamples, 'index range error'

    index += 1

    try:
      img_root_i = osp.join(self.data_root,str(index)+'.txt')
      with open(img_root_i,'r') as f:
        lines = f.readlines()
      js_p = lines[0].strip()
      img_p = lines[1].strip()
      img = Image.open(img_p)
      img = img.convert("RGB")
    except IOError:
      print('Corrupted image for %d' % index)
      return self[index+1]
    
    ori_w, ori_h = img.size

    img = img.resize((self.width,self.height))
    if self.is_train:
      if random.uniform(0,1) < 0.2:
        pass
      else:
        img = np.asarray(img, dtype=np.uint8)
        img = self.augment(image = img)
        img = Image.fromarray(img)
    try:
      pt = fileutils.red_json(js_p)
      pt[0] = pt[0]/ori_w*self.width
      pt[1] = pt[1]/ori_h*self.height
    except IndexError:
      return self[index+1]
    # img.save('seeee_'+str(index)+'_'+str(label)+'_'+ raw_label +'.jpg')

    if self.transform is not None:
      img = self.transform(img)

    return img, pt[0], pt[1]


class ResizeNormalize(object):
  def __init__(self, size, interpolation=Image.BILINEAR):
    self.size = size
    self.interpolation = interpolation
    self.toTensor = transforms.ToTensor()

  def __call__(self, img):
    img = img.resize(self.size, self.interpolation)
    img = self.toTensor(img)
    img.sub_(0.5).div_(0.5)
    return img


class RandomSequentialSampler(sampler.Sampler):

  def __init__(self, data_source, batch_size):
    self.num_samples = len(data_source)
    self.batch_size = batch_size

  def __len__(self):
    return self.num_samples

  def __iter__(self):
    n_batch = len(self) // self.batch_size
    tail = len(self) % self.batch_size
    index = torch.LongTensor(len(self)).fill_(0)
    for i in range(n_batch):
      random_start = random.randint(0, len(self) - self.batch_size)
      batch_index = random_start + torch.arange(0, self.batch_size)
      index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
    # deal with tail
    if tail:
      random_start = random.randint(0, len(self) - self.batch_size)
      tail_index = random_start + torch.arange(0, tail)
      index[(i + 1) * self.batch_size:] = tail_index

    return iter(index.tolist())


class AlignCollate(object):

  def __init__(self, imgH=128, imgW=128, min_ratio=1):
    self.imgH = imgH
    self.imgW = imgW
    self.min_ratio = min_ratio

  def __call__(self, batch):
    images, labels_x, labels_y  = zip(*batch)
    b_labels_x = torch.LongTensor(labels_x)
    b_labels_y = torch.LongTensor(labels_y)

    imgH = self.imgH
    imgW = self.imgW

    transform = ResizeNormalize((imgW, imgH))
    images = [transform(image) for image in images]
    b_images = torch.stack(images)

    return b_images, b_labels_x, b_labels_y


if __name__ == "__main__":
    test()