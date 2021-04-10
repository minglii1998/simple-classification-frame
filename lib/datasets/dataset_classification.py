from __future__ import absolute_import

# import sys
# sys.path.append('./')

import os
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
  def __init__(self, root='/home/liming/ShareWindows/Free_zome/Self-labeled_data/obj_label/crop', num_samples=10000, height=128, width=128, is_train=False, transform=None):
    super(Dataset, self).__init__()

    self.transform = transform
    self.is_train = is_train
    self.width = width
    self.height = height

    imgs, _, _ = fileutils.get_files(root)
    self.img_list = imgs

    self.nSamples = int(len(self.img_list))
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

    try:
      img = Image.open(self.img_list[index])
    except IOError:
      print('Corrupted image for %d' % index)
      return self[index]
    
    img = img.resize((self.width,self.height))
    if self.is_train:
      if random.uniform(0,1) < 0.2:
        pass
      else:
        img = np.asarray(img, dtype=np.uint8)
        img = self.augment(image = img)
        img = Image.fromarray(img)

    # Sample: 1116_hly_3_0_2.5.jpg // 1116_hly_23_0_5.jpg
    raw_label = self.img_list[index].split('_')[-1] # 2.5.jpg // 5.jpg
    raw_label = raw_label.strip('.jpg') # 2.5 // 5

    label = fileutils.weight2id(raw_label)

    # img.save('seeee_'+str(index)+'_'+str(label)+'_'+ raw_label +'.jpg')

    if self.transform is not None:
      img = self.transform(img)

    return img, label


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
    images, labels = zip(*batch)
    b_labels = torch.LongTensor(labels)

    imgH = self.imgH
    imgW = self.imgW

    transform = ResizeNormalize((imgW, imgH))
    images = [transform(image) for image in images]
    b_images = torch.stack(images)

    return b_images, b_labels


def test():
  # lmdb_path = "/share/zhui/reg_dataset/NIPS2014"
  lmdb_path = "/share/zhui/reg_dataset/IIIT5K_3000"
  train_dataset = LmdbDataset(root=lmdb_path, voc_type='ALLCASES_SYMBOLS', max_len=50)
  batch_size = 1
  train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=AlignCollate(imgH=64, imgW=256, keep_ratio=False))

  for i, (images, labels, label_lens) in enumerate(train_dataloader):
    # visualization of input image
    # toPILImage = transforms.ToPILImage()
    images = images.permute(0,2,3,1)
    images = to_numpy(images)
    images = images * 0.5 + 0.5
    images = images * 255
    for id, (image, label, label_len) in enumerate(zip(images, labels, label_lens)):
      image = Image.fromarray(np.uint8(image))
      # image = toPILImage(image)
      image.show()
      print(image.size)
      print(labels2strs(label, train_dataset.id2char, train_dataset.char2id))
      print(label_len.item())
      input()


if __name__ == "__main__":
    test()