from __future__ import print_function, absolute_import
import time
from time import gmtime, strftime
from datetime import datetime
import gc
import os.path as osp
import sys
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

from . import evaluation_metrics
from .evaluation_metrics import Accuracy
from .utils import to_numpy
from .utils.meters import AverageMeter
from .utils.serialization import load_checkpoint, save_checkpoint

metrics_factory = evaluation_metrics.factory()

class BaseTrainer(object):
  def __init__(self, model, metric, logs_dir, criterion, iters=0, best_res=-1, grad_clip=-1, use_cuda=True):
    super(BaseTrainer, self).__init__()
    self.model = model
    self.metric = metric
    self.logs_dir = logs_dir
    self.logs_txt_dir = osp.join(logs_dir, 'log.txt')
    self.criterion = criterion 
    self.iters = iters
    self.best_res = best_res
    self.grad_clip = grad_clip
    self.use_cuda = use_cuda

    self.device = torch.device("cuda" if use_cuda else "cpu")

  def train(self, epoch, data_loader, optimizer, current_lr=0.0, 
            print_freq=100, is_debug=False,
            evaluator=None, test_loader=None,
            test_dataset=None, test_freq=500):

    self.model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    for i, datas in enumerate(data_loader):
      self.model.train()
      self.iters += 1
      images, labels_x, labels_y = datas
      labels = torch.cat((labels_x.unsqueeze(1),labels_y.unsqueeze(1)),1)

      images = images.to(self.device)
      labels = labels.to(self.device).float()

      data_time.update(time.time() - end)

      batch_size = images.size(0)
      outputs = self._forward(images)
      total_loss = self.criterion(outputs,labels)
      losses.update(total_loss.item(), batch_size)

      optimizer.zero_grad()
      total_loss.backward()

      if self.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
      optimizer.step()

      batch_time.update(time.time() - end)
      end = time.time()

      if self.iters % print_freq == 0:
        print('[{}]\t'
              'Epoch: [{}][{}/{}]\t'
              'Time {:.3f} ({:.3f})\t'
              'Data {:.3f} ({:.3f})\t'
              'Loss {:.3f} ({:.3f})\t'
              # .format(strftime("%Y-%m-%d %H:%M:%S", gmtime()),
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      epoch, i + 1, len(data_loader),
                      batch_time.val, batch_time.avg,
                      data_time.val, data_time.avg,
                      losses.val, losses.avg))

        with open(self.logs_txt_dir,'a') as f:
          f.write('[{}]\t'
              'Epoch: [{}][{}/{}]\t'
              'Time {:.3f} ({:.3f})\t'
              'Data {:.3f} ({:.3f})\t'
              'Loss {:.3f} ({:.3f})\t'
              # .format(strftime("%Y-%m-%d %H:%M:%S", gmtime()),
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      epoch, i + 1, len(data_loader),
                      batch_time.val, batch_time.avg,
                      data_time.val, data_time.avg,
                      losses.val, losses.avg))
          f.write('\n')


      #====== evaluation ======#
      if self.iters % test_freq == 0:
        # only symmetry branch
        res = evaluator.evaluate(test_loader, step=self.iters, dataset=test_dataset)

        if self.metric == 'accuracy':
          is_best = res > self.best_res
          self.best_res = max(res, self.best_res)
        else:
          raise ValueError("Unsupported evaluation metric:", self.metric)

        print('\n * Finished iters {:3d}  accuracy: {:5.1%}  best: {:5.1%}{}\n'.
          format(self.iters, res, self.best_res, ' *' if is_best else ''))

        # if epoch < 1:
        #   continue
        save_checkpoint({
          'state_dict': self.model.module.state_dict(),
          'iters': self.iters,
          'best_res': self.best_res,
        }, is_best, fpath=osp.join(self.logs_dir, 'checkpoint.pth.tar'))


    # collect garbage (not work)
    # gc.collect()

  def _parse_data(self, inputs):
    raise NotImplementedError

  def _forward(self, inputs, targets):
    raise NotImplementedError


class Trainer(BaseTrainer):

  def _forward(self, input_dict):
    self.model.train()
    output_dict = self.model(input_dict)
    return output_dict