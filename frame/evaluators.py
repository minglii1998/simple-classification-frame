from __future__ import print_function, absolute_import
import time
from time import gmtime, strftime
from datetime import datetime
from collections import OrderedDict

import torch

import numpy as np
from random import randint
from PIL import Image
import sys

from . import evaluation_metrics
from .evaluation_metrics import Accuracy
from .utils.meters import AverageMeter
from .utils.visualization_utils import recognition_vis, stn_vis

metrics_factory = evaluation_metrics.factory()

# from config import get_args
# global_args = get_args(sys.argv[1:])

class BaseEvaluator(object):
  def __init__(self, model, metric, criterion, use_cuda=True):
    super(BaseEvaluator, self).__init__()
    self.model = model
    self.metric = metric
    self.criterion = criterion 
    self.use_cuda = use_cuda
    self.device = torch.device("cuda" if use_cuda else "cpu")

  def evaluate(self, data_loader, step=1, print_freq=1, dataset=None, vis_dir=None):
    self.model.eval()

    batch_time = AverageMeter()
    data_time  = AverageMeter()

    # forward the network
    end = time.time()
    for i, datas in enumerate(data_loader):
      images, labels = datas

      images = images.to(self.device)
      labels = labels.to(self.device)

      data_time.update(time.time() - end)

      outputs = self._forward(images)
      total_loss_batch = self.criterion(outputs,labels)
      _, preds = torch.max(outputs, 1)

      batch_time.update(time.time() - end)
      end = time.time()

      if i == 0:
        images_all = images
        preds_all = preds
        targets_all = labels
        loss_all = total_loss_batch
      else:
        images_all = torch.cat((images_all,images),0)
        preds_all = torch.cat((preds_all,preds),0)
        targets_all = torch.cat((targets_all,labels),0)
        loss_all = loss_all + total_loss_batch

      if (i + 1) % print_freq == 0:
        print('[{}]\t'
              'Evaluation: [{}/{}]\t'
              'Time {:.3f} ({:.3f})\t'
              'Data {:.3f} ({:.3f})\t'
              # .format(strftime("%Y-%m-%d %H:%M:%S", gmtime()),
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      i + 1, len(data_loader),
                      batch_time.val, batch_time.avg,
                      data_time.val, data_time.avg))


    # Images from different batches
    num_samples = images_all.shape[0]
    Acc = (targets_all == preds_all).sum()/num_samples
    Loss_mean = loss_all/num_samples

    print('{0}: {1:.3f} \t Mean Loss: {2:.3f} '.format(self.metric, Acc.data, Loss_mean.data))


    #====== Visualization ======#
    # if vis_dir is not None:
    #   # recognition_vis(images, outputs['pred_rec'], targets, score_list, dataset, vis_dir)
    #   stn_vis(images, outputs['rectified_images'], outputs['ctrl_points'], outputs['pred_rec'],
    #           targets, score_list, outputs['pred_score'] if 'pred_score' in outputs else None, dataset, vis_dir)
    
    return Acc


  def _parse_data(self, inputs):
    raise NotImplementedError

  def _forward(self, inputs):
    raise NotImplementedError
    

class Evaluator(BaseEvaluator):

  def _forward(self, input_dict):
    self.model.eval()
    with torch.no_grad():
      output_dict = self.model(input_dict)
    return output_dict