from __future__ import absolute_import
import sys
sys.path.append('./')

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import os.path as osp
import numpy as np
import math
import time

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision

from config import get_args
from lib import datasets, evaluation_metrics, models
from lib.models import resnet, mobilnet_v3
from lib.datasets.dataset_classification import Dataset, AlignCollate
from lib.datasets.concatdataset_classification import ConcatDataset
from lib.trainers import Trainer
from lib.evaluators import Evaluator
from lib.utils.serialization import load_checkpoint, save_checkpoint
from lib.utils.osutils import make_symlink_if_not_exists

global_args = get_args(sys.argv[1:])


def get_data(data_dir, num_samples, height, width, batch_size, workers, is_train):
  if isinstance(data_dir, list):
    dataset_list = []
    for data_dir_ in data_dir:
      dataset_list.append(Dataset(data_dir_, num_samples,height,width,is_train=is_train))
    dataset = ConcatDataset(dataset_list)
  else:
    dataset = Dataset(data_dir, num_samples,height,width)
  print('total image: ', len(dataset))

  if is_train:
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
      shuffle=True, pin_memory=True, drop_last=True,
      collate_fn=AlignCollate(imgH=height, imgW=width))
  else:
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
      shuffle=False, pin_memory=True, drop_last=False,
      collate_fn=AlignCollate(imgH=height, imgW=width))

  return dataset, data_loader

def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True

  args.cuda = args.cuda and torch.cuda.is_available()
  if args.cuda:
    print('using cuda.')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
  else:
    torch.set_default_tensor_type('torch.FloatTensor')

  if not args.evaluate:
    # make symlink
    make_symlink_if_not_exists(osp.join(args.real_logs_dir, args.logs_dir), osp.dirname(osp.normpath(args.logs_dir)))

  # Save the args to disk
  if not args.evaluate:
    cfg_save_path = osp.join(args.logs_dir, 'cfg.txt')
    cfgs = vars(args)
    with open(cfg_save_path, 'w') as f:
      for k, v in cfgs.items():
        f.write('{}: {}\n'.format(k, v))

  # Create data loaders
  if args.height is None or args.width is None:
    args.height, args.width = (128, 128)

  if not args.evaluate: 
    train_dataset, train_loader = \
      get_data(args.synthetic_train_data_dir, args.num_train,args.height, args.width, args.batch_size, args.workers, True)
  test_dataset, test_loader = \
    get_data(args.test_data_dir, args.num_test, args.height, args.width, args.batch_size, args.workers, False)

  # Create model
  if args.model_arch == 'resnet34':
    model = resnet.resnet34(num_classes=args.num_class)
    print('########## Using resnet34')

  elif args.model_arch == 'resnet18':
    model = resnet.resnet18(num_classes=args.num_class)
    print('########## Using resnet18')

  elif args.model_arch == 'mobilenet_v2':
    model = torchvision.models.mobilenet_v2(num_classes=args.num_class)
    print('########## Using mobilenet_v2')

  else:
    print('Wrong Model!')
    return

  criterion = nn.CrossEntropyLoss()

  # Load from checkpoint
  if args.evaluation_metric == 'accuracy':
    best_res = 0
  else:
    raise ValueError("Unsupported evaluation metric:", args.evaluation_metric)

  start_epoch = 0
  start_iters = 0

  if args.resume:
    checkpoint = load_checkpoint(args.resume)
    model.load_state_dict(checkpoint['state_dict'])

    # compatibility with the epoch-wise evaluation version
    if 'epoch' in checkpoint.keys():
      # start_epoch = checkpoint['epoch']
      start_epoch = 0
    else:
      # start_iters = checkpoint['iters']
      start_epoch = int(start_iters // len(train_loader)) if not args.evaluate else 0
      start_iters = 0
      start_epoch = 0
    best_res = 0
    print("=> Start iters {}  best res {:.1%}"
          .format(start_iters, best_res))
  
  if args.cuda:
    device = torch.device("cuda")
    model = model.to(device)
    model = nn.DataParallel(model)

  # Evaluator
  evaluator = Evaluator(model, args.evaluation_metric, args.logs_dir, criterion, args.cuda)

  if args.evaluate:
    print('Test on {0}:'.format(args.test_data_dir))
    if len(args.vis_dir) > 0:
      vis_dir = osp.join(args.logs_dir, args.vis_dir)
      if not osp.exists(vis_dir):
        os.makedirs(vis_dir)
    else:
      vis_dir = None

    start = time.time()
    evaluator.evaluate(test_loader, dataset=test_dataset, vis_dir=vis_dir)
    print('it took {0} s.'.format(time.time() - start))
    return

  # Optimizer
  param_groups = model.parameters()
  param_groups = filter(lambda p: p.requires_grad, param_groups)
  optimizer = optim.Adadelta(param_groups, lr=args.lr, weight_decay=args.weight_decay)
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

  # Trainer
  loss_weights = {}
  loss_weights['loss_rec'] = 1.
  if args.debug:
    args.print_freq = 1
  trainer = Trainer(model, args.evaluation_metric, args.logs_dir, criterion, 
                    iters=start_iters, best_res=best_res, grad_clip=args.grad_clip,
                    use_cuda=args.cuda)

  # Start training
  evaluator.evaluate(test_loader, step=0, dataset=test_dataset)
  for epoch in range(start_epoch, args.epochs):
    scheduler.step(epoch)
    current_lr = optimizer.param_groups[0]['lr']
    trainer.train(epoch, train_loader, optimizer, current_lr,
                  print_freq=args.print_freq,
                  is_debug=args.debug,
                  evaluator=evaluator, 
                  test_loader=test_loader, 
                  test_dataset=test_dataset,
                  test_freq=args.test_freq)

  # Final test
  print('Test with best model:')
  checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
  model.module.load_state_dict(checkpoint['state_dict'])
  evaluator.evaluate(test_loader, dataset=test_dataset)

if __name__ == '__main__':
  # parse the config
  args = get_args(sys.argv[1:])
  main(args)