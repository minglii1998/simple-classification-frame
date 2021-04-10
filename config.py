from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./')

import six
import os
import os.path as osp
import math
import argparse


parser = argparse.ArgumentParser(description="Softmax loss classification")
# data
parser.add_argument('--synthetic_train_data_dir', nargs='+', type=str, metavar='PATH',
                    default=['/share/zhui/reg_dataset/NIPS2014'])
parser.add_argument('--test_data_dir', type=str, metavar='PATH',
                    default='/share/zhui/reg_dataset/IIIT5K_3000')

parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-j', '--workers', type=int, default=8)
parser.add_argument('--height', type=int, default=128,
                    help="input height, default: 128 * 128")
parser.add_argument('--width', type=int, default=128,
                    help="input width, default: 128 * 128")
parser.add_argument('--num_train', type=int, default=math.inf)
parser.add_argument('--num_test', type=int, default=math.inf)
parser.add_argument('--aug', action='store_true', default=False,
                    help='whether use data augmentation.')
parser.add_argument('--image_path', type=str, default='',
                    help='the path of single image, used in demo.py.')

# model
parser.add_argument('--model_arch', type=str, default='resnet34',
                    help='the model to use.')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--n_group', type=int, default=1)
parser.add_argument('--num_class', type=int, default=9)

# optimizer
parser.add_argument('--lr', type=float, default=1,
                    help="learning rate of new parameters, for pretrained "
                         "parameters it is 10 times smaller than this")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0) # the model maybe under-fitting, 0.0 gives much better results.
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1,1,1])
parser.add_argument('--milestones', nargs='+', type=float, default=[2,5])

# training configs
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--start_save', type=int, default=0,
                    help="start saving checkpoints after specific epoch")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--test_freq', type=int, default=50)
parser.add_argument('--cuda', default=True, type=bool,
                    help='whether use cuda support.')

# testing configs
parser.add_argument('--evaluation_metric', type=str, default='accuracy')

# misc
working_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
parser.add_argument('--logs_dir', type=str, metavar='PATH',
                    default=osp.join(working_dir, 'logs'))
parser.add_argument('--real_logs_dir', type=str, metavar='PATH',
                    default='/media/mkyang/research/recognition/selfattention_rec')
parser.add_argument('--debug', action='store_true',
                    help="if debugging, some steps will be passed.")


def get_args(sys_args):
  global_args = parser.parse_args(sys_args)
  return global_args