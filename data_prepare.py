from cv2 import cv2
import numpy as np

import os
import os.path as osp

import json

import torch
import random

def get_filelist(path):
    name_dict = {}
    # {name:{img:xx,js:xx}}
    for home ,dirs, files in os.walk(path):
        for filename in files:
            suffix =filename.split('.')[-1]
            name = filename.split('.')[0]

            if name in name_dict.keys():
                pass
            else:
                name_dict[name] = {}

            if suffix == 'jpg' or suffix == 'png':
                name_dict[name]['img'] = os.path.join(home, filename)
            if suffix == 'json':
                name_dict[name]['js'] = os.path.join(home, filename)

    return name_dict


def red_json(js_p):
    # read 5 points from a json path
    with open(js_p,'r',encoding='utf8')as fp:
        json_data = json.load(fp)
    pts = json_data['shapes'][0]['points']
    pts = pts[1:5]
    pts = torch.tensor(pts).float()
    pt = pts.mean(0)
    return pt


def write_to_txt(txt_path):

    name_dict = get_filelist('/home/liming/chenhan/data/center')

    count = 0
    for k in name_dict.keys():
        count += 1

        js_p = name_dict[k]['js']
        img_p = name_dict[k]['img']

        txt_path_i = osp.join(txt_path,str(count)+'.txt')

        with open(txt_path_i,'a') as f:
            f.write(js_p+'\n')
            f.write(img_p+'\n')


    txt_path_num = osp.join(txt_path,'num.txt')
    with open(txt_path_num,'a') as f:
        f.write(str(count))


def write_to_txt_seperate(txt_path_train,txt_path_test):

    name_dict = get_filelist('/home/liming/chenhan/data/center')

    count_test = 0
    count_train = 0
    
    for k in name_dict.keys():

        js_p = name_dict[k]['js']
        img_p = name_dict[k]['img']

        if random.uniform(0,1) < 0.1:
            # test
            count_test += 1
            txt_path_i = osp.join(txt_path_test,str(count_test)+'.txt')
        else:
            # train
            count_train += 1
            txt_path_i = osp.join(txt_path_train,str(count_train)+'.txt')

        with open(txt_path_i,'a') as f:
            f.write(js_p+'\n')
            f.write(img_p+'\n')


    txt_path_num = osp.join(txt_path_test,'num.txt')
    with open(txt_path_num,'a') as f:
        f.write(str(count_test))

    txt_path_num = osp.join(txt_path_train,'num.txt')
    with open(txt_path_num,'a') as f:
        f.write(str(count_train))


if __name__ == '__main__':
    write_to_txt('/home/liming/chenhan/project/pipe/result/DL-based-findcenter/data_txt_50')

    write_to_txt_seperate('/home/liming/chenhan/project/pipe/result/DL-based-findcenter/data_txt_train',
                    '/home/liming/chenhan/project/pipe/result/DL-based-findcenter/data_txt_test')

    # name_dict = get_filelist('/home/liming/chenhan/data/center')
    # for k in name_dict.keys():

    #     js_p = name_dict[k]['js']
    #     img_p = name_dict[k]['img']

    #     pts = red_json(js_p)
    #     pass


    pass
