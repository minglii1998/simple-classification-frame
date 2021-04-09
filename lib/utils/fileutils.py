import os
import numpy as np
import cv2


mapping_dict_weight2id = {}
mapping_dict_weight2id['25'] = 1
mapping_dict_weight2id['5'] = 2
mapping_dict_weight2id['75'] = 3
mapping_dict_weight2id['10'] = 4
mapping_dict_weight2id['125'] = 5
mapping_dict_weight2id['15'] = 6
mapping_dict_weight2id['175'] = 7
mapping_dict_weight2id['20'] = 8

mapping_dict_id2weight = {}
mapping_dict_id2weight[0] = 'None'
mapping_dict_id2weight[1] = '2.5'
mapping_dict_id2weight[2] = '5'
mapping_dict_id2weight[3] = '7.5'
mapping_dict_id2weight[4] = '10'
mapping_dict_id2weight[5] = '12.5'
mapping_dict_id2weight[6] = '15'
mapping_dict_id2weight[7] = '17.5'
mapping_dict_id2weight[8] = '20'
    
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def weight2id(raw_label):
    # input is str type
    raw_label = raw_label.replace('.','')

    if raw_label in mapping_dict_weight2id.keys():
        id = mapping_dict_weight2id[raw_label]
    else:
        id = 0

    return id

def id2weight(id):
    # input is int

    if id in mapping_dict_id2weight.keys():
        weight = mapping_dict_id2weight[id]
    else:
        weight = 0

    return weight