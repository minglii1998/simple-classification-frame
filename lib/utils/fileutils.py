import os
import numpy as np
import cv2

import json

import torch

def red_json(js_p):
    # read 5 points from a json path
    with open(js_p,'r',encoding='utf8')as fp:
        json_data = json.load(fp)
    pts = json_data['shapes'][0]['points']
    pts = pts[1:5]
    pts = torch.tensor(pts).float()
    pt = pts.mean(0)
    return pt