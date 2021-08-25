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

def test_point(img,x,y,indx,tag=''):
    
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    cv2.circle(img,(x,y),2,(255,255,0),1)
    cv2.circle(img,(x,y),4,(255,255,0),1)
    cv2.circle(img,(x,y),6,(255,255,0),1)

    cv2.imwrite('seeee_'+str(indx)+tag+'.jpg',img)
    pass
