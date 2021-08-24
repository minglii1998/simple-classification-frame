import torch
from PIL import Image
import matplotlib.pyplot as plt
import os.path as osp

from torchvision import transforms


loader = transforms.Compose([
    transforms.ToTensor()])  

unloader = transforms.ToPILImage()

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    if image.dim() == 4:
        image = image.squeeze(0)
    image = unloader(image)
    return image




# def visulize(images_all,targets_all,pred_all,vis_dir):
#     num_samples = images_all.shape[0]

#     for i in range(num_samples):
#         image_i = (images_all[i]*0.5)+0.5
#         target_i = targets_all[i]

#         pred_i = pred_all[i]
#         bool_i = bool(target_i==pred_i)

#         target_i_weight = id2weight(int(target_i))
#         pred_i_weight = id2weight(int(pred_i))

#         image_i = tensor_to_PIL(image_i)
#         text = str(i) + '_' + str(bool_i) + '_' + str(target_i_weight) + '_' + str(pred_i_weight) + '.jpg'
#         image_i.save(osp.join(vis_dir,text))
