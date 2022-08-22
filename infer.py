"""Train with optional Global Distance, Local Distance, Identification Loss."""
# from __future__ import print_function
# import sys
# sys.path.insert(0, '.')
from time import time
import torch
import numpy as np
import argparse
import cv2
from aligned_reid.model.Model import Model
import torchvision.transforms as T

trf = T.Compose([
    # T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.486, 0.459, 0.408], std=[0.229, 0.224, 0.225])
])

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
        x: pytorch Variable
    Returns:
        x: pytorch Variable, same shape as input      
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def EuclideanDistances(a,b):
    sq_a = a**2
    # sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sum_sq_a = torch.sum(sq_a,dim=1)
    sq_b = b**2
    # sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    sum_sq_b = torch.sum(sq_b,dim=1)
    bt = b.t()
    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt) + 1e-12)


class Config(object):
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--resize_h_w', type=eval, default=(256, 128))
        parser.add_argument('--model_weight_file', type=str, default='')
        args = parser.parse_known_args()[0]

        self.resize_h_w = args.resize_h_w

        # Whether to scale by 1/255
        self.scale_im = True
        self.im_mean = [0.486, 0.459, 0.408]
        self.im_std = [0.229, 0.224, 0.225]

        ###############
        # ReID Model  #
        ###############
        self.local_conv_out_channels = 128


def main():
    cfg = Config()
    model = Model(local_conv_out_channels=cfg.local_conv_out_channels)
    ckpt = torch.load('ckpts/model_weight.pth')
    model.load_state_dict(ckpt, strict=False)
    model.cuda().eval()

    img1 = cv2.imread('imgs/4.jpg')  
    img1 = cv2.resize(img1,(128, 256))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0).cuda()

    img2 = cv2.imread('imgs/5.jpg')
    img2 = cv2.resize(img2,(128, 256))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0).cuda()

    # reid test
    with torch.no_grad():
        global_feat1, _ = model(img1)
        global_feat2, _ = model(img2)
    global_feat1 = normalize(global_feat1)
    global_feat2 = normalize(global_feat2)

    print(EuclideanDistances(global_feat1, global_feat2).item()*100)

    # speed test
    # def nshots(n):
    #     with torch.no_grad():
    #         for _ in range(n):
    #             global_feat1, _ = model(img1)
    # nshots(20)  # warmup
    # tic = time()
    # nshots(100)
    # toc = time()
    # print(100 / (toc - tic))


if __name__ == '__main__':
    main()
