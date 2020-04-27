import cv2
import os

import argparse
from numpy.core.records import fromarrays
import os
import sys
import random
import scipy.io
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math

import cv2
from models import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description='Guided-Stereo')
parser.add_argument('--datapath', default='2011_09_26_0011/',
                    help='datapath')
parser.add_argument('--loadmodel', default='checkpoints/psmnet-ft.tar',
                    help='load model')
parser.add_argument('--output_dir', default='bin/',
                    help='output directory')
parser.add_argument('--save', action='store_true', default=True, help='Save output')
parser.add_argument('--verbose', action='store_true', default=False, help='Print stats for each single image')
parser.add_argument('--no_cuda', action='store_true', default=False, help='cuda extraction')
parser.add_argument('--datatype', default='KITTI')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.maxdisp = 192

# set gpu id used
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

IMG_FORMAT = ('.jpg', '.bmp', '.png', '.mat')


def get_file_list(root_path, hint_path="lidar/"):
    left_path = 'image_2/'
    right_path = 'image_3'
    disp_path = 'disp_occ_0/'

    list = [img_name for img_name in os.listdir(os.path.join(root_path, disp_path)) if img_name.endswith(IMG_FORMAT)]
    list.sort()
    disp = [os.path.join(root_path, disp_path, img_name) for img_name in list]
    guided = [os.path.join(root_path, hint_path, img_name) for img_name in list]

    list = [img_name for img_name in os.listdir(os.path.join(root_path, left_path)) if img_name.endswith(IMG_FORMAT)]
    list.sort()
    left = [os.path.join(root_path, left_path, img_name) for img_name in list]
    right = [os.path.join(root_path, right_path, img_name) for img_name in list]

    return left, right, guided, disp


all_left, all_right, all_guide, all_disp = get_file_list(args.datapath)

# build model
model = featureNet(args.maxdisp)
if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'], strict=False)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def run(batch_idx, left_path, right_path):
    model.eval()

    a = cv2.imread(left_path, cv2.IMREAD_COLOR)
    b = cv2.imread(right_path, cv2.IMREAD_COLOR)
    sgm = cv2.StereoSGBM_create(0, 128, 11,uniquenessRatio=15,)  # block size

    disp = sgm.compute(a, b)
    cv2.filterSpeckles(disp, 0, 40, 128)
    _, disp = cv2.threshold(disp, 0, 128 * 16, cv2.THRESH_TOZERO)
    disp_scaled = (disp / 16.)

    display_and_save(batch_idx, disp_scaled, 0, 0)
    print(batch_idx)

# Dirty work to show/save results...
def display_and_save(batch_idx, disparity, top_pad, left_pad):
    disp_2show = cv2.applyColorMap(
        np.clip(50 + 2 * disparity[top_pad:, left_pad:], a_min=0, a_max=255.).astype(np.uint8),
        cv2.COLORMAP_JET)
    import scipy
    os.makedirs(args.output_dir + '/mat', exist_ok=True)
    scipy.io.savemat(args.output_dir + "/mat/%06d_10.mat" % batch_idx, {'mv_r': disparity})

    if args.save:
        os.makedirs(args.output_dir + '/res', exist_ok=True)
        cv2.imwrite(args.output_dir + "/res/%06d_10.png" % batch_idx, disp_2show)

# main
def main():
    if not os.path.exists(args.output_dir) and args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    ## Test ##
    for i in range(len(all_left)):
        run(i, all_left[i], all_right[i])


if __name__ == '__main__':
    main()

