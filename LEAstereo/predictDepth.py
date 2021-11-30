from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10

import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from retrain.LEAStereo import LEAStereo 

from config_utils.predict_args import obtain_predict_args
from utils.multadds_count import count_parameters_in_MB, comp_multadds
from time import time
from struct import unpack
import matplotlib.pyplot as plt
import pickle as pkl
import re
import numpy as np
import pdb
from path import Path

opt = obtain_predict_args()
print(opt)

torch.backends.cudnn.benchmark = True

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Building LEAStereo model')
model = LEAStereo(opt)

print('Total Params = %.2fMB' % count_parameters_in_MB(model))
print('Feature Net Params = %.2fMB' % count_parameters_in_MB(model.feature))
print('Matching Net Params = %.2fMB' % count_parameters_in_MB(model.matching))
   
mult_adds = comp_multadds(model, input_size=(3,opt.crop_height, opt.crop_width)) #(3,192, 192))
print("compute_average_flops_cost = %.2fMB" % mult_adds)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=True)      
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def test_transform(temp_data, crop_height, crop_width):
    _, h, w=np.shape(temp_data)

    if h <= crop_height and w <= crop_width: 
        # padding zero 
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp    
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3,crop_height,crop_width],'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w

def load_data(leftname, rightname):
    left = np.array(Image.open(leftname))
    right = np.array(Image.open(rightname))
    im_height, im_width, _ = left.shape 
    left = skimage.transform.rescale(left, (opt.crop_height/im_height, opt.crop_width/im_width, 1))
    right = skimage.transform.rescale(right, (opt.crop_height/im_height, opt.crop_width/im_width, 1))
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]	
    #r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data


def test_kitti(leftname, rightname, savename):
    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)
 
    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():        
        prediction = model(input1, input2)
        
    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height and width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[0, :, :]
    skimage.io.imsave(savename, (temp * 256).astype('uint16'))

def getDisparityMap(leftname, rightname):  
    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)

    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()

    start_time = time()
    with torch.no_grad():
        prediction = model(input1, input2)
    end_time = time()
    
    print("Processing time: {:.4f}".format(end_time - start_time))
    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height or width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[0, :, :]
#     plot_disparity(savename, temp, 192)
#     savename_pfm = savename.replace('png','pfm') 
#     temp = np.flipud(temp)
    return temp

def plot_disparity(savename, data, max_disp):
    plt.imsave(savename, data, vmin=0, vmax=max_disp, cmap='turbo')


if __name__ == "__main__":
    file_path = opt.data_path
    left_filelist = os.listdir(file_path + 'left_camera/')
    right_filelist = os.listdir(file_path + 'right_camera/')
    for index in range(len(left_filelist)):
        current_file = left_filelist[index]
        if current_file in right_filelist:
            if False:
                leftname = file_path + 'left_camera/' + current_file
                rightname = file_path + 'right_camera/' + current_file
                savename = "{}_disp.npy".format(opt.save_path + current_file.split('.')[0])
                disp_map = getDisparityMap(leftname, rightname)
                np.save(savename, disp_map)
            if opt.kitti2015:
                leftname = file_path + 'left_camera/' + current_file
                rightname = file_path + 'right_camera/' + current_file
                savename = "{}_disp_kitti.npy".format(opt.save_path + current_file.split('.')[0])
                disp_map = getDisparityMap(leftname, rightname)
                np.save(savename, disp_map)
