from glob import  glob
import imageio
import pickle as pkl
import numpy as np
import cv2
from pathlib import Path

input_dir = './saved_frames/'
output_dir = './videos/'

filenames = glob(input_dir + '*.pkl')
for filename in filenames:
    with open(filename, 'rb') as file:
        album = pkl.load(file)
    zed_r = np.array(album['ZED_R'])
    zed_l = np.array(album['ZED_L'])

    for i in range(zed_r.shape[0]):
        zed_r[i] = zed_r[i][...,::-1].copy()
        zed_l[i] = zed_l[i][...,::-1].copy()
    video_ind = ''.join(list(filter(str.isdigit, filename)))
    try:
        Path(output_dir + video_ind).mkdir(exist_ok = False)
    except FileExistsError:
        print(output_dir + video_ind + 'already exists')
    video_name_left = output_dir + video_ind + '/left''.mp4'
    video_name_right = output_dir + video_ind + '/right''.mp4'


    imageio.mimwrite(video_name_left, zed_l , fps = 20)
    imageio.mimwrite(video_name_right, zed_r , fps = 20)
