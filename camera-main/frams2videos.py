from glob import  glob
import imageio
import pickle as pkl
import numpy as np
import cv2

input_dir = './saved_frames/'
output_dir = './videos/'

filenames = glob(input_dir + '*.pkl')
for filename in filenames:
    print(filename)
    with open(filename, 'rb') as file:
        album = pkl.load(file)
    zed_r = np.array(album['ZED_R'])
    zed_l = np.array(album['ZED_L'])

    for i in range(zed_r.shape[0]):
        zed_r[i] = zed_r[i][...,::-1].copy()
        zed_l[i] = zed_l[i][...,::-1].copy()
    video_name_left = output_dir + 'left_' + filename.strip(input_dir) + '.mp4'
    video_name_right = output_dir + 'right_' + filename.strip(input_dir) + '.mp4'
    imageio.mimwrite(video_name_left, zed_l , fps = 20)
    imageio.mimwrite(video_name_right, zed_r , fps = 20)
