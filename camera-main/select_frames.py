import pickle as pkl
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

file_ind = '1637605842'

with open('./saved_frames/' + file_ind + '_stream.pkl', 'rb') as file:
    album = pkl.load(file)
zed_r = np.array(album['ZED_R'])
zed_l = np.array(album['ZED_L'])

target_dir = './images/' + file_ind
j = 0

if os.path.exists(target_dir):
    pass
else:
    os.mkdir(target_dir)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
for i in range(zed_r.shape[0]):
    img = cv2.resize(zed_r[i], (960, 540))
    cv2.imshow('img ',img)
    k = cv2.waitKey(0)
    if k == ord('p'): # Not saving the frames
        pass
    elif k == ord('o'): # Saving the frames
        left_name = f'left-{j}.jpg'
        right_name = f'right-{j}.jpg'
        cv2.imwrite(target_dir + '/' + left_name, zed_l[i])
        cv2.imwrite(target_dir + '/' + right_name, zed_r[i])
        j +=1
    else: # Stop selecting
        cv2.destroyAllWindows()
        break
