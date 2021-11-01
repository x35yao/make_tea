import numpy as np;
import cv2;

n = 428671
img_RS_color = np.load('/home/p4bhattachan/gripper/3DCameraServer/testImages/npyFiles/{}_RS_color.npy'.format(n))
cv2.imshow('RS Color Image {}'.format(n), img_RS_color)
#
# # img_RS_depth = np.load('/home/p4bhattachan/gripper/3DCameraServer/testImages/npyFiles/{}_RS_depth.npy'.format(n))
# # cv2.imshow('RS Depth Image {}'.format(n), img_RS_depth)
#
# img_ZED_color = np.load('/home/p4bhattachan/gripper/3DCameraServer/testImages/npyFiles/{}_ZED_color.npy'.format(n))
# cv2.imshow('ZED Color Image {}'.format(n), img_ZED_color)
#
# # img_ZED_depth = np.load('/home/p4bhattachan/gripper/3DCameraServer/testImages/npyFiles/{}_ZED_depth.npy'.format(n))
# # cv2.imshow('ZED Depth Image {}'.format(n), img_ZED_depth)

cv2.waitKey(0)
cv2.destroyAllWindows()


