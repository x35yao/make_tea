import numpy as np;
import cv2

# This script reads numpy files and overlays the corresponding images on the RealSense and ZED camera video feeds.

cap = cv2.VideoCapture(0) #ZED camera
cap2 = cv2.VideoCapture(1) #RS camera

#Images are stored as .npy
num = 143010
# img = cv2.imread('/home/p4bhattachan/gripper/3DCameraServer/testImages/20180921/ZED 17.png')
# img2 = cv2.imread('/home/p4bhattachan/gripper/3DCameraServer/testImages/20180921/RS 17.png')
img = np.load('/home/p4bhattachan/gripper/3DCameraServer/testImages/npyFiles/00sugarbox/{}_ZED_color.npy'.format(num))
# cv2.imwrite('{}_ZED_color.png'.format(num), img)

img2 = np.load('/home/p4bhattachan/gripper/3DCameraServer/testImages/npyFiles/00sugarbox/{}_RS_color.npy'.format(num))
# cv2.imwrite('{}_RS_color.png'.format(num), img2)
n = 0 # n stores the number of times in which image is zoomed

transparency = 0.6

img_name = "{}_ZED_blend.png".format(num)
img2_name = "{}_RS_blend.png".format(num)

ret, frame = cap.read()
if not img.shape[:2] == frame.shape[:2]:
    # Size of ZED image and ZED frame not equal
    img = cv2.resize(img, (frame.shape[1], frame.shape[0]))

while cap.isOpened() or cap2.isOpened():

    ret, frame = cap.read()
    ret2, frame2 = cap2.read()

    k = cv2.waitKey(1)

    if k%256 == 43:
        # + pressed to zoom in
        n = n + 1
        img = cv2.resize(img, (100+img.shape[1], 100+img.shape[0]))
        img2 = cv2.resize(img2, (100+img2.shape[1], 100+img2.shape[0]))

    if k%256 == 45:
        # - pressed to zoom out
        if n > 0:
            cv2.destroyWindow("RS {}".format(n))
            cv2.destroyWindow("ZED {}".format(n))
            n = n - 1
            img = cv2.resize(img, (img.shape[1]-100, img.shape[0]-100))
            img2 = cv2.resize(img2, (img2.shape[1]-100, img2.shape[0]-100))

    if n > 0:
        frame = cv2.resize(frame, (img.shape[1], img.shape[0]))
        frame2 = cv2.resize(frame2, (img2.shape[1], img2.shape[0]))

    if k%256 == 62:
	    # > pressed
	    transparency = transparency + 0.1
    if k%256 == 60:
        # < pressed
        transparency = transparency - 0.1

    blend = cv2.addWeighted(img, transparency, frame, 0.7, 0)
    blend2 = cv2.addWeighted(img2, transparency, frame2, 0.7, 0)

    cv2.imshow("ZED {}".format(n), blend) # opens new window with overlay image
    cv2.imshow("RS {}".format(n), blend2)

    if k%256 == 32:
        # SPACE pressed
        cv2.imwrite(img_name, blend) #image of object is captured
        cv2.imwrite(img2_name, blend2)

    if k%256 == 27:
        # ESC pressed
        break

cap.release()
cv2.destroyAllWindows()
