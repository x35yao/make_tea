import numpy as np;
import cv2;

# This script captures images from the ZED and RealSense cameras and overlays
# them on the video feed. It can be used to recreate accurate placement of
# objects after they have been moved by gripper.

# Use 1, 2, 3, .. to capture video from other cameras
cap = cv2.VideoCapture(0)  # ZED camera
cap2 = cv2.VideoCapture(1)  # RS camera

num = "spam"
img_name = "ZED {}.png".format(num)
img2_name = "RS {}.png".format(num)
# Variables associated with RealSense have a '2'.

while(cap.isOpened() or cap2.isOpened()):
#ret is a boolean which returns true if the successive frame can be grabbed
#frame stores the next frame of the video
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()

    cv2.imshow("Cam1", frame) #Cam1 is name of window in which video will open
    cv2.imshow("Cam2", frame2)

    if not ret:
        break

    #waitKey waits for a pressed key or waits for an event for at least a specified delay
    k = cv2.waitKey(1)
    if k%256 == 32:
        # SPACE pressed
        cv2.imwrite(img_name, frame) #image of object is captured
        cv2.imwrite(img2_name, frame2)

        cv2.destroyAllWindows()
        break

    if k%256 == 27:
        # ESC pressed
        break

img = cv2.imread(img_name)
img2 = cv2.imread(img2_name)

n = 0  # n stores the number of times in which image is zoomed

transparency = -0.6

while cap.isOpened() or cap2.isOpened():

    ret, frame = cap.read()
    ret2, frame2 = cap2.read()

    k = cv2.waitKey(1)

    if k % 256 == 43:
        # + pressed to zoom in
        n = n + 1
        img = cv2.resize(img, (100 + img.shape[1], 100 + img.shape[0]))
        img2 = cv2.resize(img2, (100 + img2.shape[1], 100 + img2.shape[0]))

    if k % 256 == 45:
        # - pressed to zoom out
        if n > 0:
            cv2.destroyWindow("RS {}".format(n))
            cv2.destroyWindow("ZED {}".format(n))
            n = n - 1
            img = cv2.resize(img, (img.shape[1] - 100, img.shape[0] - 100))
            img2 = cv2.resize(img2, (img2.shape[1] - 100, img2.shape[0] - 100))

    if n > 0:
        frame = cv2.resize(frame, (img.shape[1], img.shape[0]))
        frame2 = cv2.resize(frame2, (img2.shape[1], img2.shape[0]))

    if k % 256 == 62:
        # > pressed
        transparency = transparency + 0.1
    if k % 256 == 60:
        # < pressed
        transparency = transparency - 0.1

    blend = cv2.addWeighted(img, transparency, frame, 0.7, 0)
    blend2 = cv2.addWeighted(img2, transparency, frame2, 0.7, 0)

    cv2.imshow("ZED {}".format(n), blend)  # opens new window with overlay image
    cv2.imshow("RS {}".format(n), blend2)

    if k % 256 == 27:
        # ESC pressed
        break

cap.release()
cv2.destroyAllWindows()
