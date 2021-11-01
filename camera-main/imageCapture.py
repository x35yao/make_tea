import numpy as np;
import cv2;

# This script is used to test functionality of one camera.

# Use 1, 2, 3, .. to capture video from other cameras
cap = cv2.VideoCapture(0)
print(cap.isOpened())
# variable to store name of captured image
img_name = "cam1.png"

while(cap.isOpened()):
# ret is a boolean which returns true if the successive frame can be grabbed
# frame stores the next frame of the video
    ret, frame = cap.read()

    cv2.imshow("Cam1", frame) #Cam1 is name of window in which video will open

    if not ret:
        break
# waitKey waits for a pressed key or waits for an event for at least a specified delay
    k = cv2.waitKey(1)
    if k%256 == 32:
        # SPACE pressed
        cv2.imwrite(img_name, frame) #image of object is captured
        cv2.destroyAllWindows()
        break

img = cv2.imread(img_name)

n = 0
transparency = 1

while(cap.isOpened()):

	ret, frame = cap.read()

	k = cv2.waitKey(1)

	if k%256 == 43:
		# + pressed to zoom in
		n = n + 1
		img = cv2.resize(img, (100+img.shape[1], 100+img.shape[0]))

	if k%256 == 45:
		# - pressed to zoom out
		if n > 0:
			cv2.destroyWindow("Zoom {}".format(n))
			n = n - 1
			img = cv2.resize(img, (img.shape[1]-100, img.shape[0]-100))

	if n > 0:
		frame = cv2.resize(frame, (img.shape[1], img.shape[0]))

	if k%256 == 62:
		# > pressed
		transparency = transparency + 0.1
	if k%256 == 60:
		# < pressed
		transparency = transparency - 0.1

	# addWeighted adds two scaled matrices and an offset
	blend = cv2.addWeighted(img, transparency, frame, 0.7, 0)
	cv2.imshow("Zoom {}".format(n), blend) #opens new window with overlain image

	if k%256 == 27:
		# ESC pressed
		break

cap.release()
cv2.destroyAllWindows()

# The image is stored in the directory of this script.
# When the script is rerun, the previous image stored is replaced with the new image captured.