import sys
import cv2
import pickle as pkl
from time import sleep, time


OUTPUT_FILE = './saved_frames/{}_{}.pkl'
KEYBOARD_CAPTURE = False


if __name__ == '__main__':
    frames_saved, records = 0, {}
    recordType = "stream"
    key = input("Keyboard Capture(Y/n)?:\n")
    if key.lower()=='y':
        print('\nPress Enter to Capture and q to quit.')
        KEYBOARD_CAPTURE=True
        recordType = "captures"

    try:
        from ZED import ZEDCamera
        zed_cam = ZEDCamera(resolution='1080')
        zed_cam.startStream()
        zed_album = zed_cam.takePicture()
        zed_width = zed_album.color.shape[1]
        records['ZED_L'], records['ZED_R'] = [], []
    except:
        print("\nZED Camera Not Connected!")

    try:
        from realSense import RealSense2
        rs_cam = RealSense2()
        reel = rs_cam.videoStream()
        rs_album = reel.send(None)
        records['RS'] = []
    except:
        print("\nRealSense Camera Not Connected!")

    # Tracking record duration
    starting = time()
    while True:
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        try:
            if KEYBOARD_CAPTURE:
                key = input()
                if key=='q': break

            if 'RS' in records.keys():
                rs_album = reel.send(None)
                records['RS'].append(rs_album.color)

            if 'ZED_L' in records.keys():
                zed_album = zed_cam.takePicture()
                zed_color = zed_album.color
                records['ZED_L'].append(zed_color[:,:round(zed_width/2),:3])
                records['ZED_R'].append(zed_color[:,round(zed_width/2):,:3])
                im = zed_color[:,:round(zed_width/2),:3]
                ims = cv2.resize(im, (960, 540))
                cv2.imshow("output", ims)
                cv2.waitKey(20)


            frames_saved += 1
            print("\n{} Frames Captured".format(frames_saved))

        except KeyboardInterrupt:
            if KEYBOARD_CAPTURE:
                print("\nSession Terminated.")
            else:
                print("\nStream Terminated with Duration: {}s".format(round(time() - starting, 1)))
            break
    cv2.destroyAllWindows()
    if 'ZED_L' in records.keys(): zed_cam.closeStream()
    if 'RS' in records.keys(): rs_cam.closeStream()

    with open(OUTPUT_FILE.format(str(int(starting)), recordType), 'wb') as file:
        pkl.dump(records, file)
