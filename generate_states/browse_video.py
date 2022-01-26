import cv2
from glob import glob
import argparse


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument('initial_frame',type=int,
                    help='an integer for initial frame')
# Read arguments from command line
args = parser.parse_args()

video_id = '1642994619'

video_path = glob(f'../camera-main/videos/{video_id}/{video_id}-leftDLC_dlcrnetms5_make_tea_01052022Jan5shuffle1_50000_full.mp4')[0]
# video_path = glob(f'../camera-main/videos/{video_id}/{video_id}-left.mp4')[0]

vidcap = cv2.VideoCapture(video_path)
success,image = vidcap. read()
count = 0
imgs = []
while success:
    imgs.append(image)
    success, image = vidcap. read()
    count +=1

index = args.initial_frame
while(True):
    cv2.imshow(f'current image{index}', imgs[index])
    key = cv2.waitKey(0)

    if key == ord('x'):
        index -= 1
    elif key == ord('v'):
        index +=1
    elif key == ord('q'):
        break

    cv2.destroyAllWindows()
