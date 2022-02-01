from .post_processing import interpolate_data
import cv2
from glob import glob
from matplotlib import pyplot as plt
import pandas as pd
from skimage.draw import disk
from  deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.video_processor import VideoProcessorCV as vp
import os
import numpy as np


# import argparse

def browse_video_frame(video_path, index):
    # Initialize parser
    # parser = argparse.ArgumentParser()
    #
    # # Adding optional argument
    #
    # parser.add_argument('video_path',type=str,
    #                     help='the video to look at')
    #
    # parser.add_argument('initial_frame',type=int,
    #                     help='an integer for initial frame')
    # # Read arguments from command line
    # args = parser.parse_args()

    # video_id = '1642994619'
    #
    # video_path = glob(f'../camera-main/videos/{video_id}/left/{video_id}*x_linear.mp4')[0]

    # video_path = glob(f'../camera-main/videos/{video_id}/{video_id}-left.mp4')[0]

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap. read()
    count = 0
    imgs = []
    while success:
        imgs.append(image)
        success, image = vidcap. read()
        count +=1

    # index = args.initial_frame
    while(True):
        cv2.imshow(f'current image{index}', imgs[index])
        key = cv2.waitKey(0)

        if key == ord('x'):
            index -= 1
        elif key == ord('v'):
            index +=1
        elif key == ord('m'):
            index = int(input('Number of frame you want to jump to: '))
        elif key == ord('q'):
            break

        cv2.destroyAllWindows()

def create_video_with_h5file(config_path, video, h5file, suffix = None):
    '''
    This function create a new video with labels. Labels are from the h5file provided.

    config_path: The config file of the dlc project.
    video: The path to original video.
    h5file: The .h5 file that contains the detections from dlc.
    suffix: Usually it is the remove method to remove the nans. ('fill', 'interpolation', 'drop', 'ignore')

    '''

    cfg = auxiliaryfunctions.read_config(config_path)
    dotsize = cfg["dotsize"]

    file_name = os.path.splitext(video)[0]
    if not suffix == None:
        outputname = file_name + '_' + suffix +'.mp4'
    else:
        outputname = file_name + '_labeled.mp4'
    df = pd.read_hdf(h5file)
    bpts = [i for i in df.columns.get_level_values('bodyparts').unique()]
    numjoints = len(bpts)

    colorclass = plt.cm.ScalarMappable(cmap=cfg["colormap"])

    C = colorclass.to_rgba(np.linspace(0, 1, numjoints))
    colors = (C[:, :3] * 255).astype(np.uint8)
    clip = vp(fname=video, sname=outputname, codec="mp4v")
    ny, nx = clip.height(), clip.width()
    for i in range(clip.nframes):
        frame = clip.load_frame()
        plt.imshow(frame)
        fdata = df.loc[i]
        det_indices= df.columns[::3]
        for det_ind in det_indices:
            ind = det_ind[:-1]
            x = fdata[ind]['x']
            y = fdata[ind]['y']
            rr, cc = disk((y, x), dotsize, shape=(ny, nx))
            frame[rr, cc] = colors[bpts.index(det_ind[2])]
        clip.save_frame(frame)
    clip.close()
    print(f'Video is saved at {outputname}')

def create_interpolated_video(config_path,
    video,
    videotype="mp4",
    shuffle=1,
    trainingsetindex=0,
    filtertype="median",
    windowlength=5,
    p_bound=0.001,
    ARdegree=3,
    MAdegree=1,
    alpha=0.01,
    save_as_csv=False,
    destfolder=None,
    modelprefix="",
    track_method="",):

    h5files = interpolate_data(config_path, video, filtertype = filtertype, windowlength = windowlength, ARdegree = ARdegree, MAdegree= MAdegree, save_as_csv = True)
    for h5file in h5files:
        create_video_with_h5file(config_path, video, h5file, suffix = filtertype)
