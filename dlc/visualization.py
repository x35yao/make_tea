import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from dlc.post_processing import interpolate_data
import cv2
from glob import glob
from matplotlib import pyplot as plt
import pandas as pd
from skimage.draw import disk
from  deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.video_processor import VideoProcessorCV as vp
import os
import numpy as np
from tqdm import trange


# import argparse

def browse_video_frame(video_path, ind):

    vidcap = cv2.VideoCapture(video_path)
    ind = 0
    vidcap.set(1, ind)
    success,image = vidcap.read()
    while(True):
        cv2.imshow(f'current image{ind}', image)
        key = cv2.waitKey(0)
        if key == ord('x'):
            ind -= 1
            vidcap.set(1, ind)
            success,image = vidcap.read()
        elif key == ord('v'):
            ind += 1
            print(ind)
            vidcap.set(1, ind)
            success,image = vidcap.read()
            print(success)
        elif key == ord('m'):
            ind = int(input('Number of frame you want to jump to: '))
            vidcap.set(1, ind)
            success,image = vidcap.read()
        elif key == ord('q'):
            break

        cv2.destroyAllWindows()

from memory_profiler import profile

# @profile
def create_video_with_h5file(video, h5file, suffix = None, thresh = 0.98):
    '''
    This function create a new video with labels. Labels are from the h5file provided.

    video: The path to original video.
    h5file: The .h5 file that contains the detections from dlc.
    suffix: Usually it is the remove method to remove the nans. ('fill', 'interpolation', 'drop', 'ignore')

    '''
    dotsize = 6

    file_name = os.path.splitext(h5file)[0]
    outputname = file_name + '.mp4'
    if os.path.isfile(outputname):
        print('Video exists.')
        return
    df = pd.read_hdf(h5file)
    if 'scorer' in df.columns.names:
        df = df.droplevel(0, axis = 1)
    obj_bpts = []
    individuals = df.columns.get_level_values('individuals').unique()
    for individual in individuals:
        obj_name = ''.join([i for i in individual if not i.isdigit()])
        bpts = [i for i in df[individual].columns.get_level_values('bodyparts').unique()]
        obj_bpts += [obj_name + bpt for bpt in bpts]
    obj_bpts = sorted(set(obj_bpts))
    numjoints = len(obj_bpts)

    colorclass = plt.cm.ScalarMappable(cmap= 'rainbow')

    C = colorclass.to_rgba(np.linspace(0, 1, numjoints))
    colors = (C[:, :3] * 255).astype(np.uint8)
    clip = vp(fname=video, sname=outputname, codec="mp4v")
    nframes = clip.nframes
    ny, nx = clip.height(), clip.width()
    det_indices= df.columns[::3]
    for i in trange(nframes):
        frame = clip.load_frame()
        fdata = df.iloc[i]
        for det_ind in det_indices:
            individual = det_ind[0]
            obj_name = ''.join([i for i in individual if not i.isdigit()])
            bpt = det_ind[1]
            obj_bpt = obj_name + bpt
            ind = det_ind[:-1]
            x = fdata[ind]['x']
            y = fdata[ind]['y']
            likelihood = fdata[ind]['likelihood']
            if likelihood > thresh:
                rr, cc = disk((y, x), dotsize, shape=(ny, nx))
                frame[rr, cc] = colors[obj_bpts.index(obj_bpt)]
        clip.save_frame(frame)
    clip.close()
    print(f'Video is saved at {outputname}')
    return
def create_video_with_h5file_for_obj(video, h5file, suffix = None):
    '''
    This function create a new video with labels. Labels are from the h5file provided.

    video: The path to original video.
    h5file: The .h5 file that contains the detections from dlc.
    suffix: Usually it is the remove method to remove the nans. ('fill', 'interpolation', 'drop', 'ignore')

    '''
    dotsize = 12

    file_name = os.path.splitext(h5file)[0]
    outputname = file_name + '.mp4'
    df = pd.read_hdf(h5file)
    if 'scorer' in df.columns.names:
        df = df.droplevel(0, axis = 1)
    obj_bpts = []
    individuals = list(df.columns.get_level_values('individuals').unique())
    n_individuals = len(individuals)
    colorclass = plt.cm.ScalarMappable(cmap= 'rainbow')

    C = colorclass.to_rgba(np.linspace(0, 1, n_individuals))
    colors = (C[:, :3] * 255).astype(np.uint8)
    clip = vp(fname=video, sname=outputname, codec="mp4v")
    nframes = clip.nframes
    ny, nx = clip.height(), clip.width()
    det_indices= df.columns[::2]
    for i in trange(nframes):
        frame = clip.load_frame()
        fdata = df.loc[i]
        for det_ind in det_indices:
            individual = det_ind[0]
            ind = det_ind[:-1]
            x = fdata[ind]['x']
            y = fdata[ind]['y']
            rr, cc = disk((y, x), dotsize, shape=(ny, nx))
            frame[rr, cc] = colors[individuals.index(individual)]
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


# if __name__== '__main__':
#     vid_id = '1645721497'
#
#     camera = 'left'
#     obj = 'tap'
#     # video = f'/home/luke/Desktop/project/make_tea/camera-main/videos/{vid_id}/{camera}/{obj}/{vid_id}-{camera}.mp4'
#     video = f'/home/luke/Desktop/project/make_tea/camera-main/videos/{vid_id}/{camera}/{obj}/1645721497-leftDLC_dlcrnetms5_make_tea_tapJan29shuffle1_200000_el.mp4'
#     browse_video_frame(video, 1)
from glob import glob
if __name__== '__main__':
    vid_id = '318539'
    obj = 'cup'
    camera = 'left'
    video = glob(f'/home/luke/Desktop/project/make_tea/Process_data/postprocessed/2022-08-(17-21)/{vid_id}/{camera}/{vid_id}-{camera}.mp4')[0]
    h5file = glob(f'/home/luke/Desktop/project/make_tea/Process_data/postprocessed/2022-08-(17-21)/{vid_id}/{obj}_3d/{vid_id}-{camera}*.h5')[0]
    create_video_with_h5file_for_obj( video, h5file)
