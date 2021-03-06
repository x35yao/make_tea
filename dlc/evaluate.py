from glob import glob
import deeplabcut
from post_processing import interpolate_data, filter_3D_data
from utils import get_videos, get_h5files
from visualization import create_video_with_h5file
import os
import shutil
from deeplabcut.utils.auxfun_videos import VideoReader
import pandas as pd

def batch_evaluate():
    objs = ['teabag', 'cup', 'pitcher', 'tap']
    for obj in objs:
        config_path = glob(f'/home/luke/Desktop/project/make_tea/dlc/make_tea_{obj}*/config.yaml')[0]
        deeplabcut.evaluate_network(config_path, plotting=True)

def batch_analyze_video(videos, objs = ['pitcher', 'tap', 'teabag', 'cup'], shuffle = 1, make_video = True):
    for obj in objs:
        config_path = glob(f'/home/luke/Desktop/project/make_tea/dlc/make_tea_{obj}*/config.yaml')[0]
        for video in videos:
            obj_dir = os.path.join(os.path.dirname(video), obj)
            vid_name = os.path.basename(video)
            obj_video = os.path.join(obj_dir, vid_name)
            if not os.path.isdir(obj_dir):
                os.makedirs(obj_dir)
            if not os.path.isfile(obj_video):
                shutil.copyfile(video, obj_video)
            scorername = deeplabcut.analyze_videos(config_path, video, videotype='.mp4', auto_track = True, robust_nframes = True, save_as_csv = True, shuffle = shuffle, destfolder = obj_dir)
            if make_video:
                h5files = get_h5files(obj_dir)
                h5file = h5files[0]
                create_video_with_h5file(obj_video, h5file)
                # deeplabcut.create_video_with_all_detections(config_path, obj_video, videotype = '.mp4', shuffle = shuffle)


def batch_get_tracklets(videos):
    objs = ['teabag', 'cup', 'pitcher', 'tap']
    for obj in objs:
        config_path = glob(f'/home/luke/Desktop/project/make_tea/dlc/make_tea_{obj}*/config.yaml')[0]
        deeplabcut.convert_detections2tracklets(config_path, videos, videotype='mp4',
                                        shuffle=1, trainingsetindex=0)
        deeplabcut.stitch_tracklets(config_path, ['videofile_path'], videotype='mp4',
                            shuffle=1, trainingsetindex=0)

def batch_refine_tracklets():
    # TODO, we will see if we need this or not.
    pass

def batch_filter(h5files, filtertype="median", window_size = 5, ARdegree=3, MAdegree=1):
    for h5file in h5files:
        filter_3D_data(h5file,  filtertype = filtertype, window_size = window_size)


def batch_interpolate(vid_id, objs = ['pitcher', 'tap', 'teabag', 'cup'], filtertype="median", cameras = ['left'], windowlength=5, ARdegree=3, MAdegree=1, create_video = True,shuffle = 1):
    files = []
    for camera in cameras:
        for obj in objs:
            h5file = get_h5files(vid_id, obj, filtertype)
            # outputnames = interpolate_datafiltertypes= filtertype, windowlengths=windowlength,
            #     ARdegree=3, MAdegree=1,shuffle = shuffle)
            if create_video:
                for i, video in enumerate(videos):
                    create_video_with_h5file(config_path, video, outputnames[i])


def combine_h5files(h5files, to_csv = True, destdir = None):
    df_new = pd.DataFrame()
    for h5file in h5files:
        df = pd.read_hdf(h5file)
        df_new = pd.concat([df_new, df], axis = 1)
    if destdir == None:
        destdir = os.path.dirname(os.path.dirname(h5file))
    outputname = destdir + '/' + 'markers_trajectory_2d.h5'
    if os.path.isfile(outputname):
        print('Removing exsited file.')
        os.remove(outputname)
    df_new.to_hdf(outputname, mode = 'w', key = 'markers_trajectory_2d')
    if to_csv:
        df_new.to_csv(outputname.replace('.h5', '.csv'))
    print(f'The file is saved at {outputname}')
    return outputname
