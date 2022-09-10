from triangulation import triangulate
import deeplabcut
from deeplabcut.post_processing import filtering
from deeplabcut.utils import  auxiliaryfunctions
import os
from glob import glob
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from dlc.visualization import create_video_with_h5file
from dlc.evaluate import combine_h5files, analyze_video, serch_obj_h5files
if __name__ == '__main__':
    project_dir = '/Users/luke/Desktop/project/make_tea' # Modify this to your need.
    data_dir = os.path.join(project_dir, 'Process_data/postprocessed/2022-08-(17-21)')
    filterpredictions = True
    filtertype = 'median'
    make_video = True
    objs = ['teabag', 'pitcher', 'cup', 'tap']

    data_root, demo_dirs, data_files = next(os.walk(data_dir))
    DLC2D = os.path.join(project_dir, 'dlc')
    DLC3D = os.path.join(project_dir, 'dlc3D')
    dlc_root, dlc_dirs, dlc_files = next(os.walk(DLC3D))
    ### Run Deeplabcut to analyze videos
    for demo in demo_dirs:
        vid_left = glob(os.path.join(data_root, demo, 'left', '*.mp4'))[0]
        vid_right = glob(os.path.join(data_root, demo, 'right', '*.mp4'))[0]
        for vid in [vid_left, vid_right]:
            for obj in objs:
                config2d = glob(os.path.join(DLC2D, f'*{obj}*', 'config.yaml'))[0]
                obj_dir = os.path.join(os.path.dirname(vid), obj)
                analyze_video(config2d, vid, filterpredictions = filterpredictions, filtertype = filtertype, destfolder = obj_dir)

        #### Triangulation
        for obj in objs:
            if filterpredictions:
                h5_left = glob(os.path.join(os.path.dirname(vid_left), obj, '*filtered.h5'))[0]
                h5_right = glob(os.path.join(os.path.dirname(vid_right), obj, '*filtered.h5'))[0]
            else:
                h5_left = list(set(glob(os.path.join(os.path.dirname(vid_left), obj, '*.h5'))) - set(glob(os.path.join(os.path.dirname(vid_left), obj, '*filtered.h5'))))
                h5_right = list(set(glob(os.path.join(os.path.dirname(vid_left), obj, '*.h5'))) - set(glob(os.path.join(os.path.dirname(vid_right), obj, '*filtered.h5'))))
            scorer_dir = os.path.join(dlc_root, obj + '_3d')
            config3d = glob(scorer_dir + '/config.yaml')[0]
            destfolder_obj = os.path.join(data_root, demo, 'dlc3d')
            triangulate(config3d, h5_left, h5_right, destfolder_obj)
        #### Combine 3d files from different objects
        h5files = glob(os.path.join(data_root, demo, 'dlc3d', demo + '*.h5'))
        destfolder_demo = os.path.join(data_root, demo, 'dlc3d')
        combine_h5files(h5files, destdir = destfolder_demo, suffix = '3d')
