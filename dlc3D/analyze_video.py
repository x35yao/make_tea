import deeplabcut
import os
from glob import glob
import sys
import shutil

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from transformation import camera_matrix_to_fundamental_matrix, homogenous_transform
import numpy as np
from dlc.utils import combine_h5files, serch_obj_h5files
import yaml

if __name__ == '__main__':
    # Load data
    with open('../Process_data/postprocessed/2022-10-27/task_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    project_dir = config['project_path']  # Modify this to your need.
    data_dir = os.path.join(project_dir, config['postprocessed_dir'])
    filterpredictions = True
    filtertype = 'median'
    make_video = True
    objs = config['objects']
    cams = config['cameras']

    data_root, demo_dirs, data_files = next(os.walk(data_dir))
    DLC3D = os.path.join(project_dir, 'dlc3D')
    dlc_root, dlc_dirs, dlc_files = next(os.walk(DLC3D))
    ### Run Deeplabcut to analyze videos
    for i, demo in enumerate(sorted(demo_dirs)):
        if demo == 'transformations' or demo == 'processed':
            continue
        files = glob(os.path.join(data_root, demo, '*','*3d*'))
        files_to_remove = [f for f in files if 'leastereo' not in f]
        for f in files_to_remove:
            os.remove(f)
        vids = []
        h5files_3d = []
        for cam in cams:
            vids.append(glob(os.path.join(data_root, demo,f'*{cam}.mp4'))[0])
        for obj in objs:
            config2d = glob(os.path.join(project_dir, 'dlc', f'*{obj}*', 'config.yaml'))[0]
            scorer_dir_3d = os.path.join(dlc_root, obj + '_3d')
            config3d = glob(scorer_dir_3d + '/config.yaml')[0]
            destfolder = os.path.join(data_root, demo, obj)
            ##### Triangulate ###############
            deeplabcut.triangulate(config3d, [vids], filterpredictions = filterpredictions, filtertype = filtertype, destfolder = destfolder, save_as_csv = True)
            ##### Create labeled video ##############
            deeplabcut.create_labeled_video(config2d, vids, videotype = 'mp4', destfolder = destfolder, filtered = filterpredictions)
            h5files_3d.append(os.path.join(destfolder, f'{demo}_{obj}3d.h5'))
        destfolder_demo = os.path.join(data_root, demo, 'dlc3d')
        combine_h5files(h5files_3d, destdir = destfolder_demo, suffix = '3d')


