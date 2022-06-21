from triangulation import triangulate
import os
from glob import glob
import cv2
import numpy as np

data_dir = '/home/luke/Desktop/project/Process_data/postprocessed/2022-05-26'
data_root, data_dirs, data_files = next(os.walk(data_dir))

DLC3D = '/home/luke/Desktop/project/make_tea/dlc3D'
objs = ['tap', 'teabag', 'pitcher', 'cup']
dlc_root, dlc_dirs, dlc_files = next(os.walk(DLC3D))
videotype = 'mp4'



for d in dlc_dirs:
    if '3d' in d:
        scorer_dir = os.path.join(dlc_root, d)
        config3d = glob(scorer_dir + '/config.yaml')[0]
        for demo in data_dirs:
            demo_root, demo_dirs, demo_files = next(os.walk(os.path.join(data_root, demo)))
            vids = [[os.path.join(demo_root, f) for f in demo_files if 'left.mp4' in f or 'right.mp4' in f]]
            outdir = os.path.join(os.path.dirname(vids[0][0]), 'dlc3d', d)
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            triangulate(config3d, vids, videotype, filterpredictions=True, filtertype="median", destfolder=outdir, save_as_csv=True)


