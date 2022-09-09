from scipy.spatial.transform import Rotation as R
import numpy as np
from glob import glob
import os
import pickle
from transformation import homogenous_transform, camera_matrix_to_fundamental_matrix
from triangulation import triangulate
import pandas as pd

trans = np.array([-120, 0, 0])
rotvec = np.array([0.0122,-0.0086,-0.0022])
r = R.from_rotvec(rotvec)
rotmatrix = r.as_matrix()
homo1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
homo2 = homogenous_transform(rotmatrix, trans)

camera_matrix_1 = np.array([[1400.59, 0, 1030.9 ], [0, 1400.59, 526.489], [0, 0, 1]])
camera_matrix_2 = np.array([[1399.43, 0, 1059.85], [0, 1399.43, 545.848], [0, 0, 1]])
P1 = np.dot(camera_matrix_1, homo1[:3,:])
P2 = np.dot(camera_matrix_2, homo2[:3,:])
F = camera_matrix_to_fundamental_matrix(camera_matrix_1, camera_matrix_2, rotmatrix, trans)

vid_id = 'HD1080_SN3404_15-37-46'
objs = ['tap', 'cup', 'pitcher', 'teabag']

obj_templates_dlc = {}
rotmatrix = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
for obj in objs:
    obj_dlc_dir = glob(f'/home/luke/Desktop/project/make_tea/dlc/*{obj}*')[0]
    h5_left = glob(os.path.join(obj_dlc_dir, 'labeled-data', vid_id + '-left', '*.h5'))[0]
    h5_right = glob(os.path.join(obj_dlc_dir, 'labeled-data', vid_id + '-right', '*.h5'))[0]
    config3d = obj_dlc_dir = glob(f'/home/luke/Desktop/project/make_tea/dlc3D/*{obj}*/config.yaml')[0]
    df = triangulate(config3d, h5_left, h5_right, P1, P2, F)
    bps = df.columns.get_level_values('bodyparts').unique()
    bp_3d = {}
    idx = pd.IndexSlice
    individual = df.columns.get_level_values('individuals').unique()[0]
    for bp in bps:
        bp_3d[bp] = df.loc[:, idx[:, individual, bp]].to_numpy().flatten()
    obj_templates_dlc[obj] = bp_3d

basedir = '/home/luke/Desktop/project/make_tea/Process_data/postprocessed/2022-05-26'
with open(os.path.join(basedir, 'transformations', 'dlc3d', 'obj_templates.pickle'), 'wb') as f:
    pickle.dump(obj_templates_dlc, f)
