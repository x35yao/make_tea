from predictDepth import *
import zedStereoUtility as zed
import pandas as pd
from glob import glob
import cv2
import os
from tqdm import trange

def get_videos(vid_id, obj = None, filtertypes = None, cameras = ['left', 'right'], base_dir = '/home/luke/Desktop/project/make_tea'):
    '''
    This function output video path given the video ID.

    vid_id: The video ID
    obj: The object folder the video is sit in. Options: 'teabag', 'cup', 'pitcher', 'tap'. If None, video in one level higher will be outputed.
    kernel: Given kernel, this function output the video filtered by the kernel. If None, the original non-filtered vildeo path will be outputed.
    base_dir: The path to the make_tea folder.
    '''
    videos = []

    if not isinstance(cameras, list):
        cameras = [cameras]
    for camera in cameras:
        if filtertypes == None:
            if obj == None:
                video = f'{base_dir}/camera-main/videos/{vid_id}/{camera}/{vid_id}-{camera}.mp4'
            else:
                video = f'{base_dir}/camera-main/videos/{vid_id}/{camera}/{obj}/{vid_id}-{camera}.mp4'
        else:
            if not isinstance(filtertypes, list):
                suffix = f'_{filtertypes}'
            else:
                temp = '_'.join(filtertypes)
                suffix = f'_{temp}'
            if obj == None:
                video = f'{base_dir}/camera-main/videos/{vid_id}/{camera}/{vid_id}-{camera}{suffix}.mp4'
            else:
                video = f'{base_dir}/camera-main/videos/{vid_id}/{camera}/{obj}/{vid_id}-{camera}{suffix}.mp4'
        videos.append(video)
    return videos



def get_combined_h5file(vid_id, camera, base_dir = '/home/luke/Desktop/project/make_tea'):
    h5file = glob(f'{base_dir}/camera-main/videos/{vid_id}/{camera}/combined.h5')[0]

    return h5file

def triangulate(videos, h5file, outdir = None, to_csv = True):
    df = pd.read_hdf(h5file)
    df_new = df.copy()
    df_new.columns = df_new.columns.set_levels(['z', 'x', 'y'], level='coords')
    x_inds = df_new.columns.values[::3]
    y_inds = df_new.columns.values[1::3]
    z_inds = df_new.columns.values[2::3]

    video_left = [vid for vid in videos if 'left' in vid][0]
    video_right = [vid for vid in videos if 'right' in vid][0]
    vidcap_left = cv2.VideoCapture(video_left)
    vidcap_right = cv2.VideoCapture(video_right)

    nframes_left = int(vidcap_left.get(cv2.CAP_PROP_FRAME_COUNT))
    nframes_right = int(vidcap_right.get(cv2.CAP_PROP_FRAME_COUNT))
    assert nframes_right == nframes_left, 'Left and right videos have different number of frames.'

    print(f'There are {nframes_left} in total.')
    for i in trange(nframes_left):
        success_l, left_image = vidcap_left.read()
        success_r, right_image = vidcap_right.read()

        data = df.iloc[i]
        x = data[x_inds].values
        y = data[y_inds].values
        pixels = np.c_[x,y]
        disp_map = getDisparityMap(left_image, right_image)
        coordinate3D = zed.pixelTo3DCameraCoord(left_image, disp_map, pixels)
        temp_x = [j['X'] for j in coordinate3D]
        temp_y = [j['Y'] for j in coordinate3D]
        temp_z = [j['Z'] for j in coordinate3D]
        df_new.iloc[i][x_inds] = temp_x
        df_new.iloc[i][y_inds] = temp_y
        df_new.iloc[i][z_inds] = temp_z
    if outdir == None:
        outdir = os.path.dirname(os.path.dirname(h5file)) + '/leastereo'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    outname = outdir +  '/markers_trajectory_3d.h5'
    if os.path.isfile(outname):
        os.remove(outname)
    df_new.to_hdf(outname, key = 'markers_trajectory_3d')
    if to_csv:
        df_new.to_csv(outname.replace('.h5', '.csv'))
    return outname

def inverse_triangulate(videos, h5file, to_csv = True):
    vidcap_left = cv2.VideoCapture(videos[0])
    success_l, left_image = vidcap_left.read()
    img_dims = left_image.shape
    df_3D = pd.read_hdf(h5file)
    df_2D = df_3D.copy().drop(['', 'X', 'Y', 'Z', 'z'], axis = 1, level = 1)
    df_2D = zed.Coord3DToPixel(df_3D, df_2D, img_dims)
    outdir = os.path.dirname(h5file)
    outname = outdir +  '/obj_trajectory_2D.h5'
    if os.path.isfile(outname):
        os.remove(outname)
    df_2D.to_hdf(outname, key = 'obj_trajectory_2D')
    if to_csv:
        df_2D.to_csv(outname.replace('.h5', '.csv'))
    print(f'File is saved at {outname}')
    return outname

if __name__ == '__main__':
    basedir = '/home/luke/Desktop/project/Process_data/postprocessed/2022-05-26/'
    videos =  ['/home/luke/Desktop/project/Process_data/convert_reference_frame/Jun13-2022/archive/left/HD1080_SN3404_19-left.mp4', '/home/luke/Desktop/project/Process_data/convert_reference_frame/Jun13-2022/archive/right/HD1080_SN3404_19-right.mp4']
    combined_file = '/home/luke/Desktop/project/Process_data/convert_reference_frame/Jun13-2022/archive/left/markers_trajectory_2d.h5'
    triangulate(videos, combined_file, outdir= '/home/luke/Desktop/project/Process_data/convert_reference_frame/Jun13-2022/archive/left')
    # root, dirs, files = next(os.walk(basedir))
    # for d in dirs:
    #     demo = os.path.join(root, d)
    #     root_d, dirs_d, files_d = next(os.walk(demo))
    #     if 'markers_trajectory_3d.csv' in files_d:
    #         continue
    #     else:
    #         print(f'Processing demo {demo}')
    #         videos = [os.path.join(demo, f) for f in files_d if '.mp4' in f]
    #         combined_file =  [os.path.join(demo, f) for f in files_d if 'markers_trajectory_2d.h5' in f][0]
    #         triangulate(videos, combined_file, outdir = demo)

