from glob import glob
import deeplabcut
from .post_processing import interpolate_data, filter_3D_data
from .utils import get_videos, get_h5files
from .visualization import create_video_with_h5file
from deeplabcut.post_processing import filtering
import os
import shutil
from deeplabcut.utils.auxfun_videos import VideoReader
import pandas as pd


def analyze_video(config, vid,shuffle = 1,  make_video = True, filterpredictions = True, filtertype = 'median', destfolder = None):
    if destfolder == None:
        destfolder = os.path.dirname(vid)
    scorername = deeplabcut.analyze_videos(config_path, vid, videotype='.mp4',auto_track = True, robust_nframes = True, save_as_csv = True, shuffle = shuffle, destfolder = destfolder, n_tracks = n_tracks)
    if filterpredictions:
        filtering.filterpredictions(
            config_path,
            [vid],
            videotype='mp4',
            shuffle=shuffle,
            filtertype=filtertype,
            destfolder=obj_dir,
        )
    if make_video:
        h5files = glob(destfolder + '/*.h5')
        if filterpredictions:
            h5file = [f for f in h5files if 'filtered' in f ][0]
        else:
            h5file = [f for f in h5files if 'filtered' not in f ][0]
        try:
            create_video_with_h5file(vid, h5file)
        except IndexError:
            print(f'Deeplabcut fails to detect the object{obj}!')



def batch_get_tracklets(videos):
    objs = ['teabag', 'cup', 'pitcher', 'tap']
    for obj in objs:
        config_path = glob(f'/home/luke/Desktop/project/make_tea/dlc/make_tea_{obj}*/config.yaml')[0]
        deeplabcut.convert_detections2tracklets(config_path, videos, videotype='mp4',
                                        shuffle=1, trainingsetindex=0)
        deeplabcut.stitch_tracklets(config_path, ['videofile_path'], videotype='mp4',
                            shuffle=1, trainingsetindex=0)


def serch_obj_h5files(target_dir, objs, filtering = True):
    h5files = []
    for obj in objs:
        obj_dir = os.path.join(target_dir, obj)
        h5files_obj = glob(os.path.join(obj_dir, '*.h5'))
        if filtering:
            h5file_obj = [f for f in h5files_obj if 'filtered' in f][0]
        else :
            h5file_obj = [f for f in h5files_obj if 'filtered' not in f][0]
        h5files.append(h5file_obj)
    return h5files

def combine_h5files(h5files, to_csv = True, destdir = None, suffix = '2d'):
    df_new = pd.DataFrame()
    for h5file in h5files:
        df = pd.read_hdf(h5file)
        df_new = pd.concat([df_new, df], axis = 1)
    if destdir == None:
        destdir = os.path.dirname(os.path.dirname(h5file))
    else:
        if not os.path.isdir(destdir):
            os.makedirs(destdir)
    outputname = destdir + '/' + f'markers_trajectory_{suffix}.h5'
    if os.path.isfile(outputname):
        print('Removing existing file.')
        os.remove(outputname)
    df_new.to_hdf(outputname, mode = 'w', key = f'markers_trajectory_{suffix}')
    if to_csv:
        df_new.to_csv(outputname.replace('.h5', '.csv'))
    print(f'The file is saved at {outputname}')
    return outputname
