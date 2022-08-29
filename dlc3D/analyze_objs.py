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
        print('Removing exsited file.')
        os.remove(outputname)
    df_new.to_hdf(outputname, mode = 'w', key = f'markers_trajectory_{suffix}')
    if to_csv:
        df_new.to_csv(outputname.replace('.h5', '.csv'))
    print(f'The file is saved at {outputname}')
    return outputname

def analyze_objs_video(data_dir, objs, filterpredictions, filtertype, videotype = 'mp4', make_video = True):

    data_root, demo_dirs, data_files = next(os.walk(data_dir))

    DLC3D = '/home/luke/Desktop/project/make_tea/dlc3D'
    dlc_root, dlc_dirs, dlc_files = next(os.walk(DLC3D))
    dlc3d_scorers = [obj + '_3d' for obj in objs]
    ### Analyze videos to get detections
    for d in dlc_dirs:
        if d in dlc3d_scorers:
            if 'teabag' in d:
                n_tracks = 2
            else:
                n_tracks = 1
            scorer_dir = os.path.join(dlc_root, d)
            config3d = glob(scorer_dir + '/config.yaml')[0]
            cfg = auxiliaryfunctions.read_config(config3d)

            for demo in demo_dirs:
                demo_root, demo_dir, files = next(os.walk(os.path.join(data_root, demo)))
                files.sort()
                root_left, dir_left, files_left = next(os.walk(os.path.join(data_root, demo, 'left')))
                vid_left = os.path.join(root_left, [f for f in files_left if '.mp4' in f][0])
                root_right, dir_right, files_right = next(os.walk(os.path.join(data_root, demo, 'right')))
                vid_right = os.path.join(root_right, [f for f in files_right if '.mp4' in f][0])
                print(f'Analyzing video: {demo} for object {d}')
                h5_left_and_right = []
                for vid in [vid_left, vid_right]:
                    if 'left' in vid:
                        config2d = cfg['config_file_left']
                    elif 'right' in vid:
                        config2d = cfg['config_file_right']
                    outdir = os.path.join(os.path.dirname(vid), d)
                    if not os.path.isdir(outdir):
                        os.makedirs(outdir)
                    scorername = deeplabcut.analyze_videos(config2d, vid, videotype= videotype, auto_track=True,
                                                           robust_nframes=True, save_as_csv=True,
                                                           destfolder=outdir, n_tracks=n_tracks)

                    if filterpredictions:
                        filtering.filterpredictions(
                            config2d,
                            [vid],
                            videotype='mp4',
                            filtertype=filtertype,
                            destfolder=outdir,
                        )
                        h5file = glob(outdir + '/*_filtered.h5')[0]
                    else:
                        h5files = glob(outdir + '/*.h5')
                        h5file = [f for f in h5files if 'filtered' not in f][0]
                    h5_left_and_right.append(h5file)
                    if make_video:
                        create_video_with_h5file(vid, h5file)
                destfolder = os.path.join(demo_root, 'dlc3d', d)
                triangulate(config3d, h5_left_and_right[0], h5_left_and_right[1], destfolder)
            raise



if __name__ == '__main__':
    data_dir = '/home/luke/Desktop/project/make_tea/Process_data/postprocessed/2022-08-(17-21)'
    filterpredictions = True
    filtertype = 'median'
    make_video = True
    objs = ['teabag', 'pitcher', 'cup', 'tap']
    analyze_objs_video(data_dir, objs,  filterpredictions, filtertype)
    # Combined analyzed data from different objects
    # for d in demo_dirs:
    #     h5files = []
    #     for obj in objs:
    #         h5_obj = glob(os.path.join(data_root, d, obj + '_3d', '*3d.h5'))[0]
    #         h5files.append(h5_obj)
    #     outdir = os.path.join(data_root, d, 'dlc3d')
    #     combine_h5files(h5files, destdir = outdir, suffix = '3d')