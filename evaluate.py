from glob import glob
import deeplabcut
import post_processing, utils, visualization
import os
import shutil

def batch_evaluate():
    objs = ['teabag', 'cup', 'pitcher', 'tap']
    for obj in objs:
        config_path = glob(f'/home/luke/Desktop/project/make_tea/dlc/make_tea_{obj}*/config.yaml')[0]
        deeplabcut.evaluate_network(config_path, plotting=True)

def batch_analyze_video(vid_id):
    videos = util.get_videos(vid_id)
    objs = ['pitcher']
    for obj in objs:
        for video in videos:
            vid_name = os.path.basename(video)
            parent_dir = os.path.dirname(video)
            destdir = parent_dir + f'/{obj}'
            if not os.path.isdir(destdir):
                os.mkdir(destdir)
            dest_video = destdir + f'/{vid_name}'
            shutil.copy(video, dest_video)
            config_path = glob(f'/home/luke/Desktop/project/make_tea/dlc/make_tea_{obj}*/config.yaml')[0]
            scorername = deeplabcut.analyze_videos(config_path, video, videotype='.mp4', destfolder = destdir, auto_track = True)
            # h5file = glob(destdir + '/*.h5')[0]
            # visualization.create_video_with_h5file(config_path, videos, h5file)
            deeplabcut.create_video_with_all_detections(config_path, dest_video, videotype = '.mp4')

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

def batch_interpolate(vid_id, filtertype="median", windowlength=5, ARdegree=3, MAdegree=1,):
    objs = ['pitcher']
    for obj in objs:
        config_path = glob(f'/home/luke/Desktop/project/make_tea/dlc/make_tea_{obj}*/config.yaml')[0]
        videos = [f'/home/luke/Desktop/project/make_tea/camera-main/videos/{vid_id}/left/{obj}/{vid_id}-left.mp4',
                  f'/home/luke/Desktop/project/make_tea/camera-main/videos/{vid_id}/right/{obj}/{vid_id}-right.mp4']
        outputnames = post_processing.interpolate_data(config_path, videos = videos, filtertypes= filtertype, windowlengths=windowlength,
            ARdegree=3, MAdegree=1,)
