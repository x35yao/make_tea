from glob import glob
import deeplabcut
import post_processing, utils, visualization
import os
import shutil
from deeplabcut.utils.auxfun_videos import VideoReader

def batch_evaluate():
    objs = ['teabag', 'cup', 'pitcher', 'tap']
    for obj in objs:
        config_path = glob(f'/home/luke/Desktop/project/make_tea/dlc/make_tea_{obj}*/config.yaml')[0]
        deeplabcut.evaluate_network(config_path, plotting=True)

def batch_analyze_video(vid_id, objs = ['pitcher', 'tap', 'teabag', 'cup'] ):
    for obj in objs:
        config_path = glob(f'/home/luke/Desktop/project/make_tea/dlc/make_tea_{obj}*/config.yaml')[0]
        videos = utils.get_videos(vid_id, obj)
        for video in videos:
            obj_dir = os.path.dirname(video)
            if not os.path.isdir(obj_dir):
                os.makedirs(obj_dir)
            if not os.path.isfile(video):
                filename = os.path.basename(video)
                folder = os.path.dirname(os.path.dirname(os.path.dirname(video)))
                src = folder + '/' + filename
                dest = os.path.dirname(video)
                shutil.copyfile(src, video)
            scorername = deeplabcut.analyze_videos(config_path, video, videotype='.mp4', auto_track = True, robust_nframes = True, save_as_csv = True)

            deeplabcut.create_video_with_all_detections(config_path, video, videotype = '.mp4')

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

def batch_interpolate(vid_id, objs = ['pitcher', 'tap', 'teabag', 'cup'], filtertype="median", windowlength=5, ARdegree=3, MAdegree=1, create_video = True):
    for obj in objs:
        config_path = glob(f'/home/luke/Desktop/project/make_tea/dlc/make_tea_{obj}*/config.yaml')[0]
        videos = [f'/home/luke/Desktop/project/make_tea/camera-main/videos/{vid_id}/left/{obj}/{vid_id}-left.mp4',
                  f'/home/luke/Desktop/project/make_tea/camera-main/videos/{vid_id}/right/{obj}/{vid_id}-right.mp4']
        outputnames = post_processing.interpolate_data(config_path, videos = videos, filtertypes= filtertype, windowlengths=windowlength,
            ARdegree=3, MAdegree=1,)
        if create_video:
            for i, video in enumerate(videos):
                visualization.create_video_with_h5file(config_path, video, outputnames[i])
