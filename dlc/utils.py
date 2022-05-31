from glob import glob
import cv2
import os

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



def get_h5files(dirname, filtertype = None):
    h5files = glob(os.path.join(dirname, '*.h5'))
    return h5files

def video_to_frames(video_path):

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    frames = []
    while success:
      frames.append(image)
      success,image = vidcap.read()
    print(f'The number of frames is {len(frames)}')
    return frames
