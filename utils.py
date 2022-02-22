from glob import glob
import cv2

def get_videos(vid_id, obj = None, kernel = None, base_dir = '/home/luke/Desktop/project/make_tea'):
    '''
    This function output video path given the video ID.

    vid_id: The video ID
    obj: The object folder the video is sit in. Options: 'teabag', 'cup', 'pitcher', 'tap'. If None, video in one level higher will be outputed.
    kernel: Given kernel, this function output the video filtered by the kernel. If None, the original non-filtered vildeo path will be outputed.
    base_dir: The path to the make_tea folder.
    '''
    if kernel == None:
        if obj == None:
            videos = [f'{base_dir}/videos/{vid_id}/left/{vid_id}-left.mp4',
              f'{base_dir}/camera-main/videos/{vid_id}/right/{vid_id}-right.mp4']
        else:
            videos = [f'{base_dir}/camera-main/videos/{vid_id}/left/{obj}/{vid_id}-left.mp4',
              f'{base_dir}/camera-main/videos/{vid_id}/right/{obj}/{vid_id}-right.mp4']
    else:
        if not isinstance(kernel, list):
            suffix = f'_{kernel}'
        else:
            temp = '_'.join(kernel)
            suffix = f'_{temp}'
        if obj == None:
            videos = [f'{base_dir}/camera-main/videos/{vid_id}/left/{vid_id}-left{suffix}.mp4',
              f'{base_dir}/camera-main/videos/{vid_id}/right/{vid_id}-right{suffix}.mp4']
        else:
            videos = [f'{base_dir}/camera-main/videos/{vid_id}/left/{obj}/{vid_id}-left{suffix}.mp4',
              f'{base_dir}/make_tea/camera-main/videos/{vid_id}/right/{obj}/{vid_id}-right{suffix}.mp4']
    return videos



def get_h5files(vid_id, kernel):
    if kernel == None:
        h5files = glob(f'/home/luke/Desktop/project/make_tea/camera-main/videos/{vid_id}/left/{vid_id}-leftDLC*.h5')+\
        glob(f'/home/luke/Desktop/project/make_tea/camera-main/videos/{vid_id}/left/{vid_id}-leftDLC*.h5')
    else:
        if not isinstance(kernel, list):
                suffix = f'_{kernel}'
        else:
            temp = '_'.join(kernel)
            suffix = f'_{temp}'
        h5files = [f'/home/luke/Desktop/project/make_tea/camera-main/videos/{vid_id}/left/{vid_id}-left{suffix}.h5',
          f'/home/luke/Desktop/project/make_tea/camera-main/videos/{vid_id}/right/{vid_id}-right{suffix}.h5']
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
