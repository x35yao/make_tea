from glob import glob
import cv2

def get_videos(vid_id, kernel = None):
    if kernel == None:
        videos = [f'/home/luke/Desktop/project/make_tea/camera-main/videos/{vid_id}/left/{vid_id}-left.mp4',
          f'/home/luke/Desktop/project/make_tea/camera-main/videos/{vid_id}/right/{vid_id}-right.mp4']
    else:
        if not isinstance(kernel, list):
            suffix = f'_{kernel}'
        else:
            temp = '_'.join(kernel)
            suffix = f'_{temp}'
        videos = [f'/home/luke/Desktop/project/make_tea/camera-main/videos/{vid_id}/left/{vid_id}-left{suffix}.mp4',
          f'/home/luke/Desktop/project/make_tea/camera-main/videos/{vid_id}/right/{vid_id}-right{suffix}.mp4']
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
    
