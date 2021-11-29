import deeplabcut

video_id = '1636318271' # modify this value here
video_path = '/home/luke/Desktop/project/make_tea/camera-main/videos/' + video_id
config3d = '/home/luke/Desktop/project/make_tea/dlc/make_tea_3d-Luke-2021-11-22-3d/config.yaml'
deeplabcut.triangulate(config3d, video_path, videotype = '.mp4', filterpredictions=True, save_as_csv=True)
deeplabcut.create_labeled_video_3d(config3d, [video_path], videofolder = video_path, videotype = '.mp4', view = [-80, -90], start=100, end=200)
# deeplabcut.create_labeled_video_3d(config3d, [video_path], videofolder = video_path, videotype = '.mp4', view = [-80, -90], start=100, end=200, xlim = [-50,20], ylim = [-30,30], zlim = [-10,100])
