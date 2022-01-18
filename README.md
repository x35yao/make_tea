## make_tea

- Modify source code for 3D project:

  - Replace the file `anaconda3/envs/DEEPLABCUT/lib/python3.8/site-packges/deeplabcut/pose_estimation_3d/plotting3D.py` with `dlc/plotting3D.py `

  - Replace the file `anaconda3/envs/DEEPLABCUT/lib/python3.8/site-packges/deeplabcut/pose_estimation_3d/triangulation.py` with `dlc/triangulation.py `

  - Replace the file `anaconda3/envs/DEEPLABCUT/lib/python3.8/site-packges/deeplabcut/utils/auxiliaryfunctions_3d.py` with `dlc/auxiliaryfunctions_3d.py `

- Debugged why dlc failed to ouput tracklets:
  - First, set min_n_links = 1 in Class Assembler in file `/home/luke/anaconda3/envs/DEEPLABCUT/lib/python3.8/site-packages/deeplabcut/pose_estimation_tensorflow/lib/inferenceutils.py`
  - Then, Use 'box' instead of 'ellipse' when convert_detections2tracklets because using 'ellipse' will end up with no trackers in line 1488 of file predict_video.py. This could be done by add `default_track_method: 'box'` in the config file.
