import os
import pandas as pd
raw_dir = '/home/luke/Desktop/project/Process_data/postprocessed/2022-05-26/'
root, dirs, files = next(os.walk(raw_dir))
ind = 0
demo = os.path.join(root, dirs[ind])

leastereo_3d = os.path.join(demo, 'leastereo', 'markers_trajectory_3d.csv')
dlc_3d = os.path.join(demo, 'dlc3d', 'markers_trajectory_3d.csv')

df_leastereo = pd.read_csv(leastereo_3d)
df_dlc = pd.read_csv(dlc_3d)

print(df_dlc)
