import numpy as np
import pandas as pd
from math import cos, sin, acos

def angle_diff(r0, r1):
    # should
    nr0 = np.linalg.norm(r0)
    nr1 = np.linalg.norm(r1)
    cs = cos(nr0/2)*cos(nr1/2) 
    sn = sin(nr0/2)*sin(nr1/2)*np.dot(r0,r1)/(nr1*nr0)
    return np.abs(2*acos(cs+sn))

class KeypointSelector:
    def __init__(self, vel_threshold_fast, vel_threshold_slow, rot_threshold_fast, rot_threshold_slow, rotation_max,
                 velocity_window_size=10, rotation_window_size=5, min_time_sep=0.25):
        self.fast_spd = False
        self.fast_rot = False
        self.vel_th_fast = vel_threshold_fast
        self.vel_th_slow = vel_threshold_slow
        self.rot_th_fast = rot_threshold_fast
        self.rot_th_slow = rot_threshold_slow
        self.rot_max = rotation_max
        # set window size.
        self.vel_wsize = velocity_window_size
        self.rot_wsize = rotation_window_size
        self.min_time_sep = min_time_sep
    def search(self, object_traj):
        """
        returns indices of the frame when the keypoints are detected 
        according to the heuristics.
        """
        result_df = object_traj.copy()
        result_df['condition'] = None
        result_df['theta'] = None
        # object velocity 
        vel_df = np.sqrt(np.square(pd.DataFrame(object_traj.loc[:, ['x','y','z']]).diff(1)).sum(axis=1, skipna=False))
        vel_mean = vel_df.rolling(self.vel_wsize, center=True).mean()
        
        # object rotation 
        for i in range(1, len(vel_mean)):
            r0 = object_traj.iloc[i-1][['Rx', 'Ry', 'Rz']]
            r1 = object_traj.iloc[i][['Rx', 'Ry', 'Rz']]
            result_df['theta'].iloc[i] = angle_diff(r0, r1)
        rot_mean = result_df['theta'].rolling(self.rot_wsize, center=True).mean()
        
        keypoint_indices = [0]
        result_df['condition'].iloc[0] = 'start'
        for i in range(1, len(vel_mean)):
            prev_key_pt = keypoint_indices[-1]
#             if self.min_time_sep > abs(object_traj['Time'].iloc[prev_key_pt] - object_traj['Time'].iloc[i]): continue
            # find velocity keypoints
            if not vel_mean[i-1:i+1].isnull().values.any():
                if (vel_mean[i-1] < self.vel_th_fast and self.vel_th_fast < vel_mean[i] and 
                    not self.fast_spd):
                    keypoint_indices.append(i)
                    result_df['condition'].iloc[i] = 'velocity'
                    self.fast_spd = True
                elif (vel_mean[i-1] > self.vel_th_slow and self.vel_th_slow > vel_mean[i] and
                    self.fast_spd):
                    keypoint_indices.append(i)
                    result_df['condition'].iloc[i] = 'velocity'
                    self.fast_spd = False
                    
            # find rotational keypoints
            if not rot_mean[i-1:i+1].isnull().values.any():
                if (rot_mean[i] > self.rot_th_fast and self.rot_th_fast > rot_mean[i-1] and 
                   not self.fast_rot):
                    keypoint_indices.append(i)
                    result_df['condition'].iloc[i] = 'rotation'
                    self.fast_rot = True
                elif (rot_mean[i] < self.rot_th_slow and self.rot_th_slow < rot_mean[i-1] and
                    self.fast_rot):
                    keypoint_indices.append(i)
                    result_df['condition'].iloc[i] = 'rotation'
                    self.fast_rot = False

#             if not rot_df_sm.iloc[i-1:i+1, 0].isnull().values.any():   
#                 if not ((rot_df_sm.iloc[i-1, 0] > self.rot_threshold) ^ 
#                         (self.rot_threshold > rot_df_sm.iloc[i, 0])) or abs(object_traj['Rx'].iloc[i-1] - object_traj['Rx'].iloc[prev_key_pt]) > self.rot_max:
#                     keypoint_indices.append(i)
#                     result_df['condition'].iloc[i] = 'rotation'
                    
#             if not rot_df_sm.iloc[i-1:i+1, 1].isnull().values.any():   
#                 if not ((rot_df_sm.iloc[i-1, 1] > self.rot_threshold) ^ 
#                         (self.rot_threshold > rot_df_sm.iloc[i, 1])) or abs(object_traj['Ry'].iloc[i-1] - object_traj['Ry'].iloc[prev_key_pt]) > self.rot_max:
#                     keypoint_indices.append(i)  
#                     result_df['condition'].iloc[i] = 'rotation'
                    
#             if not rot_df_sm.iloc[i-1:i+1, 2].isnull().values.any():   
#                 if not ((rot_df_sm.iloc[i-1, 2] > self.rot_threshold) ^ 
#                         (self.rot_threshold > rot_df_sm.iloc[i, 2])) or abs(object_traj['Rz'].iloc[i-1] - object_traj['Rz'].iloc[prev_key_pt]) > self.rot_max:
#                     keypoint_indices.append(i)
#                     result_df['condition'].iloc[i] = 'rotation'
            if object_traj['action'].iloc[i] != None:
                keypoint_indices.append(i)
                result_df['condition'].iloc[i] = 'grasp'
                
        return result_df, sorted(list(set(keypoint_indices))), vel_mean, rot_mean