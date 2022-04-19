import numpy as np
import pandas as pd


class KeypointSelector:
    def __init__(self, velocity_threshold, rotation_threshold, rotation_max,
                 velocity_window_size=10, rotation_window_size=5):
        self.vel_threshold = velocity_threshold
        self.rot_threshold = rotation_threshold
        self.rot_max = rotation_max
        # set window size.
        self.vel_wsize = velocity_window_size
        self.rot_wsize = rotation_window_size
        
    def search(self, object_traj):
        """
        returns indices of the frame when the keypoints are detected 
        according to the heuristics.
        """
        result_df = object_traj.copy()
        result_df['condition'] = None
        # object velocity 
        velocity_df = np.sqrt(np.square(pd.DataFrame(object_traj.loc[:, ['x','y','z']]).diff(1)).sum(axis=1, skipna=False))
        velocity_df_filtered = velocity_df.rolling(self.vel_wsize, center=True).mean()
        # object rotations 
        rotation_df = pd.DataFrame(object_traj.loc[:, ['Rx','Ry','Rz']]).diff(1).abs()
        rotation_df_filtered = rotation_df.rolling(self.rot_wsize, center=True).mean()
        keypoint_indices = [0]
        result_df.iloc[0, -1] = 'start'
        for i in range(1, len(velocity_df_filtered)):
            prev_pt = keypoint_indices[-1]
            # find velocity keypoints
            if not velocity_df_filtered.iloc[i-1:i+1].isnull().values.any():
                if not ((velocity_df_filtered.iloc[i-1] > self.vel_threshold) ^ 
                        (self.vel_threshold > velocity_df_filtered.iloc[i])):
                    keypoint_indices.append(i)
                    result_df.iloc[i, -1] = 'velocity'
         
            # find rotational keypoints
            if not rotation_df_filtered.iloc[i-1:i+1, 0].isnull().values.any():   
                if not ((rotation_df_filtered.iloc[i-1, 0] > self.rot_threshold) ^ 
                        (self.rot_threshold > rotation_df_filtered.iloc[i, 0])) or abs(rotation_df_filtered.iloc[i-1, 0] - rotation_df_filtered.iloc[prev_pt, 0]) > self.rot_max:
                    keypoint_indices.append(i)
                    result_df.iloc[i, -1] = 'rotation'
                    
            if not rotation_df_filtered.iloc[i-1:i+1, 1].isnull().values.any():   
                if not ((rotation_df_filtered.iloc[i-1, 1] > self.rot_threshold) ^ 
                        (self.rot_threshold > rotation_df_filtered.iloc[i, 1])) or abs(rotation_df_filtered.iloc[i-1, 1] - rotation_df_filtered.iloc[prev_pt, 1]) > self.rot_max:
                    keypoint_indices.append(i)  
                    result_df.iloc[i, -1] = 'rotation'
                    
            if not rotation_df_filtered.iloc[i-1:i+1, 2].isnull().values.any():   
                if not ((rotation_df_filtered.iloc[i-1, 2] > self.rot_threshold) ^ 
                        (self.rot_threshold > rotation_df_filtered.iloc[i, 2])) or abs(rotation_df_filtered.iloc[i-1, 2] - rotation_df_filtered.iloc[prev_pt, 2]) > self.rot_max:
                    keypoint_indices.append(i)
                    result_df.iloc[i, -1] = 'rotation'
                    
            if object_traj.iloc[i, -1] != None:
                keypoint_indices.append(i)
                result_df.iloc[i, -1] = 'grasp'
                
        return result_df, sorted(list(set(keypoint_indices))), velocity_df_filtered, rotation_df_filtered