import numpy as np
import pandas as pd


class KeypointSelector:
    def __init__(self, velocity_threshold, rotation_threshold, 
                 velocity_window_size=10, rotation_window_size=5):
        self.vel_threshold = vel_threshold
        self.rot_threshold = rot_threshold
        # set window size.
        self.vel_wsize = velocity_window_size
        self.rot_wsize = rotation_window_size
        
    def search(self, object_traj, object_rot):
        """
        returns indices of the frame when the keypoints are detected 
        according to the heuristics.
        """
        # object velocity 
        velocity_df = np.sqrt(np.square(pd.DataFrame(object_traj).diff(1)).sum(axis=1, skipna=False))
        velocity_df_filtered = velocity_df.rolling(self.vel_wsize).mean()
        # object rotations 
        rotation_df = pd.DataFrame(object_rot).diff(1)
        rotation_df_filtered = rotation_df.rolling(self.rot_wsize).mean()
        keypoint_indices = []
        for i in range(1, len(velocity_df_filtered)):
            # find velocity keypoints
            if not velocity_df_filtered.iloc[i-1:i+1].isnull().values.any():
                if not ((velocity_df_filtered.iloc[i-1] > self.vel_threshold) ^ 
                        (self.vel_threshold > velocity_df_filtered.iloc[i])):
                    keypoint_indices.append(i)
            
            # find rotational keypoints
            if not rotation_df_filtered.iloc[i-1:i+1, 0].isnull().values.any():   
                if not ((rotation_df_filtered.iloc[i-1, 0] > self.rot_threshold) ^ 
                        (self.rot_threshold > rotation_df_filtered.iloc[i, 0])):
                    keypoint_indices.append(i)  
                    
            if not rotation_df_filtered.iloc[i-1:i+1, 1].isnull().values.any():   
                if not ((rotation_df_filtered.iloc[i-1, 1] > self.rot_threshold) ^ 
                        (self.rot_threshold > rotation_df_filtered.iloc[i, 1])):
                    keypoint_indices.append(i)  
                    
            if not rotation_df_filtered.iloc[i-1:i+1, 2].isnull().values.any():   
                if not ((rotation_df_filtered.iloc[i-1, 2] > self.rot_threshold) ^ 
                        (self.rot_threshold > rotation_df_filtered.iloc[i, 2])):
                    keypoint_indices.append(i)  
        return keypoint_indices