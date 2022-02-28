import numpy as np
import pandas as pd


def relative_distance(object_traj, target_traj):
    return np.sqrt(np.sum((object_pos - target_traj[i])**2, axis=1))

class DistanceBasedSelector:
    def __init__(self, window_size, dist_threshold, selection_strategy='vote'):
        self.window_size = window_size
        self.d_threshold = dist_threshold
        self.strategy = selection_strategy
        
    def target_search(self, object_traj, target_traj_set):
        """
        search for potential interactions between current
        moving object and a set of target object. 
        """
        target_dists = {}
        for target_id, target_traj in target_traj_set.items():
            target_dists[target_id] = relative_distance(object_traj, target_traj)
        distance_df = pd.DataFrame(target_dists)
        mean_dist_df =  distance_df.rolling(self.window_size).mean()
        
        if self.strategy=='vote':
            best_votes, best_obj = 0, None
            for col_index, col_name in enumerate(mean_dist_df.columns):
                col_dists = mean_dist_df.iloc[:, col_index]
                col_votes = (col_dists < self.d_threshold).value_counts()[True]
                if best_votes < col_votes:
                    best_votes = col_votes
                    best_obj = col_index
            if best_obj==None:
                return None
            else:
                return mean_dist_df.columns[best_obj]
            
        elif self.stratgy=='closest':
            best_min, best_obj = float('inf'), None
            min_indices =  mean_dist_df.idxmin()
            for col_index, min_index in min_indices:
                col_min = mean_dist_df.iloc[min_index, col_index]
                if best_min > col_min and col_min < self.d_threshold:
                    best_min = col_min
                    best_obj = col_index
            if best_obj==None:
                return None
            else:
                return mean_dist_df.columns[best_obj]
        else: 
            raise
            
class VelocityBasedSelector:
    def __init__(self, window_size, dist_threshold, velo_threshold, selection_strategy='slowest'):
        self.window_size = window_size
        self.d_threshold = dist_threshold
        self.v_threshold = velo_threshold
        self.strategy = selection_strategy
        
    def target_search(self, object_traj, target_traj_set):
        """
        search for potential interactions between current
        moving object and a set of target object. 
        """
        velocity_df = relative_distance(object_traj[:-1], object_traj[1:])
        mean_velocity_df = velocity_df.rolling(self.window_size).mean()
        
        target_dists = {}
        for target_id, target_traj in target_traj_set.items():
            target_dists[target_id] = relative_distance(object_traj, target_traj)
        distance_df = pd.DataFrame(target_dists)
        
        if self.strategy=='vote':
            velo_mask = (mean_velocity_df < self.v_threshold)
            masked_df = distance_df[velo_mask]
            if len(masked_df)==0: 
                return None
            most_votes = masked_df.idxmin(axis=1).value_counts().idxmax()
            return most_votes
            
        elif self.stratgy=='slowest':
            best_velo_min, best_obj = float('inf'), None
            min_index =  mean_velocity_df.idxmin()
            for k, mean_velo in enumerate(mean_velocity_df):
                if best_velo_min > mean_velo and mean_velo < self.v_threshold:
                    best_velo_min = mean_velo
                    best_dist_min = float('inf')
                    for col in distance_df.columns:
                        col_min = distance_df.iloc[k+1, col]
                        if best_dist_min > col_min and col_min < self.d_threshold:
                            best_dist_min = col_min
                            best_obj = col_min
            if best_obj==None:
                return None
            else:
                return mean_dist_df.columns[best_obj]
        else: 
            raise