import numpy as np
import pandas as pd


def relative_distance(object_traj, target_traj):
    return np.sqrt(np.sum((np.array(object_traj) - np.array(target_traj))**2, axis=1))

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
        
        mask = (mean_dist_df < self.d_threshold)
        if self.strategy=='vote':
            # Pick object that is closest to the moving object for most of the sliding windows
            # within the MA threshold.
            best_votes, best_obj = 0, None
            for col_index, col_name in enumerate(mean_dist_df.columns):
                obj_dists = mean_dist_df.iloc[:, col_index]
                try:
                    true_votes = (obj_dists < self.d_threshold).value_counts()[True]
                except:
                    continue
                if best_votes < true_votes:
                    best_votes = true_votes
                    best_obj = col_index
            if best_obj==None:
                return None
            else:
                return mean_dist_df.columns[best_obj]
            
        elif self.strategy=='closest':
            # Pick object that is closest to the moving object for whole trajectory if
            # MA threshold is reached.
            valid_dist_df = distance_df[mask]
            min_index = valid_dist_df.min(axis=0)
            min_column = min_index.idxmin()
            if min_index.min() > self.d_threshold:
                return None
            else: return min_column
        else: 
            return None
            
class VelocityBasedSelector:
    def __init__(self, window_size, dist_threshold, velo_threshold, selection_strategy='closest'):
        self.window_size = window_size
        self.d_threshold = dist_threshold
        self.v_threshold = velo_threshold
        self.strategy = selection_strategy
        
    def target_search(self, object_traj, target_traj_set):
        """
        search for potential interactions between current
        moving object and a set of target object. 
        """
        velocity_df = np.sqrt(np.square(pd.DataFrame(object_traj).diff(1)).sum(axis=1, skipna=False))
        mean_velocity_df = velocity_df.rolling(self.window_size).mean()
       
        target_dists = {}
        for target_id, target_traj in target_traj_set.items():
            target_dists[target_id] = relative_distance(object_traj, target_traj)
        distance_df = pd.DataFrame(target_dists)
        
        velo_mask = (mean_velocity_df < self.v_threshold)
        if self.strategy=='vote':
            # Pick object that is closest to the moving object for most of the sliding windows
            # within the MA velocity threshold.
            masked_df = distance_df[velo_mask]
            masked_df = masked_df[masked_df.min(axis=1) < self.d_threshold]
            if len(masked_df)==0: return None
            most_votes = masked_df.idxmin(axis=1).value_counts().idxmax()
            return most_votes
            
        elif self.strategy=='slowest':
            # Pick object that is closest to the moving object for whole trajectory during
            # the window with lowest velocity.
            min_index =  mean_velocity_df[velo_mask].idxmin()
            best_col = distance_df.iloc[min_index].idxmin()
            min_val = distance_df[best_col].iloc[min_index]
#             print(mean_velocity_df, distance_df, min_index, velo_mask)
            if min_val > self.d_threshold:
                return None
            else: return best_col
        elif self.strategy=='closest':
            # Pick object that is closest to the moving object for whole trajectory if
            # MA velocity threshold is reached.
            valid_dist_df = distance_df[velo_mask]
            min_index = valid_dist_df.min(axis=0)
            min_column = min_index.idxmin()
            if min_index.min() > self.d_threshold:
                return None
            else: return min_column
        else: 
            return None