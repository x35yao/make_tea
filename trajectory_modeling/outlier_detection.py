import numpy as np
import pandas as pd
    
    
def detect_outlier(trajs, n, dims=['x', 'y', 'z']):
    """
    takes in a set of trajectory demos and returns a list of the outlier.
    Parameters:
    -----------
    trajs: list
        A list of trajectories containing the positional features.
    n: float 
        nth std threshold
    dims: list 
        a list of name of the features to be considered
    Returns:
    --------
    outliers: list
        A list of index of the trajectory that are outliers
    """
    outliers = []
    keys = sorted(list(trajs.keys()))
    while True:
        cur_outliers = []
        end_points, start_points = [], []
        # gather start and end points
        for i_key in keys:
            start_points.append(trajs[i_key][dims].iloc[0])
            end_points.append(trajs[i_key][dims].iloc[-1])
        start_df = pd.DataFrame(start_points)
        end_df = pd.DataFrame(end_points)

        # generate the means and std
        means_start = {col: start_df[col].mean() for col in dims}
        stds_start = {col: start_df[col].std() for col in dims}
        means_end = {col: end_df[col].mean() for col in dims}
        stds_end = {col: end_df[col].std() for col in dims}
        # collect the outlier demos.
        for i_key in keys:
            start_pt = trajs[i_key].iloc[0]
            end_pt = trajs[i_key].iloc[-1]
            for dim in dims:
                z1 = np.abs(start_pt[dim] - means_start[dim])/stds_start[dim]
                z2 = np.abs(end_pt[dim] - means_end[dim])/stds_end[dim]
                if z1 > n or z2 > n:
                    cur_outliers.append(i_key)
                    break
        if len(cur_outliers) < 1:
            break
        else:
            outliers = outliers + cur_outliers
            keys = [k for k in keys if k not in outliers]

    return outliers
    
def detect_lowest_var_outlier(trajs, n, steps=0, dims=['x', 'y', 'z']):
    """
    takes in a set of realigned trajectory demos with same number of time steps
    and returns a list of the outlier by iteratively searching points at the time step 
    with the lowest maximum dimension variance.
    
    Parameters:
    -----------
    trajs: list
        A list of trajectories containing the positional features
    n: float 
        nth std threshold
    steps: int
        Time steps to be considered for selecting outliers. 
        By default selects only the starting time and ending time.
    dims: list 
        a list of name of the features to be considered
    Returns:
    --------
    outliers: list
        A list of index of the trajectory that are outliers
    """
    outliers = []
    keys = sorted(list(trajs.keys()))
     
    # confirm trajectories are all of the same length
    traj_lens = set([len(trajs[d]) for d in trajs.keys()])
    max_steps = max(traj_lens)
    if len(traj_lens) > 1: 
        raise Exception("The trajectories does not all have the same length")
    
    # select timesteps to be examined
    search_timesteps = [0]
    if steps > 0: search_timesteps = list(range(0, max_steps-1, steps))
    search_timesteps.append(max_steps-1)
    while True:
        cur_outliers = []
        # select the timestep with the most consistent points(lowest variance across all dimension)
        max_var_timesteps = []
        for t in search_timesteps:
            timestep_pts = []
            for i_key in keys:
                timestep_pts.append(trajs[i_key][dims].iloc[t])
            timestep_pts = np.array(timestep_pts)
            # select max variance across all dims
            max_dim_var = np.max(np.std(timestep_pts, axis=0))
            max_var_timesteps.append(max_dim_var)
        # fetch index of the minimum overall variance 
        best_t = search_timesteps[max_var_timesteps.index(min(max_var_timesteps))]
#         print(f"This iteration's best timestamp: {best_t}")
        
        selected_pts = []
        # gather start and end points
        for i_key in keys:
            selected_pts.append(trajs[i_key][dims].iloc[best_t])
        selected_pts_df = pd.DataFrame(selected_pts)

        # generate the means and std
        means_pts = {col: selected_pts_df[col].mean() for col in dims}
        stds_pts = {col: selected_pts_df[col].std() for col in dims}
        # collect the outlier demos.
        for i_key in keys:
            test_pt = trajs[i_key].iloc[best_t]
            for dim in dims:
                z1 = np.abs(test_pt[dim] - means_pts[dim])/stds_pts[dim]
                if z1 > n:
                    cur_outliers.append(i_key)
                    break
        if len(cur_outliers) < 1:
            break
        else:
            outliers = outliers + cur_outliers
#             print(f"This iteration's outliers: {cur_outliers}")
            keys = [k for k in keys if k not in outliers]
            
    return outliers
    

# def traj_summarization(traj_df):
#     """
#     generate the aggragate features from the given trajectory.
#     Parameters:
#     -----------
#     traj_df: Dataframe
#         the raw data of the second trajectory to be summarized
#     Returns:
#     --------
#     feature_vector: list of float 
#         the summarized feature of the trajectory
#     """
#     feature_vector = []
#     # get start and end points
#     start_vector = traj_df.iloc[0][['x', 'y', 'z']]
#     end_vector = traj_df.iloc[-1][['x', 'y', 'z']]
#     feature_vector.append(start_vector.tolist())
#     feature_vector.append(end_vector.tolist())
#     # get max/min and average heading of the trajectory
#     displacement = traj_df[['x', 'y', 'z']].diff(1)
#     max_heading = displacement.max()
#     min_heading = displacement.min()
#     ave_heading = displacement.mean()
#     feature_vector.append(max_heading.tolist() + min_heading.tolist() + ave_heading.tolist())
    
#     # derive the average, max and min speed of the trajectory
#     temp = ((np.sqrt(np.square(traj_df.loc[:, ['x', 'y', 'z']].diff(1)).sum(axis=1))))
#     speed = np.array(temp)
#     feature_vector.append([speed.max(), speed.min(), speed.mean()])
#     return feature_vector
   
    
# def traj_weighted_dist(traj1, traj2, weights):
#     """
#     calculate the distance between two given trajectories 
#     Parameters:
#     -----------
#     traj1: Dataframe
#         the raw data of the first trajectory
#     traj2: Dataframe
#         the raw data of the second trajectory
#     weights: list of float
#         weights for the different feature comparisons
#     Returns:
#     --------
#     weighted_distance: float 
#         the weighted distance between the two points
#     """
#     traj1_feat = traj_summarization(traj1)
#     traj2_feat = traj_summarization(traj2)
#     feat_diff = [] 
#     for i in range(len(traj1_feat)):
#         feat1 = np.array(traj1_feat[i])
#         feat2 = np.array(traj2_feat[i])
#         feat_diff.append(np.sqrt(np.square(feat1-feat2).sum()))
#     weighted_distance = np.dot(feat_diff, weights) 
#     return weighted_distance
    

# def detect_outlier(trajs, p, D, dims=['x', 'y', 'z']):
#     """
#     takes in a set of trajectory demos and returns a list of the outlier.
#     Parameters:
#     -----------
#     trajs: list
#         A list of trajectories containing the positional features.
#     p: float 
#         (1 - p)% of the demos needs to be within distance D to not be considered an outlier
#     D: float 
#         distance threshold
#     Returns:
#     --------
#     outliers: list
#         A list of index of the trajectory that are outliers
#     """
#     keys = sorted(list(trajs.keys()))
#     n_trajs = len(keys)
#     threshold = ceil(n_trajs*(1-p))
#     neighbors = {i:[] for i in keys}
#     distances = []
#     for i in range(n_trajs):
#         i_key = keys[i]
#         for j in range(i+1, n_trajs):
#             j_key = keys[j]
#             # uses different distance method for comparing trajectory
#             align_dist = traj_weighted_dist(trajs[i_key][dims], trajs[j_key][dims], [1, 1, 0, 0])
#             # _, align_dist = dtw_funcs[config["alignment_method"]](trajs[i_key][dims], trajs[j_key][dims])
#             neighbors[i_key].append(align_dist)
#             neighbors[j_key].append(align_dist)
#             distances.append(align_dist)
#     std = np.std(distances)
#     D = D*std
#     outliers = []
#     for i in range(n_trajs):
#         i_key = keys[i]
#         if len([d for d in neighbors[i_key] if d < D]) < threshold:
#             outliers.append(i_key)
#     return outliers  
          