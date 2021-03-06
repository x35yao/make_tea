import numpy as np
from dtw import dtw

# basic implementation
def dynamic_time_warp(data1, data2, window=0):
    """
    Takes two sequences and returns the list of matching indices 
    representing the path and its minimum cost 
    
    Parameters
    ----------
    data1: 1D numpy array
        reference values
    data2: 1D numpy array
        target values
    window: float 0-1 
        
    Returns
    -------
    match_list: list
        contains corresponding index of minimum path
    min_cost: float
        average cost of path with minimum cost
    """
    len1, len2 = len(data1), len(data2)
    # Initialize grid
    dtw = np.full((len1, len2), fill_value=float('inf'))
    dtw[0, 0] = 0
    w = int(len2*window)
    for i in range(1, len1):
        for j in range(1, len2):
            p = int(i/len1*len2)
            if window and (j < p - w or p + w < j): continue
            cost = abs(data1[i] - data2[j])
            dtw[i,j] = cost + min([dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1]])
    index_match = []
    
    # get optimal matching moving backwards. 
    cur_pair = [len1-1, len2-1]
    while cur_pair!=[0,0]:
        index_match.append(cur_pair.copy())
        min_pair_val = min([dtw[cur_pair[0]-1, cur_pair[1]], dtw[cur_pair[0], cur_pair[1]-1], dtw[cur_pair[0]-1, cur_pair[1]-1]])
        if dtw[cur_pair[0]-1, cur_pair[1]-1] <= min_pair_val:
            cur_pair[0] -= 1
            cur_pair[1] -= 1
        elif dtw[cur_pair[0], cur_pair[1]-1] <= min_pair_val:
            cur_pair[1] -= 1
        elif dtw[cur_pair[0]-1, cur_pair[1]] <= min_pair_val:
            cur_pair[0] -= 1
    index_match.reverse()
    match_list = np.array(index_match)-1
    return match_list, dtw[len1-1, len2-1]/len(index_match)


# Implementation by another library with constrains.
def dtw_lib(traj1, traj2):
    traj_align = dtw(traj1, traj2, keep_internals=True)
    return np.array([traj_align.index1, traj_align.index2]).transpose(), traj_align.distance