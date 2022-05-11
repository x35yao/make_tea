import numpy as np
import pandas as pd


def attach_grip_states(df, grip):
    new_df = df.copy()
    new_df['action'] = None
    for i, row in grip.iterrows():
        check_time = row['Timestamp']
        new_df.iloc[(df['Time'] - check_time).abs().argsort()[:1], -1] = row['Gripper state']
    return new_df

def extract_points(dfs, interval_size, backward=False):
    max_lens = [len(df) for df in dfs]
    frame_periods, active_demos = [], []
    for t in range(max(max_lens)):
        if t > 0 and t%interval_size==0:
            if backward:
                holder = [[df.iloc[max_lens[s] - t].to_numpy()] for s, df in enumerate(dfs) if t < len(df)]
            else:
                holder = [[df.iloc[t].to_numpy()] for df in dfs if t < len(df)]
            active_demos.append(len(holder))
            frame_periods.append(np.concatenate(holder))
    return frame_periods, active_demos

def get_mean_cov(pts_lst, rho=1, shrink=False):
    mean_lst, cov_mats = [], []
    for period, pts in enumerate(pts_lst):
        cov_mats.append(np.cov(pts, rowvar=False) + np.diag(np.full(pts.shape[1], rho)))
#         multipliers.append(np.trace(cov_mats[-1])/np.trace(cov_mats[-2]))
#         if shrink and np.trace(cov_mats[-1]):
        mean_lst.append(np.mean(pts, axis=0))
    return mean_lst, cov_mats

def get_mean_cov_hats(ref_means, ref_covs, min_len=None):
    sigma_hats, ref_pts = [], len(ref_means)
    if not min_len:
        min_len = min([len(r) for r in ref_means]) 
    # solve for global covariance
    for p in range(min_len):
        covs = [cov[p] for cov in ref_covs]
        inv_sum = np.linalg.inv(covs[0])
        for ref in range(1, ref_pts):
            inv_sum = inv_sum + np.linalg.inv(covs[ref])      
        sigma_hat = np.linalg.inv(inv_sum)
        sigma_hats.append(sigma_hat)
    
    mean_hats = []
    for p in range(min_len):
        mean_w_sum = np.matmul(np.linalg.inv(ref_covs[0][p]), ref_means[0][p]) 
        for ref in range(1, ref_pts):
            mean_w_sum = mean_w_sum + np.matmul(np.linalg.inv(ref_covs[ref][p]), ref_means[ref][p])
        mean_hats.append(np.matmul(sigma_hats[p], mean_w_sum))
    return np.array(mean_hats), np.array(sigma_hats)

