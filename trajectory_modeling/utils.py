from transformations import *

def get_mean_cov_hats(ref_means, ref_covs, min_len=None):
    '''
    This function computes the average mean and covariance across different object model.

    Parameters:
    ----------
    ref_means: list
        The means for models in each object reference frame.
    ref_covs: list
        The means for models in each object reference frame.
    min_len: int
        The minimum length that are desired. If None is given, it will be the minimum length ref_mean

    Returns:
    -------
    mean_hats: array
        N by D array, where N is the number of data points and D is the dimension of the data. Average mean at each data point.
    sigma_hats: array
        N * D * D array,where N is the number of data points and D is the dimension of the data. Average covariance at each data point.
    '''
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


def sample_trajectory_gmm(gmm, basis_mat, size=1, dims=7):
    """
    return a sample weight from the sampled gmm model and use basis function to convert to trajectory.
    Parameters:
    -----------
    gmm: object
        A fitted GMM model that weights of size d*N can be sampled from, 
        where d is dimension and N is number of basis functions
    basis_mat: np.array
        A (d*t)-by-(d*N) basis matrix that convert weight to trajectory and t is time size. 
    Returns:
    --------
    sampled_trajectories: list 
        A list of trajectories.
    """
    sampled_trajectories = []
    sampled_weights = gmm.sample(size)[0]
    for w in sampled_weights:
        traj = (basis_mat @ w).reshape(-1, dims)
        sampled_trajectories.append(traj)
    return sampled_trajectories