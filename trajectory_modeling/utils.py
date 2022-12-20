from transformations import *

def compute_diag_norm(cov):
    n_dim = cov.shape[0]
    sum = 0
    for i in range(n_dim):
        sum += cov[i][i]**2
    return np.sqrt(sum)

def get_mean_cov_hats(ref_means, ref_covs, min_len=None, modify_cov = False):
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
        inv_sum = np.zeros(ref_covs[0][0].shape)
        for ref in range(ref_pts):
            inv_sum += np.linalg.inv(covs[ref])
        sigma_hat = np.linalg.inv(inv_sum)
        sigma_hats.append(sigma_hat)

    # solve for global mean
    mean_hats = []
    for p in range(min_len):
        mean_w_sum = np.zeros(ref_means[0][0].shape)
        for ref in range(ref_pts):
            mean_w_sum += np.matmul(np.linalg.inv(ref_covs[ref][p]), ref_means[ref][p])
        mu = np.matmul(sigma_hats[p], mean_w_sum)
        mean_hats.append(mu)
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


def min_max_var(covs):
    """
    returns the index and minimum overall dimension variance across all timestep
    Parameters:
    -----------
    covs: list
        A list of covariances index by time
    Returns:
    --------
    best_index: int 
        index of the minimum.
    best_min_var: float 
        the optimal variance founded
    """
    best_index, best_min_var = 0, float('inf')
    for i, cov in enumerate(covs):
        t_max_var = np.diag(cov).max()
        if t_max_var < best_min_var:
            best_index = i
            best_min_var = t_max_var
    return best_index, best_min_var

def obj_ref_traj(ref_covs):
    """
    return the name of the reference frame with the best minimum overall dimension variance
    across all timestep.
    Parameters:
    -----------
    ref_covs: dict
        a dictionary containing the name of the reference frames as keys and the covariance 
        across timesteps as values.
    Returns:
    --------
    best_ref: str
        name of the best reference frame
    """
    best_ref, best_min_var_all = None, float('inf')
    for ref, covs in ref_covs.items():
        _, min_ref_var = min_max_var(covs)
        if min_ref_var < best_min_var_all:
            best_min_var_all = min_ref_var
            best_ref = ref
    return best_ref


def combinations_from_list(lens_set):
    """
    create a list of all combination of length of lens_set and value of each dimension up to 
    the corresponding value in lens_set.
    Parameters:
    -----------
    lens_set: list
        a list of positive ints.
    Returns:
    --------
    combinations: list
        a 2D list of combinations
    """
    combinations = [[]]
    curdim = 0
    while curdim < len(lens_set):
        new_combinations = []
        for i in range(lens_set[curdim]):
            for j in combinations:
                new_combinations.append(j + [i])
        combinations = new_combinations
        curdim += 1
    return combinations