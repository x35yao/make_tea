from utils import get_mean_cov_hats
from TP_PMP import pmp
import random
import pickle
import TP_GMM.gmm as gmm
from matplotlib import pyplot as plt
from plot import plot_position, plot_orientation
from process_data import Task_data
# import os
from transformations import *
import numpy as np
import configparser
import yaml


def prepare_data_for_pmp(gripper_traj_in_obj, individual, demos, dims):
    '''
    This function will make sure the data can be used to fit a probabilistic movement primitives model.

    Parameters
    ----------
    gripper_traj_in_obj: dict
        The dictionary that has the gripper trajectories in each object for each demonstration.
    individual: str
        The object that the trajectory is converted to.
    demos: list
        A list of demos that will be used to fit the model
    dims: list
        A list of dimensions that will be modeled.['x', 'y', 'z'] or ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

    Returns
    -------
    Q: list
        A list of trajectories, where each entry is for a demo.
    times: list
        A list ot time dimension for each trajectory, where each entry is for a demo.

    '''
    Q = []
    times = []
    for d in demos:
        t = gripper_traj_in_obj[individual][d]['Time'].to_numpy().flatten()
        t = t / t[-1]
        Q.append(
            gripper_traj_in_obj[individual][d].loc[:, dims].to_numpy())
        times.append(t)
    return Q, times

def prepare_data_for_gmm(gripper_traj_in_obj, individual, demos, dims):
    '''
    This function will make sure the data can be used to fit a probabilistic movement primitives model.

    Parameters
    ----------
    gripper_traj_in_obj: dict
        The dictionary that has the gripper trajectories in each object for each demonstration.
    individual: str
        The object that the trajectory is converted to.
    demos: list
        A list of demos that will be used to fit the model
    dims: list
        A list of dimensions that will be modeled.['x', 'y', 'z'] or ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

    Returns
    -------
    data: array
    N by D array, where N is the number of datapoints from all demonstrations. D is the number of dimensions.
    '''
    temp = []
    for d in demos:
        data_demo = gripper_traj_in_obj[individual][d][dims].to_numpy()
        temp.append(data_demo)
    data = np.concatenate(temp)
    return data

def marginal_t(model, t):
    '''
    This function marginalize over time for a gmm model

    Parameters
    ----------
    model: gmm or pmp model
    t: array
        time that will be marginalized. If the model is pmp model, t should be from 0 to 1.

    '''
    if isinstance(model, gmm.GMM):
        n_dims = model.nb_dim
        mu, sigma = model.condition(t[:, None], dim_in=slice(0, 1), dim_out=slice(1, n_dims))
    else:
        mu, sigma = model.marginal_w(t)
    return np.array(mu), np.array(sigma)

def predict(models, t, demo, Hs, individuals):
    '''
    Predict the trajectory for a test demo.

    Parameters
    ----------
    models: dict
        A dictionary of models for each train demos. Models could be pmp or gmm.
    t: Ndarray
        The time of the trajectory to be predicted. If the model is pmp model, t should be from 0 to 1.
    demo: str
        The demo id of the trajectory to be predicted.
    Hs: dict
        The homogeneous transformations for each individual in each demo. Hs[individual][demo]
    individuals: list
        The objects to be used to predict the trajectory.
    '''
    mus = []
    sigmas = []
    for i, individual in enumerate(individuals):
        H = Hs[demo][individual]
        model = models[individual]
        mu, sigma = marginal_t(model, t) # Marginalize t
        mu = lintrans(np.array(mu), H) # Linear
        mus.append(mu)
        sigmas.append(sigma)
    mu_mean, sigma_mean = get_mean_cov_hats(mus, sigmas)
    return mu_mean, sigma_mean

def get_position_difference_per_step(d1, d2):
    return np.linalg.norm(d1 - d2, axis = 1)

def get_bic(LL, k, n):
    '''
    This function computes the bic score of a model.

    Parameters
    ----------
    LL: int
        The maximized value of the likelihood function of the model
    k: int
        The number of parameters of the model
    n: int
        The number of datapoints

    '''

    return k * np.log(n) - 2 * LL * n

def get_aic(LL, k, n):
    '''
    This function computes the bic score of a model.

    Parameters
    ----------
    LL: int
        The maximized value of the likelihood function of the model
    k: int
        The number of parameters of the model
    n: int
        The number of datapoints

    '''

    return 2 * k - 2 * LL * n

if __name__ == '__main__':
    # Load data
    with open('../task_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    project_dir = config["project_path"] # Modify this to your need
    base_dir = os.path.join(project_dir, config["postprocessed_dir"])
    template_dir = os.path.join(project_dir, config["postprocessed_dir"],'transformations/dlc3d')
    individuals = config["individuals"] # The objects that we will place a reference frame on
    objs = config["objects"]
    d = Task_data(base_dir, template_dir, individuals, objs)
    gripper_trajs_full = d.load_gripper_trajectories()
    n_actions = d.get_number_of_actions()
    gripper_trajs_truncated = d.get_gripper_trajectories_for_each_action()

    with open(os.path.join(base_dir, 'processed', 'gripper_trajs_in_obj_aligned_filtered.pickle',), 'rb') as f1:
        gripper_trajs_in_obj = pickle.load(f1)
    with open(os.path.join(base_dir, 'processed', 'HTs_obj_in_ndi.pickle',), 'rb') as f2:
        HTs_obj_in_ndi = pickle.load(f2)

    ind = 1 # index of action to be tested
    gripper_traj_in_ndi = gripper_trajs_truncated[ind]

    gripper_traj_in_obj = gripper_trajs_in_obj[ind]
    HT_obj_in_ndi = HTs_obj_in_ndi[ind]
    demos = gripper_traj_in_ndi.keys()

    # Train model
    n_train = 6
    n_dims = len(d.dims)
    # bad_demos = ['463678', '636936', '636938', '463675']
    bad_demos = ['740521', '506373', '506365']
    demos = [demo for demo in demos if demo not in bad_demos]
    train_demos = random.sample(demos, k=n_train)
    test_demos = [demo for demo in demos if demo not in train_demos and demo not in bad_demos]
    test_demo = random.sample(test_demos, k = 1)[0]

    gmms = {}
    pmps = {}

    for individual in d.individuals:
        data_pmp, times = prepare_data_for_pmp(gripper_traj_in_obj, individual, train_demos, d.dims)
        data_gmm = prepare_data_for_gmm(gripper_traj_in_obj, individual, train_demos, ['Time'] + d.dims)
        n_states = 15
        n_data = data_gmm.shape[0]
        model_gmm = gmm.GMM(nb_states= n_states, nb_dim=n_dims + 1)
        model_gmm.em(data_gmm, reg=1e-3, maxiter=100, verbose = True)
        LL = np.max(model_gmm.LL)
        bic = get_bic(LL, n_states * 2, n_data)
        gmms[individual] = model_gmm

        model_pmp = pmp.PMP(data_pmp, times, n_dims)
        model_pmp.train()
        pmps[individual] = model_pmp

    ### Test on test traj

    ground_truth = gripper_traj_in_ndi[test_demo][d.dims].to_numpy()
    t_pmp = np.linspace(0, 1, ground_truth.shape[0])
    t_gmm = gripper_traj_in_ndi[test_demo]['Time'].to_numpy()

    mu_mean_gmm, sigma_mean_gmm = predict(gmms, t_gmm, test_demo, HT_obj_in_ndi, individuals)
    mu_mean_pmp, sigma_mean_pmp = predict(pmps, t_pmp, test_demo, HT_obj_in_ndi, individuals)

    mid = 0.8
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(1, 1, 1, projection = '3d')
    plot_position(ax1, mu_mean_gmm[:, :3], ground_truth[:, :3], mid, title = f'GMM position prediction for demo {test_demo}')
    fig1.legend()
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
    plot_position(ax2, mu_mean_pmp[:, :3], ground_truth[:, :3], mid, title = f'PMP position prediction for demo {test_demo}')
    fig2.legend()


    if n_dims > 3:
        fig3, axes = plt.subplots(4, 1, figsize=(10, 6))
        plot_orientation(mu_mean_gmm[:, 3:], ground_truth[:, 3:], axes)
        plt.legend()
        fig4, axes = plt.subplots(4, 1, figsize=(10, 6))
        plot_orientation(mu_mean_pmp[:, 3:], ground_truth[:, 3:], axes)
        plt.legend()
    plt.show()

    mid_ind = int(mid * len(ground_truth))
    dist_start_gmm = np.linalg.norm(ground_truth[0, :3] - mu_mean_gmm[0, :3])
    dist_mid_gmm = np.linalg.norm(ground_truth[mid_ind, :3] - mu_mean_gmm[mid_ind, :3])
    dist_end_gmm = np.linalg.norm(ground_truth[-1, :3] - mu_mean_gmm[-1, :3])
    print(f'GMM: This distance at the start: {dist_start_gmm} mm, middle :{dist_mid_gmm} mm, and end: {dist_end_gmm} mm')

    dist_start_pmp = np.linalg.norm(ground_truth[0, :3] - mu_mean_pmp[0, :3])
    dist_mid_pmp = np.linalg.norm(ground_truth[mid_ind, :3] - mu_mean_pmp[mid_ind, :3])
    dist_end_pmp = np.linalg.norm(ground_truth[-1, :3] - mu_mean_pmp[-1, :3])
    print(f'PMP: This distance at the start: {dist_start_pmp} mm, middle :{dist_mid_pmp} mm, and end: {dist_end_pmp} mm')
