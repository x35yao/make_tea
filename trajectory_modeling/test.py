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
from quaternion_metric import norm_diff_quat, inner_prod_quat
import yaml
from outlier_detection import detect_outlier
from quaternion_metric import process_quaternions


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

def marginal_w(model, t):
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

def predict(models, t, demo, Hs, individuals, mixture = False, modify_cov = False):

    '''
    Predict the trajectory for a test demo.

    Parameters
    ----------
    models: dict
        A dictionary of models for individual. Models could be pmp or gmm.
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
        H = Hs[individual][demo]
        model = models[individual]
        if not mixture:
            mu, sigma = marginal_w(model, t)
        else:
            mu, sigma = model.sample_mode_and_get_mean_traj(t)

        if modify_cov:
            for j, sig in enumerate(sigma):
                std_max = np.max(np.sqrt(np.diagonal(sig)))
                std_min = np.min(np.sqrt(np.diagonal(sig)))
                # if int(std_max / std_min) > 5:
                ind_max = np.argmax(np.sqrt(np.diagonal(sig)))
                for k in range(sig.shape[0]):
                    sig[k, k] = sig[ind_max, ind_max]
        new_mu = lintrans(np.array(mu), H) # Linear
        n_dims = len(mu[0])
        if n_dims != 3:
            quats = new_mu[:, -4:]
            quats_new = process_quaternions(quats, sigma = None)
            new_mu[:, -4:] = quats_new
        new_sigma = lintrans_cov(sigma, H)
        mus.append(new_mu)
        sigmas.append(new_sigma)
    mu_mean, sigma_mean = get_mean_cov_hats(mus, sigmas, modify_cov = modify_cov)
    return mu_mean, sigma_mean

def get_position_difference_per_step(d1, d2):
    return np.linalg.norm(d1 - d2, axis = 1)

def get_orientation_difference_per_step(Q1, Q2, method = 'prod'):
    result = []
    for i in range(Q1.shape[0]):
        q1 = Q1[i]
        q2 = Q2[i]
        if method == 'diff':
            result.append(norm_diff_quat(q1, q2))
        elif method == 'prod':
            result.append(inner_prod_quat(q1, q2))
    return result

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
    task_config_dir = '../Process_data/postprocessed/2022-10-06'
    with open(os.path.join(task_config_dir, 'task_config.yaml')) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    project_dir = config["project_path"] # Modify this to your need
    base_dir = os.path.join(project_dir, config["postprocessed_dir"])
    triangulation = 'leastereo'
    template_dir = os.path.join(project_dir, config["postprocessed_dir"],f'transformations/{triangulation}')
    individuals = config["individuals"] # The objects that we will place a reference frame on
    objs = config["objects"]
    d = Task_data(base_dir, template_dir, individuals, objs)
    n_actions = d.get_number_of_actions()
    gripper_trajs_truncated = d.get_gripper_trajectories_for_each_action()

    with open(os.path.join(base_dir, 'processed', triangulation,'gripper_trajs_in_obj_aligned_filtered.pickle',), 'rb') as f1:
        gripper_trajs_in_obj = pickle.load(f1)
    with open(os.path.join(base_dir, 'processed', triangulation,'HTs_obj_in_ndi.pickle',), 'rb') as f2:
        HTs_obj_in_ndi = pickle.load(f2)
    with open(os.path.join(base_dir, 'processed', triangulation,'gripper_trajs_in_grouped_objs_aligned_filtered.pickle',), 'rb') as f3:
        gripper_trajs_in_obj = pickle.load(f3)
    with open(os.path.join(base_dir, 'processed', triangulation,'HTs_grouped_objs_in_ndi.pickle',), 'rb') as f4:
        HTs_obj_in_ndi = pickle.load(f4)

    ind = 0 # index of action to be tested
    sigma = .4
    gripper_traj_in_ndi = gripper_trajs_truncated[ind]

    gripper_traj_in_obj = gripper_trajs_in_obj[ind]
    HT_obj_in_ndi = HTs_obj_in_ndi[ind]
    demos = gripper_traj_in_ndi.keys()

    # Train model
    n_train = 5
    # bad_demos = ['740521',  '648027', '781806', '506373', '318539', '506374']
    '''
    ['318539', '506374']: pitcher in action 2
    ['740521']: action 4 5 6
    ['648027', '781806']: cup in action 1
    ['506373'] tap in action 2, tap in action 3
    ['318539'] tap in action 5
    ['506374] tap in action 2
    '''
    bad_demos = ['134733']
    # train_demos = demos[:n_train]
    # test_demo = demos[-1]
    # test_demo = '506366'
    n_dims = len(d.dims)
    gmms = {}
    pmps = {}
    pmps_gmm = {}

    individuals = d.individuals
    individuals = [i for i in d.individuals if i != 'global']
    # print(individuals)
    outliers = []
    for individual in individuals:
        if individual == 'teabag1':
            n_std = 2.2
        else:
            n_std = 2.2
        outlier_individual = detect_outlier(gripper_traj_in_obj[individual], n = n_std)
        print(f'The outliers for individual {individual} are {outlier_individual}')
        outliers += outlier_individual
    outliers = list(set(outliers))

    bad_demos = bad_demos + outliers
    valid_demos = [demo for demo in demos if demo not in bad_demos]
    print(f'The number of valid demos is {len(valid_demos)}')
    train_demos = random.sample(valid_demos, k=n_train)
    test_demos = [demo for demo in valid_demos if demo not in train_demos and demo not in bad_demos]
    test_demos = ['134737']
    test_demo = random.sample(test_demos, k=1)[0]
    for individual in individuals:
        data_pmp, times = prepare_data_for_pmp(gripper_traj_in_obj, individual, train_demos, d.dims)
        # data_gmm = prepare_data_for_gmm(gripper_traj_in_obj, individual, train_demos, ['Time'] + d.dims)
        # n_states = 15
        # n_data = data_gmm.shape[0]
        # model_gmm = gmm.GMM(nb_states= n_states, nb_dim=n_dims + 1)
        # model_gmm.em(data_gmm, reg=1e-3, maxiter=100, verbose = True)
        # LL = np.max(model_gmm.LL)
        # bic = get_bic(LL, n_states * 2, n_data)
        # gmms[individual] = model_gmm

        model_pmp = pmp.PMP(data_pmp, times, n_dims, sigma = sigma, n_modes = 1)
        model_pmp.train(mixture = False)
        pmps[individual] = model_pmp

        model_pmp2 = pmp.PMP(data_pmp, times, n_dims, sigma=sigma, n_modes=1)
        model_pmp2.train(mixture=True)
        pmps_gmm[individual] = model_pmp2

        # print('Conditioning number for MAP: ',np.log(np.linalg.cond(model_pmp.promp.Sigma_w)))
        # # print('Conditioning number for LSM: ',np.log(np.linalg.cond(model_pmp2.gmm.covariances_[0])))
        # print('Conditioning number for LSM: ', np.log(np.linalg.cond(model_pmp2.Sigma_w)))

    ### Test on test traj

    ground_truth = gripper_traj_in_ndi[test_demo][d.dims].to_numpy()
    t_pmp = np.linspace(0, 1, ground_truth.shape[0])
    t_gmm = gripper_traj_in_ndi[test_demo]['Time'].to_numpy()

    # mu_mean_gmm, sigma_mean_gmm = predict(gmms, t_gmm, test_demo, HT_obj_in_ndi, individuals)
    mu_mean_pmp, sigma_mean_pmp = predict(pmps, t_pmp, test_demo, HT_obj_in_ndi, individuals)
    mu_mean_pmp_gmm, sigma_mean_pmp_gmm = predict(pmps_gmm, t_pmp, test_demo, HT_obj_in_ndi, individuals, mixture = False)

    mid = 0.8
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(1, 1, 1, projection = '3d')
    plot_position(ax1, mu_mean_pmp_gmm[:, :3], ground_truth[:, :3], mid, title = f'GMM position prediction for demo {test_demo}')
    fig1.legend()
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
    plot_position(ax2, mu_mean_pmp[:, :3], ground_truth[:, :3], mid, title = f'PMP position prediction for demo {test_demo}')
    fig2.legend()


    if n_dims > 3:
        fig3, axes = plt.subplots(4, 1, figsize=(10, 6))
        plot_orientation(mu_mean_pmp_gmm[:, 3:], ground_truth[:, 3:], axes)
        plt.legend()
        fig4, axes = plt.subplots(4, 1, figsize=(10, 6))
        plot_orientation(mu_mean_pmp[:, 3:], ground_truth[:, 3:], axes)
        plt.legend()
    plt.show()

    mid_ind = int(mid * len(ground_truth))
    dist_start_gmm = np.linalg.norm(ground_truth[0, :3] - mu_mean_pmp_gmm[0, :3])
    dist_mid_gmm = np.linalg.norm(ground_truth[mid_ind, :3] - mu_mean_pmp_gmm[mid_ind, :3])
    dist_end_gmm = np.linalg.norm(ground_truth[-1, :3] - mu_mean_pmp_gmm[-1, :3])
    print(f'GMM: This distance at the start: {dist_start_gmm} mm, middle :{dist_mid_gmm} mm, and end: {dist_end_gmm} mm')

    dist_start_pmp = np.linalg.norm(ground_truth[0, :3] - mu_mean_pmp[0, :3])
    dist_mid_pmp = np.linalg.norm(ground_truth[mid_ind, :3] - mu_mean_pmp[mid_ind, :3])
    dist_end_pmp = np.linalg.norm(ground_truth[-1, :3] - mu_mean_pmp[-1, :3])
    print(f'PMP: This distance at the start: {dist_start_pmp} mm, middle :{dist_mid_pmp} mm, and end: {dist_end_pmp} mm')
