from utils import get_mean_cov_hats
from TP_PMP import pmp
import random
import pickle
import TP_GMM.gmm as gmm
import naive.naive as naive
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
        q = gripper_traj_in_obj[individual][d].loc[:, dims].to_numpy()
        Q.append(q)
        times.append(t)
    return Q, times

def prepare_data_for_pmp_all_frames(gripper_traj_in_obj, frames, demos, dims):
    '''
    This function will make sure the data can be used to fit a probabilistic movement primitives model.

    Parameters
    ----------
    gripper_traj_in_obj: dict
        The dictionary that has the gripper trajectories in each object for each demonstration.
    frames: str
        The reference frames that the trajectory is expressed in.
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
        t = gripper_traj_in_obj[frames[0]][d]['Time'].to_numpy().flatten()
        t = t / t[-1]
        temp = np.hstack([gripper_traj_in_obj[frame][d].loc[:, dims].to_numpy() for frame in frames])
        Q.append(temp)
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
    dims = ['Time'] + dims
    for d in demos:
        data_demo = gripper_traj_in_obj[individual][d][dims].to_numpy()
        temp.append(data_demo)
    data = np.concatenate(temp)
    return data

def marginal_w(model, t, mode_selected = 0):
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
    elif isinstance(model, naive.Naive):
        mu,sigma = model.predict(t)
    else:
        if model.gmm: ##### GMM #####
            mu, sigma = model.pmp.marginal_w(t, mode_selected)
        else: ##### Only one Gaussian #####
            mu, sigma = model.pmp.marginal_w(t)
    return np.array(mu), np.array(sigma)

def predict1(models, t, demo, Hs, individuals, modify_cov = False):

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
        mu, sigma = marginal_w(model, t)
        if modify_cov:
            ## Modify the variance to be the max value across dimensions
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

def predict2(model, t, demo, Hs, individuals, mode_selected = 0, n_dim = 7, modify_cov = False):

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

    for i, individual in enumerate(sorted(individuals)):
        H = Hs[individual][demo]
        mu, sigma = marginal_w(model, t, mode_selected)
        mu = mu[:, i*n_dim: (i+1)*n_dim]
        sigma = sigma[:, i*n_dim: (i+1)*n_dim, i*n_dim: (i+1)*n_dim]
        if modify_cov:
            ## Modify the variance to be the max value across dimensions
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


