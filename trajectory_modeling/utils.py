from glob import glob
import pandas as pd
import os
import re
import numpy as np
from dtw_util import *

from transformations import *
import pickle

def load_gripper_trajectories(base_dir):
    '''
    This function look for the demos in base_dir, and load the ndi file.

    Parameters
    ----------
    base_dir: string
        The path to the folder where all the demonstrations are saved.

    Returns
    -------
    gripper_trajs: dict
        A dictionary whose keys are the demonstrations' ids.
        The values are the dataframes of the corresponding
        demonstration's gripper trajectory data in ndi reference frame.
    '''
    all_files = os.listdir(base_dir)
    r = re.compile("^[0-9]+$")
    demos = list(filter(r.match, all_files))
    gripper_trajs = {}
    for demo in demos:
        demo_dir = os.path.join(base_dir, demo)
        ndi_file = glob(os.path.join(demo_dir, '*NDI*'))[0]
        df_temp = pd.read_csv(ndi_file)
        gripper_trajs[demo] = df_temp

    return gripper_trajs

def load_obj_trajectories(base_dir, triangulation = 'dlc3d'):
    '''
    This function look for the demos in base_dir, and load marker_3d file.

    Parameters
    ----------
    base_dir: string
        The path to the folder where all the demonstrations are saved.
    triangulation: string
        'leastereo' or 'dlc3d', which corresponds to which triangulation method is used to get the

    Returns
    -------
    markers_trajs: dict
        A dictionary whose keys are the demonstrations' ids. The values are the dataframes of the corresponding
        demonstration's objects' pose trajectories in camera reference frame.
    '''

    all_files = os.listdir(base_dir)
    r = re.compile("^[0-9]+$")
    demos = list(filter(r.match, all_files))
    markers_trajs = {}
    for demo in demos:
        demo_dir = os.path.join(base_dir, demo)
        markers_traj_file = os.path.join(demo_dir, triangulation, 'markers_trajectory_3d.h5')
        df_temp = pd.read_hdf(markers_traj_file)
        markers_trajs[demo] = df_temp.droplevel('scorer', axis=1)
    return  markers_trajs

def get_number_of_actions(base_dir):
    n_actions = []
    all_files = os.listdir(base_dir)
    r = re.compile("^[0-9]+$")
    demos = list(filter(r.match, all_files))
    for demo in demos:
        n_action = 0
        demo_dir = os.path.join(base_dir, demo)
        servo_file = glob(os.path.join(demo_dir, '*Servo*'))[0]
        with open(servo_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'closed' in line:
                    n_action += 1
        n_actions.append(n_action)
    return int(np.median(n_actions))

def get_gripper_trajectories_for_each_action(base_dir, dfs_ndi, n_actions):
    '''
        Given the gripper trajs this function will go in basedir and look for the Servo file
        to chunk the trajs and output each action's traj.

        Parameters
        ---------
        dfs_ndi: dict
            A directory that contains the full gripper trajectories for different demos.
        base_dir: string
            The path to the directory that contains the demos
        i: int
            The index of the action

        Returns
        gripper_trajs: The gripper trajectories for each action from different demonstrations.
        -------
        '''
    gripper_trajs = {}
    demos = dfs_ndi.keys()
    for demo in demos:
        gripper_trajs[demo] = []
        df_ndi = dfs_ndi[demo]
        demo_dir = os.path.join(base_dir, demo)
        servo_file = glob(os.path.join(demo_dir, '*Servo*'))[0]
        with open(servo_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'closed' in line:
                    t_start = float(lines[i].split(',')[0])
                    t_end = float(lines[1 + i].split(',')[0])
                    idx_start = df_ndi['Time'].sub(t_start).abs().idxmin()
                    idx_end = df_ndi['Time'].sub(t_end).abs().idxmin()
                    df_gripper = df_ndi.copy()[idx_start: idx_end]
                    df_gripper = df_gripper.drop(columns=df_gripper.columns[0])
                    # Set the time to start from 0
                    df_gripper.loc[:, 'Time'] = df_gripper.loc[:, 'Time'] - df_gripper.loc[:, 'Time'].iloc[0]
                    gripper_trajs[demo].append(df_gripper)
    action_trajs = []
    for i in range(int(n_actions)):
        temp = {}
        for demo in demos:
            temp[demo] = gripper_trajs[demo][i]
        action_trajs.append(temp)
    return action_trajs

def get_obj_trajectories_for_for_each_action(base_dir, dfs_camera, n_actions, slack = 20):
    '''
    Given the markers_trajs loaded, this function will go in basedir and look for the Servo file
    to chunk the trajs and output the each action's traj.

    Parameters
    ---------
    dfs_camera: dict
        A directory that contains the full object markers trajectories for different demos.
    basedir: string
        The path to the directory that contains the demos
    slack: The amount of time to look back at the beginning of an action and to look forward at the end.
    '''
    markers_trajs = {}
    demos = dfs_camera.keys()
    for demo in demos:
        markers_trajs[demo] = []
        df_camera = dfs_camera[demo]
        demo_dir = os.path.join(base_dir, demo)
        servo_file = glob(os.path.join(demo_dir, '*Servo*'))[0]
        with open(servo_file, 'r') as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                if 'closed' in line:
                    t_start = float(lines[i].split(',')[0]) - slack
                    t_end = float(lines[1 + i].split(',')[0]) + slack
                    task_duration =  float(lines[-1].split(',')[0])
                    if t_start < 0:
                        t_start =0
                    if t_end > task_duration:
                        t_end = task_duration
                    action_duration = t_end - t_start
                    idx_start = int(len(df_camera) * t_start / task_duration)
                    idx_end = int(len(df_camera) * t_end / task_duration)
                    df_markers = df_camera[idx_start: idx_end].reset_index(drop=True)
                    df_markers.loc[:, 'Time'] = np.arange(len(df_markers)) / len(df_markers) * action_duration
                    markers_trajs[demo].append(df_markers)
    action_trajs = []
    for i in range(int(n_actions)):
        temp = {}
        for demo in demos:
            temp[demo] = markers_trajs[demo][i]
        action_trajs.append(temp)
    return action_trajs


def convert_trajectories_to_objects_reference_frame(gripper_trajs_in_ndi, HTs, individuals, dims = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'], ignore_orientation = True ):
    '''
    This function will convert the gripper trajectories from NDI reference frame to objects' frames.

    Parameters:
    ----------
    gripper_trajs_in_ndi: dict
        A dictionary that contains the gripper trajectory in each demonstration.
    HTs: dict
        A dictionary that contains the homogeneous transformation matrix that will convert the trajectories from NDI to object's reference frames.
    individuals: list
        objects that are relative to the task.
    dims: list
        The dimensions that will be converted.
    ignore_orientation: bool
        Whether or not take the orientation of the objects into consideration when covert trajectories to objects reference frames.
    '''
    gripper_trajs_in_obj = {}
    for individual in individuals:
        gripper_trajs_in_obj[individual] = {}
        for demo in gripper_trajs_in_ndi.keys():
            obj_in_ndi = HTs[individual][demo]
            if ignore_orientation:
                obj_in_ndi[:3, :3] = np.eye(3)
            ndi_in_obj = inverse_homogenous_transform(obj_in_ndi)
            original_traj = gripper_trajs_in_ndi[demo][dims].to_numpy()
            traj_transformed = lintrans(original_traj, ndi_in_obj)
            df_temp = gripper_trajs_in_ndi[demo].copy()
            df_temp[dims] = traj_transformed
            gripper_trajs_in_obj[individual][demo] = df_temp
    return gripper_trajs_in_obj

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



