from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np
import os
import pickle
import yaml
from outlier_detection import detect_outlier


def plot_position(ax, prediction, ground_truth, mid = 0.5, title = 'predictions and groundtruth' ):
    '''
    This function 3D plots the prediction, ground truth trajectory including the start, middle, and the end of the trajectory.

    Parameters
    ----------
    ax: axis object
        The axis that will be plotted on.
    prediction: array
        N by 3 array, where N is the number of datapoints.
    ground_truth: array
        N by 3 array, where N is the number of datapoints.
    mid_ind: float(0 - 1)
        The middle of the trajectory. Default is 0.5
    title: str
        The title of the plot.
    '''
    mid_ind = int(mid * len(prediction))
    ax.plot(ground_truth[:, 2], ground_truth[:, 1], -ground_truth[:, 0], 'r', label='ground_truth')
    ax.plot(prediction[:, 2], prediction[:, 1], -prediction[:, 0], 'b', label='prediction')
    ax.plot(ground_truth[0, 2], ground_truth[0, 1], -ground_truth[0, 0], 'ro', label='start')
    ax.plot(ground_truth[mid_ind, 2], ground_truth[mid_ind, 1], -ground_truth[mid_ind, 0], 'rs', label='middle')
    ax.plot(ground_truth[-1, 2], ground_truth[-1, 1], -ground_truth[-1, 0], 'rx', label='end')
    ax.plot(prediction[0, 2], prediction[0, 1], -prediction[0, 0], 'bo', label='start')
    ax.plot(prediction[mid_ind, 2], prediction[mid_ind, 1], -prediction[mid_ind, 0], 'bs', label='middle')
    ax.plot(prediction[-1, 2], prediction[-1, 1], -prediction[-1, 0], 'bx', label='end')
    ax.set_xlabel('x(mm)')
    ax.set_ylabel('y(mm)')
    ax.set_zlabel('z(mm)')
    ax.set_box_aspect((np.ptp(ground_truth[:, 2]), np.ptp(ground_truth[:, 1]), np.ptp(-ground_truth[:, 0])))
    ax.set_title(title)

def plot_orientation( prediction, ground_truth , axes):
    '''
    This function will plot the ground truth orientation and the predicted orientation.

    Parameters
    ----------
    prediction: array
        N by 4 array, where N is the number of datapoints. The 4 dimensions are for the quaternions.
    ground_truth: array
        N by 4 array, where N is the number of datapoints. The 4 dimensions are for the quaternions.
    axes: list
        axes that will be used to plot the quaternions.


    '''
    prediction_normalized = normalize(prediction, axis=1)
    q = ['qx', 'qy', 'qz', 'qw']
    for i, ax in enumerate(axes):
        ax.plot(ground_truth[:, i], 'r', label='ground_truth')
        ax.plot(prediction[:, i], 'b', label='prediction')
        ax.plot(prediction_normalized[:, i], 'g', label = 'prediction_normalized')
        ax.set_title(f'{q[i]}')


def plot_position_in_objs(gripper_traj_in_obj, obj, ax, title = None, bad_demos = ['740521'], plot_line = True):
    demos = list(gripper_traj_in_obj[obj].keys())
    for i, demo in enumerate(gripper_traj_in_obj[obj]):
        df = gripper_traj_in_obj[obj][demo]
        if demo not in demos:
            continue
        else:
            if demo in bad_demos:
                line = ax.plot(df.loc[:, 'x'], df.loc[:, 'y'], df.loc[:, 'z'], 'k--', label=f'demo #{demo}', visible = plot_line)
                ax.plot(df.loc[:, 'x'].iloc[0], df.loc[:, 'y'].iloc[0], df.loc[:, 'z'].iloc[0], 'o',
                        color=line[0].get_color())
                ax.plot(df.loc[:, 'x'].iloc[-1], df.loc[:, 'y'].iloc[-1], df.loc[:, 'z'].iloc[-1], 'x',
                        color=line[0].get_color())
            else:
                line = ax.plot(df.loc[:, 'x'], df.loc[:, 'y'], df.loc[:, 'z'], label=f'demo #{demo}', visible = plot_line)
                # line = ax.plot(df.loc[:, 'z'], df.loc[:, 'y'], -df.loc[:, 'x'])
                ax.plot(df.loc[:, 'x'].iloc[0], df.loc[:, 'y'].iloc[0], df.loc[:, 'z'].iloc[0], 'o',
                        color=line[0].get_color())
                ax.plot(df.loc[:, 'x'].iloc[-1], df.loc[:, 'y'].iloc[-1], df.loc[:, 'z'].iloc[-1], 'x',
                        color=line[0].get_color())


        # plt.legend()
        ax.set_xlabel('x(mm)')
        ax.set_ylabel('y(mm)')
        ax.set_zlabel('z(mm)')

        if title == None:
            ax.set_title(f'Gripper trajectories in {obj} reference frame')
        else:
            ax.set_title(title)
    win_lens = 500
    ax.set_xlim(-win_lens, win_lens)
    ax.set_ylim(-win_lens, win_lens)
    ax.set_zlim(-win_lens, win_lens)
    plt.legend()

def plot_orientation_in_objs(gripper_traj_in_obj, obj, axes, title = None):
    dims = ['qx', 'qy', 'qz', 'qw']
    demos = sorted(list(gripper_traj_in_obj[obj].keys()))
    for i, ax in enumerate(axes):
        for demo in demos[10:]:
            df = gripper_traj_in_obj[obj][demo]
            quats = df.loc[:, ['qx', 'qy', 'qz', 'qw']].to_numpy()
            ax.plot(quats[:, i], label = f'{demo}')
        ax.set_title(f'{dims[i]}')
    plt.title = (f'Orientation in {obj} reference frame')
    # plt.legend()
    plt.tight_layout()

def remove_repetitive_labels(handles, labels):
    '''
    Remove repetitive labels for Matplotlib
    '''
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    return newHandles, newLabels

if __name__ == '__main__':
    # Load data
    task_config_dir = '../Process_data/postprocessed/2022-10-27'
    # task_config_dir = '../Process_data/postprocessed/2022-08-(17-21)'
    with open(os.path.join(task_config_dir, 'task_config.yaml')) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    base_dir = os.path.join(config["project_path"], config["postprocessed_dir"])
    triangulation = 'dlc3d'
    with open(os.path.join(base_dir, 'processed', triangulation,'gripper_trajs_in_obj_aligned_filtered.pickle', ), 'rb') as f:
        gripper_trajs_in_obj_only = pickle.load(f)
    with open(os.path.join(base_dir, 'processed', triangulation,'gripper_traj_in_grouped_objs_aligned_filtered.pickle', ), 'rb') as f:
        gripper_trajs_in_grouped_obj_only = pickle.load(f)

    obj_frames = ['pitcher-cup', 'tap', 'pitcher', 'tap', 'tap', 'tap']
    obj_frames = ['cup-pitcher', 'tap', 'pitcher', 'tap', 'tap', 'tap']
    obj_frames = ['global', 'tap', 'pitcher', 'tap', 'tap', 'tap']
    # obj_frames = ['teabag1', 'tap', 'pitcher', 'tap', 'tap', 'tap']

    n_actions = len(gripper_trajs_in_obj_only)
    plot_pos = False
    actions = ['Pour water to cup', 'Put pitcher under tap', 'Push tap switch', 'Wait',
               'Pull tap switch', 'Pour water to cup']
    std_thres = 2.5
    for i in range(n_actions):
        gripper_traj_in_obj = gripper_trajs_in_obj_only[i] | gripper_trajs_in_grouped_obj_only[i]
        outliers = {}
        # individuals = config['individuals']
        # individuals = ['teabag1-cup', 'cup-teabag1']
        individuals = [ind for ind in list(gripper_traj_in_obj.keys()) if ind != 'global']
        for individual in individuals:
            n_std = std_thres
            outlier_individual = detect_outlier(gripper_traj_in_obj[individual], n=n_std)
            start_points = []
            end_points = []
            demos = list(gripper_traj_in_obj[individual].keys())
            for demo in demos:
                if demo in outlier_individual:
                    continue
                df = gripper_traj_in_obj[individual][demo]
                start_points.append([df.loc[:, 'x'].iloc[0], df.loc[:, 'y'].iloc[0], df.loc[:, 'z'].iloc[0]])
                end_points.append([df.loc[:, 'x'].iloc[-1], df.loc[:, 'y'].iloc[-1], df.loc[:, 'z'].iloc[-1]])
            start_points = np.array(start_points)
            end_points = np.array(end_points)
            stds_start = np.std(start_points, axis = 0)
            stds_end = np.std(end_points, axis=0)
            print(f'The std for starting points is: {stds_start}')
            print(f'The std for end points is: {stds_end}')
            print(f'The outliers for individual {individual} are {outlier_individual}')
            outliers[individual] = outlier_individual
        obj_frame = obj_frames[i]
        if plot_pos:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            title = f'Gripper trajectories in {obj_frame} reference frame for the #{i+1} action: {actions[i]}'
            # title = f'Gripper trajectory for the action: {actions[i]}'
            plot_position_in_objs(gripper_traj_in_obj, obj_frame, ax, title, outliers[obj_frame], plot_line = True)
        else:
            fig, axes = plt.subplots(4,1, figsize = (8,8))
            plot_orientation_in_objs(gripper_traj_in_obj,obj_frame,axes)
            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='right')
            fig.suptitle(f'Orientation in {obj_frame} reference frame for the action: {actions[i]}')
    plt.show()



