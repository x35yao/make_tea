from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np
import os
import pickle
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


def plot_position_in_objs(gripper_traj_in_obj, obj, ax, title = None, bad_demos = ['740521']):
    for i, demo in enumerate(gripper_traj_in_obj[obj]):
        if demo in bad_demos:
            continue
        # if i != 1:
        #     continue
        df = gripper_traj_in_obj[obj][demo]
        line = ax.plot(df.loc[:, 'z'], df.loc[:, 'y'], -df.loc[:, 'x'], label=f'demo #{demo}')
        # line = ax.plot(df.loc[:, 'z'], df.loc[:, 'y'], -df.loc[:, 'x'])
        ax.plot(df.loc[:, 'z'].iloc[0], df.loc[:, 'y'].iloc[0], -df.loc[:, 'x'].iloc[0], 'o',
                color=line[0].get_color())
        ax.plot(df.loc[:, 'z'].iloc[-1], df.loc[:, 'y'].iloc[-1], -df.loc[:, 'x'].iloc[-1], 'x',
                color=line[0].get_color())
        # plt.legend()
        ax.set_xlabel('x(mm)')
        ax.set_ylabel('y(mm)')
        ax.set_zlabel('z(mm)')
        if title == None:
            ax.set_title(f'Gripper trajectories in {obj} reference frame')
        else:
            ax.set_title(title)
    plt.legend()

def plot_orientation_in_objs(gripper_traj_in_obj, obj, axes, title = None):
    dims = ['qx', 'qy', 'qz', 'qw']
    for i, ax in enumerate(axes):
        for demo in gripper_traj_in_obj[obj]:
            df = gripper_traj_in_obj[obj][demo]
            quats = df.loc[:, ['qx', 'qy', 'qz', 'qw']].to_numpy()
            ax.plot(quats[:, i], label = f'{demo}')
        ax.set_title(f'{dims[i]}')
    plt.title = (f'Orientation in {obj} reference frame')
    plt.legend()
    plt.tight_layout()


if __name__ == '__main__':
    base_dir = '/home/luke/Desktop/project/make_tea/Process_data/postprocessed/2022-08-(17-21)'
    with open(os.path.join(base_dir, 'processed', 'gripper_trajs_in_obj_aligned_filtered.pickle', ), 'rb') as f:
        gripper_trajs_in_obj = pickle.load(f)
    objs = ['teabag1', 'teabag1', 'pitcher', 'tap', 'tap', 'cup']
    bad_demos = ['740521',  '506365']
    #bad_demos = []
    n_actions = len(gripper_trajs_in_obj)
    plot_pos = True
    actions = ['Put teabag into cup', 'Put pitcher under tap', 'Push tap switch', 'Wait',
               'Pull tap switch', 'Pour water to cup']
    for i in range(n_actions):
        gripper_traj_in_obj = gripper_trajs_in_obj[i]
        obj = objs[i]
        if plot_pos:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            title = f'Gripper trajectories in {obj} reference frame for the #{i+1} action: {actions[i]}'
            # title = f'Gripper trajectory for the action: {actions[i]}'
            plot_position_in_objs(gripper_traj_in_obj, obj, ax, title, bad_demos)
        else:
            fig, axes = plt.subplots(4,1, figsize = (8,8))
            plot_orientation_in_objs(gripper_traj_in_obj,obj,axes)
            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='right')
            fig.suptitle(f'Orientation in {obj} reference frame for the action: {actions[i]}')
    plt.show()



