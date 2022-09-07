from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np
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

