import matplotlib.pyplot as plt
import yaml
import os
from process_data import Task_data
import pickle
from TP_PMP import pmp as pmp
from TP_PMP import pmp3 as pmp3
from test import prepare_data_for_pmp, predict1,predict2, get_position_difference_per_step, prepare_data_for_pmp_all_frames
import numpy as np
from transformations import inverse_homogeneous_transform, lintrans, homogeneous_transform
import matplotlib
from plot import remove_repetitive_labels
from outlier_detection import detect_outlier
import random
from tqdm import trange
import math
from quaternion_metric import norm_diff_quat, normalize_quats
from scipy.spatial.transform import Rotation as R

if __name__ == '__main__':

    PLOT = True
    n_tests = 1

    font = {'size': 14}
    matplotlib.rc('font', **font)

    # Load data
    task_config_dir = '../Process_data/postprocessed/2022-10-27'
    with open(os.path.join(task_config_dir, 'task_config.yaml')) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    project_dir = config["project_path"] # Modify this to your need
    base_dir = os.path.join(project_dir, config["postprocessed_dir"])
    triangulation = 'dlc3d'
    # triangulation = 'leastereo'
    template_dir = os.path.join(project_dir, config["postprocessed_dir"],f'transformations/{triangulation}')
    individuals = config["individuals"] # The objects that we will place a reference frame on
    objs = config["objects"]
    data = Task_data(base_dir, triangulation, individuals)
    n_actions = data.get_number_of_actions()
    gripper_trajs_truncated = data.get_gripper_trajectories_for_each_action()

    with open(os.path.join(base_dir, 'processed', triangulation, 'gripper_trajs_in_obj_aligned_filtered.pickle',), 'rb') as f1:
        gripper_trajs_in_obj_all_actions = pickle.load(f1)
    with open(os.path.join(base_dir, 'processed', triangulation, 'HTs_obj_in_ndi.pickle',), 'rb') as f2:
        HTs_obj_in_ndi_all_actions = pickle.load(f2)
    with open(os.path.join(base_dir, 'processed', triangulation, 'gripper_traj_in_grouped_objs_aligned_filtered.pickle',), 'rb') as f3:
        gripper_trajs_in_grouped_objs_all_actions = pickle.load(f3)
    with open(os.path.join(base_dir, 'processed', triangulation, 'HTs_grouped_objs_in_ndi.pickle',), 'rb') as f4:
        HTs_grouped_objs_in_ndi_all_actions = pickle.load(f4)

    ind = 0  # index of action to be tested
    gripper_trajs_in_ndi = gripper_trajs_truncated[ind]
    gripper_traj_in_obj = gripper_trajs_in_obj_all_actions[ind]
    gripper_traj_in_grouped_obj = gripper_trajs_in_grouped_objs_all_actions[ind]
    gripper_traj_in_generalized_obj = gripper_traj_in_obj | gripper_traj_in_grouped_obj

    HTs_obj_in_ndi = HTs_obj_in_ndi_all_actions[ind]
    HTs_grouped_obj_in_ndi = HTs_grouped_objs_in_ndi_all_actions[ind]
    HTs_generalized_obj_in_ndi = HTs_obj_in_ndi | HTs_grouped_obj_in_ndi

    outliers = []
    std_thres = 2.9
    individuals = [ind for ind in list(HTs_generalized_obj_in_ndi.keys()) if ind != 'global']
    individuals = ['cup', 'pitcher', 'cup-pitcher', 'pitcher-cup']
    # individuals = ['cup']
    # individuals = ['cup', 'pitcher']
    for individual in individuals:
        n_std = std_thres
        outlier_individual = detect_outlier(gripper_traj_in_generalized_obj[individual], n=n_std)
        print(f'The outliers for individual {individual} are {outlier_individual}')
        # print(f'The outliers for individual {individual} are {outlier_individual}')
        outliers += outlier_individual
    outliers = list(set(outliers))
    bad_demos = outliers + ['331732']

    demos = sorted(list(HTs_obj_in_ndi['global'].keys()))
    train_demos_pool = [demo for demo in demos if demo not in bad_demos]
    test_demos_pool = [demo for demo in demos[0:20] if demo not in bad_demos]
    # Train model
    print(f'The number of training pool is: {len(train_demos_pool)}')
    print(f'The number of outliers is: {len(outliers)}')
    n_dims = len(data.dims)
    n_train = 6
    max_iter = 20
    d1_pos, d2_pos= [], []
    d1_ori, d2_ori = [], []
    for i in trange(n_tests):
        selected_demo = random.sample(train_demos_pool[:10], k=1)
        traj_in_cup, times = prepare_data_for_pmp(gripper_traj_in_generalized_obj, 'cup', selected_demo, data.dims)
        n_rotations = 10
        step = int(360 / n_rotations)
        rotation_degrees  = np.arange(0, 360, step)
        rotation_radians = np.array([math.radians(d) for d in rotation_degrees])
        trajs_in_cup = []
        trajs_in_global = []
        trajs_in_pitcher = []
        trajs_in_cup_pitcher = []
        trajs_in_pitcher_cup = []
        HTs = []
        for rotation_radian in rotation_radians:
            rotation_matrix = R.from_euler('x', -rotation_radian).as_matrix()
            HT = homogeneous_transform(rotation_matrix, [0, 0, 0])
            HTs.append(HT)
            traj_in_cup_rotated = lintrans(traj_in_cup[0], HT)
            traj_in_global_rotated = lintrans(traj_in_cup_rotated, HTs_generalized_obj_in_ndi['cup'][selected_demo[0]])
            traj_in_pitcher_rotated = lintrans(traj_in_global_rotated, inverse_homogeneous_transform(HTs_generalized_obj_in_ndi['pitcher'][selected_demo[0]]))
            traj_in_cup_pitcher_rotated = lintrans(traj_in_global_rotated, inverse_homogeneous_transform(HTs_generalized_obj_in_ndi['cup-pitcher'][selected_demo[0]]))
            traj_in_pitcher_cup_rotated = lintrans(traj_in_global_rotated, inverse_homogeneous_transform(HTs_generalized_obj_in_ndi['pitcher-cup'][selected_demo[0]]))
            trajs_in_cup.append(traj_in_cup_rotated)
            trajs_in_global.append(traj_in_global_rotated)
            trajs_in_pitcher.append(traj_in_pitcher_rotated)
            trajs_in_cup_pitcher.append(traj_in_cup_pitcher_rotated)
            trajs_in_pitcher_cup.append(traj_in_pitcher_cup_rotated)
        train_inds = random.sample(list(np.arange(0,n_rotations)), k = n_train)
        test_demos_pool = [ind for ind in list(np.arange(0, n_rotations)) if ind not in train_inds]
        test_inds = random.sample(test_demos_pool, k = 1)
        test_ind = test_inds[0]
        data_all_frames = []
        data_all_frames_pop = []
        for ind in train_inds:
            data_all_frames.append(np.concatenate([trajs_in_cup[ind], trajs_in_pitcher[ind]]))
            data_all_frames_pop.append(np.concatenate([trajs_in_cup_pitcher[ind], trajs_in_pitcher_cup[ind]]))
        times = times * 6
        Tn = len(times[0])
        model_pmp = pmp.PMP(data_all_frames, times, n_dims, sigma=0.035, n_components=1, covariance_type = 'diag',  max_iter = max_iter)
        model_pmp.train(print_lowerbound=False)
        model_pmp.refine()
        ### Test on test traj
        ground_truth = trajs_in_global[test_inds[0]]
        t_pmp = np.linspace(0, 1, ground_truth.shape[0])
        HTs_test = {}
        for individual in ['cup', 'pitcher']:
            HT = HTs[test_ind]
            obj_in_ndi = HTs_generalized_obj_in_ndi[individual][test_ind] @ HT
            HTs_test[individual] = obj_in_ndi
        mu_mean_tp_pmp1, sigma_mean_tp_pmp1 = predict2(model_pmp, t_pmp, HTs_test, ['cup', 'pitcher'], mode_selected=ind)
        mu_pos_tp_pmp1 = np.array(mu_mean_tp_pmp1)[:, :3]
        mu_ori_tp_pmp1 = np.array(mu_mean_tp_pmp1)[:, 3:]

        model_pmp_2 = pmp.PMP(data_all_frames_pop, times, n_dims, sigma=0.035, n_components=1,
                            covariance_type='diag', max_iter=max_iter)
        model_pmp_2.train(print_lowerbound=False)
        model_pmp_2.refine()
        ### Test on test traj
        t_pmp = np.linspace(0, 1, ground_truth.shape[0])
        HTs_test_pop = {}
        for individual in ['cup-pitcher', 'pitcher-cup']:
            HT = HTs[test_ind]
            obj_in_ndi = HTs_generalized_obj_in_ndi[individual][test_ind] @ HT
            HTs_test_pop[individual] = obj_in_ndi
        mu_mean_tp_pmp2, sigma_mean_tp_pmp2 = predict2(model_pmp_2, t_pmp, HTs_test_pop,
                                                       ['cup-pitcher', 'pitcher-cup'], mode_selected=0)
        mu_pos_tp_pmp2 = np.array(mu_mean_tp_pmp2)[:, :3]
        mu_ori_tp_pmp2 = np.array(mu_mean_tp_pmp2)[:, 3:]

        mid = 0.7
        mid_ind = int(mid * len(ground_truth))

        ##### TP-PMP ##########

        d = get_position_difference_per_step(ground_truth[:, :3], mu_pos_tp_pmp1[:, :3])
        d1_pos.append([d[0], d[mid_ind], d[-1], np.mean(d)])

        d = get_position_difference_per_step(ground_truth[:, :3], mu_pos_tp_pmp2[:, :3])
        d2_pos.append([d[0], d[mid_ind], d[-1], np.mean(d)])


        d1_ori.append(norm_diff_quat(ground_truth[:, 3:], mu_ori_tp_pmp1))
        d2_ori.append(norm_diff_quat(ground_truth[:, 3:], mu_ori_tp_pmp2))


        # print(
        #     f'TP-PMP with objects only: The average quaternion distance is {np.mean(d1_ori, axis=0)}')
        if PLOT:
            ####### PLot position #######################
            fig = plt.figure(figsize = (12, 6))
            ax = fig.add_subplot(1, 1, 1, projection='3d')

            for ind in train_inds + test_inds:
                traj = trajs_in_global[ind]
                middle = int(mid * len(traj))
                if ind in train_inds and ind not in test_inds:
                    line = ax.plot(traj[:, 2], traj[:, 1], -traj[:, 0], '--', color = 'grey', label = 'Training demos')
                    ax.plot(traj[0, 2], traj[0, 2], -traj[0, 2], 'o',
                            color='black', label='start')
                    ax.plot(traj[middle, 2], traj[middle, 2], -traj[middle, 2], 's',
                            color='black', label='middle')
                    ax.plot(traj[-1, 2], traj[-1, 2], -traj[-1, 2], 'x',
                            color='black', label='end')
                else:
                    line = ax.plot(df.loc[:, 'z'], df.loc[:, 'y'], -df.loc[:, 'x'], '-', color='red', label='Test demo')
                    ax.plot(df.loc[:, 'z'].iloc[0], df.loc[:, 'y'].iloc[0], -df.loc[:, 'x'].iloc[0], 'o',
                            color= line[0].get_color(), label='start')
                    ax.plot(df.loc[:, 'z'].iloc[middle], df.loc[:, 'y'].iloc[middle], -df.loc[:, 'x'].iloc[middle],
                            's',
                            color='black', label='middle')
                    ax.plot(df.loc[:, 'z'].iloc[-1], df.loc[:, 'y'].iloc[-1], -df.loc[:, 'x'].iloc[-1], 'x',
                            color=line[0].get_color(), label='end')


            line1 = ax.plot(mu_pos_tp_pmp1[:, 2], mu_pos_tp_pmp1[:, 1] , -mu_pos_tp_pmp1[:, 0], '-', color='blue', label='TP-ProMP GMM')
            ax.plot(mu_pos_tp_pmp1[0, 2], mu_pos_tp_pmp1[0, 1] , -mu_pos_tp_pmp1[0, 0], 'o',
                    color= line1[0].get_color(), label='start')
            ax.plot(mu_pos_tp_pmp1[mid_ind, 2], mu_pos_tp_pmp1[mid_ind, 1], -mu_pos_tp_pmp1[mid_ind, 0], 's',
                    color=line1[0].get_color(), label='middle')
            ax.plot(mu_pos_tp_pmp1[-1, 2], mu_pos_tp_pmp1[-1, 1] , -mu_pos_tp_pmp1[-1, 0], 'x',
                    color= line1[0].get_color(), label='end')
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_zlabel('z (mm)')

            line2 = ax.plot(mu_pos_tp_pmp2[:, 2], mu_pos_tp_pmp2[:, 1], -mu_pos_tp_pmp2[:, 0], '-', color='green',
                            label='TP-ProMP Gaussian')
            ax.plot(mu_pos_tp_pmp2[0, 2], mu_pos_tp_pmp2[0, 1], -mu_pos_tp_pmp2[0, 0], 'o',
                    color=line2[0].get_color(), label='start')
            ax.plot(mu_pos_tp_pmp2[mid_ind, 2], mu_pos_tp_pmp2[mid_ind, 1], -mu_pos_tp_pmp2[mid_ind, 0], 's',
                    color=line2[0].get_color(), label='middle')
            ax.plot(mu_pos_tp_pmp2[-1, 2], mu_pos_tp_pmp2[-1, 1], -mu_pos_tp_pmp2[-1, 0], 'x',
                    color=line2[0].get_color(), label='end')
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_zlabel('z (mm)')


            ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
            # ax.set_box_aspect((np.ptp(mu_pos_tp_pmp1[:,2]), np.ptp(mu_pos_tp_pmp1[:,1]), np.ptp(mu_pos_tp_pmp1[:,0])))

            handles, labels = ax.get_legend_handles_labels()
            newHandles, newLabels = remove_repetitive_labels(handles, labels)
            plt.legend(newHandles, newLabels, loc = 'upper left')
            # plt.title('Position')
            plt.show()
            # plt.savefig('position.eps', bbox_inches='tight', format='eps')
            #### Plot Orientation ##############
            fig2, axes = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
            for demo in train_demos + [test_demo]:
                quats = gripper_traj_in_obj['global'][demo][data.dims].to_numpy()[:, 3:]
                if demo in train_demos:
                    for i, ax in enumerate(axes):
                        ax.plot(quats[:, i], '--', color='grey', label='Training demos')
                else:
                    for i, ax in enumerate(axes):
                        ax.plot(quats[:, i], '-', color='red', label='Test demo')

            # Plot pmp and gmm prediction
            quats = mu_ori_tp_pmp1
            quats_normalized = normalize_quats(quats)
            # mu__tp_pmp1[:, 3:] = quats_normalized
            for i, ax in enumerate(axes):
                ax.plot(quats_normalized[:, i], '-', color='blue', label='TP-ProMP GMM')
                ax.set_title(data.dims[i + 3])
                if i == 3:
                    ax.set_xlabel('Time')

            quats = mu_ori_tp_pmp2
            quats_normalized = normalize_quats(quats)
            # mu__tp_pmp1[:, 3:] = quats_normalized
            for i, ax in enumerate(axes):
                ax.plot(quats_normalized[:, i], '-', color='green', label='TP-ProMP Gaussian')
                ax.set_title(data.dims[i + 3])
                if i == 3:
                    ax.set_xlabel('Time')
            handles, labels = ax.get_legend_handles_labels()
            newHandles, newLabels = remove_repetitive_labels(handles, labels)
            plt.legend(newHandles, newLabels, loc = 'lower left')
            plt.savefig('orientation.eps', bbox_inches='tight', format='eps')
            plt.show()

    print(
        f'1 : This distance at the start: {np.mean(np.array(d1_pos), axis=0)[0]} mm, middle: {np.mean(np.array(d1_pos), axis=0)[1]}, end: {np.mean(np.array(d1_pos), axis=0)[2]} mm, average: {np.mean(np.array(d1_pos), axis=0)[-1]} mm, average quaternion: {np.mean(np.array(d1_ori))}')
    print(
        f'1 : This distance at the start: {np.mean(np.array(d2_pos), axis=0)[0]} mm, middle: {np.mean(np.array(d2_pos), axis=0)[1]}, end: {np.mean(np.array(d2_pos), axis=0)[2]} mm, average: {np.mean(np.array(d2_pos), axis=0)[-1]} mm, average quaternion: {np.mean(np.array(d2_ori))}')
