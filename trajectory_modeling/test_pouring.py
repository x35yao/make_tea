import matplotlib.pyplot as plt
import yaml
import os
from process_data import Task_data
import pickle
from TP_PMP import pmp
from test import prepare_data_for_pmp, predict, get_position_difference_per_step
import numpy as np
from transformations import inverse_homogenous_transform, lintrans
import matplotlib
from plot import remove_repetitive_labels
from outlier_detection import detect_outlier
import random
from tqdm import trange
from quaternion_metric import norm_diff_quat, inner_prod_quat

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
    std_thres = 3
    individuals = [ind for ind in list(HTs_generalized_obj_in_ndi.keys()) if ind != 'global']
    for individual in individuals:
        n_std = std_thres
        outlier_individual = detect_outlier(gripper_traj_in_generalized_obj[individual], n=n_std)
        print(f'The outliers for individual {individual} are {outlier_individual}')
        # print(f'The outliers for individual {individual} are {outlier_individual}')
        outliers += outlier_individual
    outliers = list(set(outliers))
    bad_demos = outliers

    demos = sorted(list(HTs_obj_in_ndi['global'].keys()))
    train_demos_pool = [demo for demo in demos[10:] if demo not in bad_demos]
    test_demos_pool = [demo for demo in demos[10:] if demo not in bad_demos]
    # Train model
    print(f'The number of training pool is: {len(train_demos_pool)}')
    print(f'The number of outliers is: {len(outliers)}')
    # data.dims = ['x', 'y', 'z']
    n_dims = len(data.dims)
    n_train = 5
    d1, d2, d3, d4, d5 = [],[],[],[],[]
    for i in trange(n_tests):
        train_demos = random.sample(train_demos_pool, k=n_train)
        # train_demos = train_demos_pool[:n_train]
        test_demos_pool_updated = [demo for demo in test_demos_pool if demo not in train_demos]
        test_demos = random.sample(test_demos_pool_updated, k=1)

        test_demo = test_demos[0]
        ground_truth = gripper_traj_in_obj['global'][test_demo][data.dims].to_numpy()
        tp_pmps = {}

        for individual in individuals:
            data_pmp, times = prepare_data_for_pmp(gripper_traj_in_generalized_obj, individual, train_demos, data.dims)
            model_pmp = pmp.PMP(data_pmp, times, n_dims, sigma=0.5)
            model_pmp.train()
            tp_pmps[individual] = model_pmp

        ### Test on test traj
        # ground_truth = gripper_trajs_in_ndi[test_demo][data.dims].to_numpy()
        t_pmp = np.linspace(0, 1, ground_truth.shape[0])
        mu_mean_tp_pmp1, sigma_mean_tp_pmp1 = predict(tp_pmps, t_pmp, test_demo, HTs_generalized_obj_in_ndi, config['individuals'])
        mu_pos_tp_pmp1 = mu_mean_tp_pmp1

        mu_mean_tp_pmp2, sigma_mean_tp_pmp2 = predict(tp_pmps, t_pmp, test_demo, HTs_generalized_obj_in_ndi, list(HTs_grouped_obj_in_ndi.keys()))
        mu_pos_tp_pmp2 = mu_mean_tp_pmp2

        mu_mean_tp_pmp3, sigma_mean_tp_pmp3 = predict(tp_pmps, t_pmp, test_demo, HTs_generalized_obj_in_ndi, individuals)
        mu_pos_tp_pmp3 = mu_mean_tp_pmp3

        mu_mean_tp_pmp4, sigma_mean_tp_pmp4 = predict(tp_pmps, t_pmp, test_demo, HTs_generalized_obj_in_ndi, individuals, modify_cov=True)
        mu_pos_tp_pmp4 = mu_mean_tp_pmp4

        print(f'Test demo is {test_demo}')
        mid = 0.75
        mid_ind = int(mid * len(ground_truth))
        ### PMP #####
        # H_cup_in_ndi = HTs_generalized_obj_in_ndi['cup'][test_demo]
        # H_pitcher_in_ndi = HTs_generalized_obj_in_ndi['pitcher'][test_demo]
        # H_ndi_in_pitcher = inverse_homogenous_transform(H_pitcher_in_ndi)
        # H_cup_in_pitcher = H_ndi_in_pitcher @ H_cup_in_ndi
        # data_pmp, times = prepare_data_for_pmp(gripper_traj_in_obj, 'pitcher', train_demos, data.dims)
        # model_pmp = pmp.PMP(data_pmp, times, n_dims, sigma=0.035)
        # model_pmp.train()
        # model_pmp.condition(mid, 1, q=H_cup_in_pitcher[:-1, 3], ignore_Sy=False)
        # mu_mean_pmp_pitcher, sigma_mean_pmp_pitcher = model_pmp.marginal_w(t_pmp, q=True)
        # mu_mean_pmp_pitcher = np.array(mu_mean_pmp_pitcher)
        # mu_mean_pmp = lintrans(mu_mean_pmp_pitcher, H_pitcher_in_ndi)

        ##### TP-PMP ##########
        d = get_position_difference_per_step(ground_truth[:, :3], mu_pos_tp_pmp1[:, :3])
        d1.append([d[0], d[mid_ind], d[-1], np.mean(d)])
        # print(
        #     f'TP-PMP with objects: This distance at the start: {dist_start_tp_pmp} mm, end: {dist_end_tp_pmp} mm')

        d = get_position_difference_per_step(ground_truth[:, :3], mu_pos_tp_pmp2[:, :3])
        d2.append([d[0], d[mid_ind], d[-1], np.mean(d)])
        # print(
        #     f'TP-PMP with grouped objects: This distance at the start: {dist_start_tp_pmp} mm, end: {dist_end_tp_pmp} mm')

        d = get_position_difference_per_step(ground_truth[:, :3], mu_pos_tp_pmp3[:, :3])
        d3.append([d[0], d[mid_ind], d[-1], np.mean(d)])

        # print(
        #     f'TP-PMP with objects and grouped objects: This distance at the start: {dist_start_tp_pmp} mm, end: {dist_end_tp_pmp} mm')

        d = get_position_difference_per_step(ground_truth[:, :3], mu_pos_tp_pmp4[:, :3])
        d4.append([d[0], d[mid_ind], d[-1], np.mean(d)])
        # print(
        #     f'TP-PMP with objects and grouped objects and modified covariance matrix: This distance at the start: {dist_start_tp_pmp} mm, end: {dist_end_tp_pmp} mm')

        # d = get_position_difference_per_step(ground_truth[:, :3], mu_mean_pmp[:, :3])
        # d5.append([d[0], d[mid_ind], d[-1], np.mean(d)])
        # # print(
        # #     f'PMP: This distance at the start: {dist_start_pmp} mm, end: {dist_end_pmp} mm')

        if PLOT:
            ####### PLot position #######################
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(1, 1, 1, projection='3d')

            for demo in train_demos + test_demos:
                df = gripper_trajs_in_ndi[demo]
                middle = int(mid * len(df))
                if demo in train_demos and demo not in test_demos:
                    line = ax.plot(df.loc[:, 'z'], df.loc[:, 'y'], -df.loc[:, 'x'], '--', color = 'grey', label = 'Training demos')
                    ax.plot(df.loc[:, 'z'].iloc[0], df.loc[:, 'y'].iloc[0], -df.loc[:, 'x'].iloc[0], 'o',
                            color='black', label='start')
                    ax.plot(df.loc[:, 'z'].iloc[middle], df.loc[:, 'y'].iloc[middle], -df.loc[:, 'x'].iloc[middle], 's',
                            color='black', label='middle')
                    ax.plot(df.loc[:, 'z'].iloc[-1], df.loc[:, 'y'].iloc[-1], -df.loc[:, 'x'].iloc[-1], 'x',
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


            line1 = ax.plot(mu_pos_tp_pmp1[:, 2], mu_pos_tp_pmp1[:, 1] , -mu_pos_tp_pmp1[:, 0], '-', color='blue', label='TP-ProbMP-objects')
            ax.plot(mu_pos_tp_pmp1[0, 2], mu_pos_tp_pmp1[0, 1] , -mu_pos_tp_pmp1[0, 0], 'o',
                    color= line1[0].get_color(), label='start')
            ax.plot(mu_pos_tp_pmp1[mid_ind, 2], mu_pos_tp_pmp1[mid_ind, 1], -mu_pos_tp_pmp1[mid_ind, 0], 's',
                    color=line1[0].get_color(), label='middle')
            ax.plot(mu_pos_tp_pmp1[-1, 2], mu_pos_tp_pmp1[-1, 1] , -mu_pos_tp_pmp1[-1, 0], 'x',
                    color= line1[0].get_color(), label='end')

            line2 = ax.plot(mu_pos_tp_pmp2[:, 2], mu_pos_tp_pmp2[:, 1], -mu_pos_tp_pmp2[:, 0], '-', color='yellow', label='TP-ProbMP2-grouped-objects')
            ax.plot(mu_pos_tp_pmp2[0, 2], mu_pos_tp_pmp2[0, 1], -mu_pos_tp_pmp2[0, 0], 'o',
                    color=line2[0].get_color(), label='start')
            ax.plot(mu_pos_tp_pmp2[mid_ind, 2], mu_pos_tp_pmp2[mid_ind, 1], -mu_pos_tp_pmp2[mid_ind, 0], 's',
                    color=line2[0].get_color(), label='middle')
            ax.plot(mu_pos_tp_pmp2[-1, 2], mu_pos_tp_pmp2[-1, 1], -mu_pos_tp_pmp2[-1, 0], 'x',
                    color=line2[0].get_color(), label='end')

            line3 = ax.plot(mu_pos_tp_pmp3[:, 2], mu_pos_tp_pmp3[:, 1], -mu_pos_tp_pmp3[:, 0], '-', color='purple',
                           label='TP-ProbMP2-objects-and-grouped-objects')
            ax.plot(mu_pos_tp_pmp3[0, 2], mu_pos_tp_pmp3[0, 1], -mu_pos_tp_pmp3[0, 0], 'o',
                    color=line3[0].get_color(), label='start')
            ax.plot(mu_pos_tp_pmp3[mid_ind, 2], mu_pos_tp_pmp3[mid_ind, 1], -mu_pos_tp_pmp3[mid_ind, 0], 's',
                    color=line3[0].get_color(), label='middle')
            ax.plot(mu_pos_tp_pmp3[-1, 2], mu_pos_tp_pmp3[-1, 1], -mu_pos_tp_pmp3[-1, 0], 'x',
                    color=line3[0].get_color(), label='end')

            line4 = ax.plot(mu_pos_tp_pmp4[:, 2], mu_pos_tp_pmp4[:, 1], -mu_pos_tp_pmp4[:, 0], '-', color='green',
                           label='modified')
            ax.plot(mu_pos_tp_pmp4[0, 2], mu_pos_tp_pmp4[0, 1], -mu_pos_tp_pmp4[0, 0], 'o',
                    color=line4[0].get_color(), label='start')
            ax.plot(mu_pos_tp_pmp4[mid_ind, 2], mu_pos_tp_pmp4[mid_ind, 1], -mu_pos_tp_pmp4[mid_ind, 0], 's',
                    color=line4[0].get_color(), label='middle')
            ax.plot(mu_pos_tp_pmp4[-1, 2], mu_pos_tp_pmp4[-1, 1], -mu_pos_tp_pmp4[-1, 0], 'x',
                    color=line4[0].get_color(), label='end')

            # line5 = ax.plot(mu_mean_pmp[:, 2], mu_mean_pmp[:, 1], -mu_mean_pmp[:, 0], '-', color='Green', label='ProbMP')
            # ax.plot(mu_mean_pmp[0, 2], mu_mean_pmp[0, 1], -mu_mean_pmp[0, 0], 'o',
            #         color=line5[0].get_color(), label='start')
            # ax.plot(mu_mean_pmp[mid_ind, 2], mu_mean_pmp[mid_ind, 1], -mu_mean_pmp[mid_ind, 0], 's',
            #         color=line5[0].get_color(), label='middle')
            # ax.plot(mu_mean_pmp[-1, 2], mu_mean_pmp[-1, 1], -mu_mean_pmp[-1, 0], 'x',
            #         color=line5[0].get_color(), label='end')
            handles, labels = ax.get_legend_handles_labels()
            newHandles, newLabels = remove_repetitive_labels(handles, labels)

            plt.legend(newHandles, newLabels)
            plt.title('ProbMP vs. TP-ProbMP for a pick-and-place action')
            plt.show()

            #### Plot Orientation ##############
            fig2, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
            for demo in train_demos + [test_demo]:
                quats = gripper_traj_in_obj['global'][demo][data.dims].to_numpy()[:, 3:]
                if demo in train_demos:
                    for i, ax in enumerate(axes):
                        ax.plot(quats[:, i], '--', color='grey', label='Training demos')
                else:
                    for i, ax in enumerate(axes):
                        ax.plot(quats[:, i], '-', color='red', label='Test demo')

            # Plot pmp and gmm prediction
            for i, ax in enumerate(axes):
                ax.plot(mu_pos_tp_pmp1[:, i + 3], '-', color='blue', label='Objects only')
                ax.plot(mu_pos_tp_pmp2[:, i + 3], '-', color='yellow', label='Grouped objects only')
                ax.plot(mu_pos_tp_pmp3[:, i + 3], '-', color='purple', label='Objects + Grouped objects')
                ax.plot(mu_pos_tp_pmp3[:, i + 3], '-', color='green', label='Modified')
                ax.set_title(data.dims[i + 3])
                if i == 3:
                    ax.set_xlabel('Time')
            handles, labels = ax.get_legend_handles_labels()
            newHandles, newLabels = remove_repetitive_labels(handles, labels)
            plt.legend(newHandles, newLabels)
            plt.show()

    print(f'TP-PMP with objects only: This distance at the start: {np.mean(np.array(d1), axis = 0)[0]} mm, middle: {np.mean(np.array(d1), axis = 0)[1]}, end: {np.mean(np.array(d1), axis = 0)[2]} mm, average: {np.mean(np.array(d1), axis = 0)[-1]} mm')
    print(
        f'TP-PMP with grouped objects only: This distance at the start: {np.mean(np.array(d2), axis=0)[0]} mm, middle: {np.mean(np.array(d2), axis = 0)[1]}, end: {np.mean(np.array(d2), axis=0)[2]} mm, average: {np.mean(np.array(d2), axis = 0)[-1]} mm')
    print(
        f'TP-PMP with objects and grouped objects: This distance at the start: {np.mean(np.array(d3), axis=0)[0]} mm, middle: {np.mean(np.array(d3), axis = 0)[1]}, end: {np.mean(np.array(d3), axis=0)[2]} mm, average: {np.mean(np.array(d3), axis = 0)[-1]} mm')
    print(
        f'TP-PMP with objects and grouped objects and modified covariance matrix: This distance at the start: {np.mean(np.array(d4), axis=0)[0]} mm, middle: {np.mean(np.array(d4), axis = 0)[1]}, end: {np.mean(np.array(d4), axis=0)[2]} mm, average: {np.mean(np.array(d4), axis = 0)[-1]} mm')
    # print(
    #     f'PMP with objects: This distance at the start: {np.mean(np.array(d5), axis=0)[0]} mm, middle: {np.mean(np.array(d5), axis = 0)[1]}, end: {np.mean(np.array(d5), axis=0)[2]} mm, average: {np.mean(np.array(d5), axis = 0)[-1]} mm')

