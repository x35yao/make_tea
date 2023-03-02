import matplotlib.pyplot as plt
import yaml
import os
from process_data import Task_data
import pickle
from TP_PMP import pmp
from TP_GMM import gmm
from naive import naive
from test import prepare_data_for_pmp, predict2, predict1, get_position_difference_per_step, prepare_data_for_gmm
import numpy as np
from transformations import inverse_homogenous_transform, lintrans
import matplotlib
from plot import remove_repetitive_labels
from outlier_detection import detect_outlier
import random
from tqdm import trange
from quaternion_metric import norm_diff_quat, normalize_quats
if __name__ == '__main__':

    PLOT = True
    n_tests = 1
    max_iter = 30
    # font = {'size': 12}
    # matplotlib.rc('font', **font)

    # Load data
    task_config_dir = '../Process_data/postprocessed/2022-12-01'
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
    # demos_not_detected = ['882778', '882777', '882774', '882816', '882792']
    # bad_demos = outliers + ['882782', '882786']
    bad_demos_middle = ['882775']
    bad_demos_left = ['882779', '882784', '882785','882786', '882793', '882789', '882810', '882796', '882797', '882807']
    bad_demos = bad_demos_middle + bad_demos_left + outliers

    demos = sorted(list(HTs_obj_in_ndi['global'].keys()))

    middle_pool = [d for d in demos[:8] if d not in bad_demos]
    left_pool = [d for d in demos[8:26] if d not in bad_demos]
    right_pool = [d for d in demos[26:] if d not in bad_demos]

    train_demos_pool = [demo for demo in demos if demo not in bad_demos]
    test_demos_pool = [demo for demo in demos if demo not in bad_demos]
    # Train model
    print(f'The number of training pool is: {len(train_demos_pool)}')
    # print(f'The number of outliers is: {len(outliers)}')
    data.dims = ['x', 'y', 'z']
    n_dims = len(data.dims)

    d1_pos, d2_pos, d3_pos, d4_pos = [], [], [], []
    d1_ori, d2_ori, d3_ori, d4_ori = [], [], [], []
    for i in trange(n_tests):
        train_left = random.sample(left_pool, k=2)
        train_middle = random.sample(middle_pool, k=2)
        train_right = random.sample(right_pool, k=2)
        train_demos = train_left + train_middle + train_right
        # train_demos = train_left
        test_demos_pool_updated = [demo for demo in test_demos_pool if demo not in train_demos]
        # test_demos = random.sample(test_demos_pool_updated, k=1)
        right_pool_updated = [demo for demo in right_pool if demo not in train_demos]
        middle_pool_updated = [demo for demo in middle_pool if demo not in train_demos]
        left_pool_updated = [demo for demo in left_pool if demo not in train_demos]
        test_demos = random.sample(test_demos_pool_updated, k=1)
        test_demo = test_demos[0]
        print(test_demo)

        ground_truth = gripper_traj_in_obj['global'][test_demo][data.dims].to_numpy()

        data_all_frames_tp_pmp = {}
        data_all_frames_pmp = {}
        individuals =['puck-net', 'net-puck', 'puck', 'net']
        individuals = ['puck-net', 'net-puck']
        for individual in sorted(individuals):
            data_per_frame_tp_pmp, times = prepare_data_for_pmp(gripper_traj_in_generalized_obj, individual,
                                                                train_demos,
                                                                data.dims)
            data_all_frames_tp_pmp[individual] = data_per_frame_tp_pmp
        data_per_frame_pmp, times = prepare_data_for_pmp(gripper_traj_in_generalized_obj, 'global', train_demos,
                                                         data.dims)
        data_all_frames_pmp['global'] = data_per_frame_pmp

        ### TP-ProMP model
        sigma = 3
        model_tp_pmp = pmp.PMP(data_all_frames_tp_pmp, times, n_dims, sigma=sigma, n_components=1,
                               covariance_type='diag', max_iter=max_iter, gmm=False)
        model_tp_pmp.train(print_lowerbound=False)
        # model_tp_pmp.refine()

        t_pmp = np.linspace(0, 1, ground_truth.shape[0])
        mu_mean_tp_pmp, sigma_mean_tp_pmp = predict2(model_tp_pmp, t_pmp, test_demo, HTs_generalized_obj_in_ndi,
                                                     individuals, n_dim=n_dims, mode_selected = 0)
        mu_pos1 = mu_mean_tp_pmp[:, :3]
        mu_ori1 = mu_mean_tp_pmp[:, 3:]
        d = get_position_difference_per_step(ground_truth[:, :3], mu_pos1)
        d1_pos.append([d[0], d[-1], np.mean(d)])
        d1_ori.append(norm_diff_quat(ground_truth[:, 3:], mu_ori1))

        ### ProMP model
        model_pmp = pmp.PMP(data_all_frames_pmp, times, n_dims, sigma=sigma, n_components=1,
                            covariance_type='diag', max_iter=max_iter, gmm=False)
        model_pmp.train(print_lowerbound=False)

        H_puck_in_ndi = HTs_generalized_obj_in_ndi['puck'][test_demo]
        start = ground_truth[0].copy()
        # start[:3] = H_puck_in_ndi[:-1, 3] + np.array([-250, -10, 0])

        model_pmp.condition(0, 1, q = start, ignore_Sy=False)
        mu_mean_pmp, sigma_mean_pmp = model_pmp.pmp.marginal_w(t_pmp)
        mu_pos2 = np.array(mu_mean_pmp)[:, :3]
        mu_ori2 = np.array(mu_mean_pmp)[:, 3:]
        d = get_position_difference_per_step(ground_truth[:, :3], mu_pos2)
        d2_pos.append([d[0], d[-1], np.mean(d)])
        d2_ori.append(norm_diff_quat(ground_truth[:, 3:], mu_ori2))

        #### TP-GMM model
        max_nb_states = 6
        average_dist_min = np.inf
        for i in range(max_nb_states):
            nb_states = i + 2
            tp_gmms = {}
            t_gmm = gripper_traj_in_obj['global'][test_demo]['Time'].to_numpy()
            for individual in sorted(['puck', 'net']):
                data_per_frame_gmm = prepare_data_for_gmm(gripper_traj_in_generalized_obj, individual, train_demos,
                                                          data.dims)
                model_tp_gmm = gmm.GMM(nb_states=nb_states, nb_dim=data_per_frame_gmm.shape[1])
                model_tp_gmm.em(data_per_frame_gmm, reg=1e-8, maxiter=200, verbose=False)
                tp_gmms[individual] = model_tp_gmm
            try:
                mu_mean_tp_gmm, sigma_mean_tp_gmm = predict1(tp_gmms, t_gmm, test_demo, HTs_generalized_obj_in_ndi,
                                                         ['puck', 'net'])
            except ValueError:
                continue
            d = get_position_difference_per_step(ground_truth[:, :3], mu_mean_tp_gmm[:, :3])
            d_mean = np.mean(d)
            if d_mean < average_dist_min:
                average_dist_min = d_mean
                pos_temp = [d[0], d[1], d_mean]
                mu_pos3 = mu_mean_tp_gmm[:, :3]
                mu_ori3 = mu_mean_tp_gmm[:, 3:]
                ori_temp = norm_diff_quat(ground_truth[:, 3:], mu_ori3)
        d3_pos.append(pos_temp)
        d3_ori.append(ori_temp)

        # ### Naive model
        # naives = {}
        # for individual in sorted(individuals):
        #     data_per_frame_naive, times = prepare_data_for_pmp(gripper_traj_in_generalized_obj, individual,
        #                                                        train_demos,
        #                                                        data.dims)
        #     naive_model = naive.Naive()
        #     naive_model.train(data_per_frame_naive, times)
        #     naives[individual] = naive_model
        # mu_mean_naive, sigma_mean_naive = predict1(naives, t_pmp, test_demo, HTs_generalized_obj_in_ndi,
        #                                            individuals)
        # mu_pos4 = mu_mean_naive[:,:3]
        # mu_ori4 = mu_mean_naive[:,3:]
        # d = get_position_difference_per_step(ground_truth[:, :3], mu_pos4)
        # d4_pos.append([d[0], d[-1], np.mean(d)])
        # d4_ori.append(norm_diff_quat(ground_truth[:, 3:], mu_ori4))

    H_net_in_ndi = HTs_generalized_obj_in_ndi['net'][test_demo]
    net_position = H_net_in_ndi[:-1, 3]
    puck_position = H_puck_in_ndi[:-1, 3]
    if PLOT:
        ###### PLot position #######################
        matplotlib.rcParams.update({'font.size': 10})
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_facecolor('white')
        ax.locator_params(nbins=3, axis='z')
        for demo in train_demos + test_demos:
            df = gripper_traj_in_obj['global'][demo]
            if demo in train_demos and demo not in test_demos:
                line = ax.plot(df.loc[:, 'z'], df.loc[:, 'y'], -df.loc[:, 'x'], '--', color='grey',
                               label='Training demos')
                ax.plot(df.loc[:, 'z'].iloc[0], df.loc[:, 'y'].iloc[0], -df.loc[:, 'x'].iloc[0], 'o',
                        color='grey', label='start')
                ax.plot(df.loc[:, 'z'].iloc[-1], df.loc[:, 'y'].iloc[-1], -df.loc[:, 'x'].iloc[-1], 'x',
                        color='grey', label='end')
            else:
                line = ax.plot(df.loc[:, 'z'], df.loc[:, 'y'], -df.loc[:, 'x'], '-', color='red', label='Test demo')
                ax.plot(df.loc[:, 'z'].iloc[0], df.loc[:, 'y'].iloc[0], -df.loc[:, 'x'].iloc[0], 'o',
                        color=line[0].get_color(), label='start')
                ax.plot(df.loc[:, 'z'].iloc[-1], df.loc[:, 'y'].iloc[-1], -df.loc[:, 'x'].iloc[-1], 'x',
                        color=line[0].get_color(), label='end')

        line1 = ax.plot(mu_pos1[:, 2], mu_pos1[:, 1], -mu_pos1[:, 0], '-', color='blue', label='POP-ProMP')
        ax.plot(mu_pos1[0, 2], mu_pos1[0, 1], -mu_pos1[0, 0], 'o',
                color=line1[0].get_color(), label='start')
        ax.plot(mu_pos1[-1, 2], mu_pos1[-1, 1], -mu_pos1[-1, 0], 'x',
                color=line1[0].get_color(), label='end')

        line2 = ax.plot(mu_pos2[:, 2], mu_pos2[:, 1], -mu_pos2[:, 0], '-', color='yellow',
                        label='ProMP')
        ax.plot(mu_pos2[0, 2], mu_pos2[0, 1], -mu_pos2[0, 0], 'o',
                color=line2[0].get_color(), label='start')
        ax.plot(mu_pos2[-1, 2], mu_pos2[-1, 1], -mu_pos2[-1, 0], 'x',
                color=line2[0].get_color(), label='end')

        line3 = ax.plot(mu_pos3[:, 2], mu_pos3[:, 1], -mu_pos3[:, 0], '-', color='green',
                        label='TP-GMM')
        ax.plot(mu_pos3[0, 2], mu_pos3[0, 1], -mu_pos3[0, 0], 'o',
                color=line3[0].get_color(), label='start')
        ax.plot(mu_pos3[-1, 2], mu_pos3[-1, 1], -mu_pos3[-1, 0], 'x',
                color=line3[0].get_color(), label='end')

        # line4 = ax.plot(mu_pos4[:, 2], mu_pos4[:, 1], -mu_pos4[:, 0], '-', color='purple',
        #                 label='Naive')
        # ax.plot(mu_pos4[0, 2], mu_pos4[0, 1], -mu_pos4[0, 0], 'o',
        #         color=line4[0].get_color(), label='start')
        # ax.plot(mu_pos4[-1, 2], mu_pos4[-1, 1], -mu_pos4[-1, 0], 'x',
        #         color=line4[0].get_color(), label='end')

        ax.plot(net_position[2], net_position[1], -net_position[0], '*',
                color='gold')
        ax.text(net_position[2], net_position[1], -net_position[0], 'net')
        # ax.plot(puck_position[2], puck_position[1], -puck_position[0], '*',
        #         color='gold')
        # ax.text(puck_position[2], puck_position[1], -puck_position[0], 'puck')

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')

        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
        # ax.set_box_aspect((np.ptp(mu_pos_tp_pmp1[:,2]), np.ptp(mu_pos_tp_pmp1[:,1]), np.ptp(mu_pos_tp_pmp1[:,0])))
        handles, labels = ax.get_legend_handles_labels()
        newHandles_temp, newLabels_temp = remove_repetitive_labels(handles, labels)
        newLabels, newHandles = [], []

        for handle, label in zip(newHandles_temp, newLabels_temp):
            if label not in ['start', 'middle', 'end']:
                newLabels.append(label)
                newHandles.append(handle)
        plt.legend(newHandles, newLabels, loc='upper left',  prop={'size': 10})
        ax.view_init(elev=70,azim=270) # adjust the camera angle
        # plt.savefig('position_shooting.eps', bbox_inches='tight', format='eps')
        plt.show()



        #### Plot Orientation ##############
        # matplotlib.rcParams.update({'font.size': 18})
        # fig2, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        # for demo in train_demos + [test_demo]:
        #     quats = gripper_traj_in_obj['global'][demo][data.dims].to_numpy()[:, 3:]
        #     quats_normalized = normalize_quats(quats)
        #     if demo in train_demos:
        #         for i, ax in enumerate(axes):
        #             ax.plot(quats[:, i], '--', color='grey', label='Training demos')
        #     else:
        #         for i, ax in enumerate(axes):
        #             ax.plot(quats[:, i], '-', color='red', label='Test demo')
        #
        # # Plot pmp and gmm prediction
        # quats = mu_ori1
        # quats_normalized = normalize_quats(quats)
        # for i, ax in enumerate(axes):
        #     ax.plot(quats_normalized[:, i], '-', color='blue', label='POP-ProMP')
        #     ax.set_title(data.dims[i + 3])
        #     if i == 3:
        #         ax.set_xlabel('Time')
        #
        # quats = mu_ori2
        # quats_normalized = normalize_quats(quats)
        # for i, ax in enumerate(axes):
        #     ax.plot(quats_normalized[:, i], '-', color='yellow', label='ProMP')
        #     ax.set_title(data.dims[i + 3])
        #     if i == 3:
        #         ax.set_xlabel('Time (seconds)')
        # # plt.legend(newHandles, newLabels, loc='lower left')
        #
        # quats = mu_ori3
        # quats_normalized = normalize_quats(quats)
        # for i, ax in enumerate(axes):
        #     ax.plot(quats_normalized[:, i], '-', color='green', label='TP-GMM')
        #     ax.set_title(data.dims[i + 3])
        #     if i == 3:
        #         ax.set_xlabel('Time (seconds)')

        # quats = mu_ori4
        # quats_normalized = normalize_quats(quats)
        # for i, ax in enumerate(axes):
        #     ax.plot(quats_normalized[:, i], '-', color='purple', label='naive')
        #     ax.set_title(data.dims[i + 3])
        #     if i == 3:
        #         ax.set_xlabel('Time (seconds)')
        handles, labels = ax.get_legend_handles_labels()
        newHandles, newLabels = remove_repetitive_labels(handles, labels)
        # plt.legend(newHandles, newLabels, loc='lower left',  prop={'size': 10})
        fig.tight_layout()
        # plt.savefig('orientation_shooting.eps', bbox_inches='tight', format='eps')
        plt.show()

    print(
        f'1 : TP-ProMP: This distance at the start: {np.mean(np.array(d1_pos), axis=0)[0]} mm, end: {np.mean(np.array(d1_pos), axis=0)[1]}, average: {np.mean(np.array(d1_pos), axis=0)[-1]} mm, average quaternion: {np.mean(np.array(d1_ori))}')
    print(
        f'2 : ProMP: This distance at the start: {np.mean(np.array(d2_pos), axis=0)[0]} mm, end: {np.mean(np.array(d2_pos), axis=0)[1]},  average: {np.mean(np.array(d2_pos), axis=0)[-1]} mm, average quaternion: {np.mean(np.array(d2_ori))}')
    print(
        f'3 : TP-GMM: This distance at the start: {np.mean(np.array(d3_pos), axis=0)[0]} mm, end: {np.mean(np.array(d3_pos), axis=0)[1]}, average: {np.mean(np.array(d3_pos), axis=0)[-1]} mm, average quaternion: {np.mean(np.array(d3_ori))}')
    # print(
    #     f'3 : TP-GMM: This distance at the start: {np.mean(np.array(d4_pos), axis=0)[0]} mm, end: {np.mean(np.array(d4_pos), axis=0)[1]}, average: {np.mean(np.array(d4_pos), axis=0)[-1]} mm, average quaternion: {np.mean(np.array(d4_ori))}')
