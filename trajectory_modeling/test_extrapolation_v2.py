import matplotlib.pyplot as plt
import yaml
import os
from process_data import Task_data
import pickle
from TP_PMP import pmp
from TP_GMM import gmm
from naive import naive
from test import prepare_data_for_pmp,prepare_data_for_gmm, predict2, predict1,get_position_difference_per_step
import numpy as np
import matplotlib
from plot import remove_repetitive_labels
from outlier_detection import detect_outlier
import random

if __name__ == '__main__':
    PLOT = True

    font = {'size': 14}
    n_tests = 1
    matplotlib.rc('font', **font)

    # Load data
    task_config_dir = '../Process_data/postprocessed/2022-10-06'
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
    data.dims = ['x', 'y', 'z']
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
    # individuals = [ind for ind in list(HTs_generalized_obj_in_ndi.keys()) if ind != 'global']
    individuals= ['cup', 'teabag1']
    # individuals = ['cup-teabag1', 'teabag1-cup']

    for individual in individuals:
        n_std = std_thres
        outlier_individual = detect_outlier(gripper_traj_in_generalized_obj[individual], n=n_std)
        print(f'The outliers for individual {individual} are {outlier_individual}')
        # print(f'The outliers for individual {individual} are {outlier_individual}')
        outliers += outlier_individual
    outliers = list(set(outliers))
    bad_demos = outliers

    demos = sorted(list(HTs_obj_in_ndi['global'].keys()))

    train_demos_pool = [demo for demo in demos[:20] if demo not in bad_demos]
    test_demos_pool = [demo for demo in demos[20:] if demo not in bad_demos]
    # Train model
    print(f'The number of training pool is: {len(train_demos_pool)}')
    print(f'The number of outliers is: {len(outliers)}')
    n_dims = len(data.dims)
    n_train = 6
    d1, d2, d3, d4 = [],[],[],[]
    for i in range(n_tests):
        train_demos = random.sample(train_demos_pool, k=n_train)
        test_demos_pool_updated = [demo for demo in test_demos_pool if demo not in train_demos]
        test_demos = random.sample(test_demos_pool_updated, k=1)
        # test_demos = ['134776']
        test_demo = test_demos[0]
        # if test_demo in train_demos:
        #     raise
        ground_truth = gripper_trajs_in_ndi[test_demo][data.dims].to_numpy()
        t_pmp = np.linspace(0, 1, ground_truth.shape[0])
        t_gmm = gripper_trajs_in_ndi[test_demo]['Time'].to_numpy()

        data_temp = []
        times_temp = []
        data_all_frames_tp_pmp = {}
        data_all_frames_pmp = {}
        for individual in sorted(individuals):
            data_per_frame_tp_pmp, times = prepare_data_for_pmp(gripper_traj_in_generalized_obj, individual,
                                                                train_demos,
                                                                data.dims)
            data_all_frames_tp_pmp[individual] = data_per_frame_tp_pmp

        data_per_frame_pmp, times = prepare_data_for_pmp(gripper_traj_in_generalized_obj, 'global', train_demos,
                                                         data.dims)
        data_all_frames_pmp['global'] = data_per_frame_pmp

        n_components = 1
        max_iter = 30

        ### TP-ProMP model
        model_tp_pmp = pmp.PMP(data_all_frames_tp_pmp, times, n_dims, sigma=0.4, n_components=n_components,
                            covariance_type='diag', max_iter= max_iter, gmm = False)
        model_tp_pmp.train(print_lowerbound=False)
        mu_mean_tp_pmp, sigma_mean_tp_pmp = predict2(model_tp_pmp, t_pmp, test_demo, HTs_generalized_obj_in_ndi,
                                                     individuals, n_dim=n_dims, mode_selected=0)
        mu_pos1 = mu_mean_tp_pmp
        d = get_position_difference_per_step(ground_truth, mu_pos1)
        d1.append([d[0], d[-1], np.mean(d)])

        ###ProMP model
        model_pmp = pmp.PMP(data_all_frames_pmp, times, n_dims, sigma=0.4, n_components=n_components,
                               covariance_type='diag', max_iter=max_iter, gmm = False)
        model_pmp.train(print_lowerbound=False)

        H_cup_in_ndi = HTs_generalized_obj_in_ndi['cup'][test_demo]
        H_teabag_in_ndi = HTs_generalized_obj_in_ndi['teabag1'][test_demo]
        model_pmp.condition(1, 1, q=H_cup_in_ndi[:-1, 3] + np.array([-20, 0, 0]), ignore_Sy=False)
        model_pmp.condition(0, 1, q=H_teabag_in_ndi[:-1, 3] + np.array([-20, 0, 0]), ignore_Sy=False)
        mu_mean_pmp, sigma_mean_pmp = model_pmp.pmp.marginal_w(t_pmp)
        mu_pos2 = np.array(mu_mean_pmp)
        d = get_position_difference_per_step(ground_truth, mu_pos2)
        d2.append([d[0], d[-1], np.mean(d)])

        # ### TP-GMM model
        max_nb_states = 6
        average_dist_min = np.inf
        for i in range(max_nb_states):
            nb_states = i + 2
            # print(nb_states)
            tp_gmms = {}
            for individual in sorted(individuals):
                data_per_frame_gmm = prepare_data_for_gmm(gripper_traj_in_generalized_obj, individual, train_demos,
                                                          data.dims)
                model_tp_gmm = gmm.GMM(nb_states=nb_states, nb_dim=data_per_frame_gmm.shape[1])
                model_tp_gmm.em(data_per_frame_gmm, reg=1e-8, maxiter=200, verbose=False)
                tp_gmms[individual] = model_tp_gmm

            mu_mean_tp_gmm, sigma_mean_tp_gmm = predict1(tp_gmms, t_gmm, test_demo, HTs_generalized_obj_in_ndi,
                                                         individuals)
            d = get_position_difference_per_step(ground_truth[:, :3], mu_mean_tp_gmm[:, :3])
            d_mean = np.mean(d)
            if d_mean < average_dist_min:
                average_dist_min = d_mean
                pos_temp = [d[0], d[1], d_mean]
                mu_pos3 = mu_mean_tp_gmm[:, :3]
                mu_ori3 = mu_mean_tp_gmm[:, 3:]
        d3.append(pos_temp)

        ### Naive model
        naives = {}
        for individual in sorted(individuals):
            data_per_frame_naive, times = prepare_data_for_pmp(gripper_traj_in_generalized_obj, individual,
                                                               train_demos,
                                                               data.dims)
            naive_model = naive.Naive()
            naive_model.train(data_per_frame_naive, times)
            naives[individual] = naive_model
        mu_mean_naive, sigma_mean_naive = predict1(naives, t_pmp, test_demo, HTs_generalized_obj_in_ndi,
                                                   individuals)
        mu_pos4 = mu_mean_naive
        d = get_position_difference_per_step(ground_truth, mu_pos4)
        d4.append([d[0], d[-1], np.mean(d)])

        if PLOT:
            matplotlib.rcParams.update({'font.size': 10})
            fig = plt.figure(figsize = (18, 6))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_facecolor('white')
            ax.locator_params(nbins=3, axis='z')
            for demo in train_demos + [test_demo]:
                df = gripper_trajs_in_ndi[demo]
                if demo in train_demos and demo not in [test_demo]:
                    line = ax.plot(df.loc[:, 'z'], df.loc[:, 'y'], -df.loc[:, 'x'], '--', color = 'grey', label = 'Training demos')
                    ax.plot(df.loc[:, 'z'].iloc[0], df.loc[:, 'y'].iloc[0], -df.loc[:, 'x'].iloc[0], 'o',
                            color='grey', label='start')
                    ax.plot(df.loc[:, 'z'].iloc[-1], df.loc[:, 'y'].iloc[-1], -df.loc[:, 'x'].iloc[-1], 'x',
                            color='grey', label='end')
                else:
                    line = ax.plot(df.loc[:, 'z'], df.loc[:, 'y'], -df.loc[:, 'x'], '-', color='red', label='Test demo')
                    ax.plot(df.loc[:, 'z'].iloc[0], df.loc[:, 'y'].iloc[0], -df.loc[:, 'x'].iloc[0], 'o',
                            color= line[0].get_color(), label='start')
                    ax.plot(df.loc[:, 'z'].iloc[-1], df.loc[:, 'y'].iloc[-1], -df.loc[:, 'x'].iloc[-1], 'x',
                            color=line[0].get_color(), label='end')


            line1 = ax.plot(mu_pos1[:, 2], mu_pos1[:, 1] , -mu_pos1[:, 0], '-', color='blue', label='TP-ProMP')
            ax.plot(mu_pos1[0, 2], mu_pos1[0, 1] , -mu_pos1[0, 0], 'o',
                    color= line1[0].get_color(), label='start')
            ax.plot(mu_pos1[-1, 2], mu_pos1[-1, 1] , -mu_pos1[-1, 0], 'x',
                    color= line1[0].get_color(), label='end')

            line2 = ax.plot(mu_pos2[:, 2], mu_pos2[:, 1], -mu_pos2[:, 0], '-', color='yellow', label='ProMP')
            ax.plot(mu_pos2[0, 2], mu_pos2[0, 1], -mu_pos2[0, 0], 'o',
                    color=line2[0].get_color(), label='start')
            ax.plot(mu_pos2[-1, 2], mu_pos2[-1, 1], -mu_pos2[-1, 0], 'x',
                    color=line2[0].get_color(), label='end')
            #
            line3 = ax.plot(mu_pos3[:, 2], mu_pos3[:, 1], -mu_pos3[:, 0], '-', color='green',
                           label='TP-GMM')
            ax.plot(mu_pos3[0, 2], mu_pos3[0, 1], -mu_pos3[0, 0], 'o',
                    color=line3[0].get_color(), label='start')
            ax.plot(mu_pos3[-1, 2], mu_pos3[-1, 1], -mu_pos3[-1, 0], 'x',
                    color=line3[0].get_color(), label='end')

            line4 = ax.plot(mu_pos4[:, 2], mu_pos4[:, 1], -mu_pos4[:, 0], '-', color='purple',
                            label='Naive')
            ax.plot(mu_pos4[0, 2], mu_pos4[0, 1], -mu_pos4[0, 0], 'o',
                    color=line4[0].get_color(), label='start')
            ax.plot(mu_pos4[-1, 2], mu_pos4[-1, 1], -mu_pos4[-1, 0], 'x',
                    color=line4[0].get_color(), label='end')

            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_zlabel('z (mm)')
            ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

            handles, labels = ax.get_legend_handles_labels()
            newHandles_temp, newLabels_temp = remove_repetitive_labels(handles, labels)
            newLabels, newHandles = [], []

            for handle, label in zip(newHandles_temp, newLabels_temp):
                if label not in ['start', 'middle', 'end']:
                    newLabels.append(label)
                    newHandles.append(handle)
            plt.legend(newHandles, newLabels, loc = 'upper left',  prop={'size': 10})
            plt.savefig('extrapolation.eps', bbox_inches='tight', format='eps')
            plt.show()
    print(np.array(d1))
    print(f'TP-ProMP : This distance at the start: {np.mean(np.array(d1), axis = 0)[0]} mm, end: {np.mean(np.array(d1), axis = 0)[1]} mm, average: {np.mean(np.array(d1), axis = 0)[-1]}')
    print(
        f'ProMP : This distance at the start: {np.mean(np.array(d2), axis=0)[0]} mm, end: {np.mean(np.array(d2), axis=0)[1]} mm, average: {np.mean(np.array(d2), axis = 0)[-1]}')
    print(
        f'TP-GMM : This distance at the start: {np.mean(np.array(d3), axis=0)[0]} mm, end: {np.mean(np.array(d3), axis=0)[1]} mm, average: {np.mean(np.array(d3), axis = 0)[-1]}')
    print(
        f'Naive : This distance at the start: {np.mean(np.array(d4), axis=0)[0]} mm, end: {np.mean(np.array(d4), axis=0)[1]} mm, average: {np.mean(np.array(d4), axis=0)[-1]}')
