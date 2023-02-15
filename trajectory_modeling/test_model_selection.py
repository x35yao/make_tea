import matplotlib.pyplot as plt
import yaml
import os
from process_data import Task_data
import pickle
from TP_PMP import pmp as pmp
from test import prepare_data_for_pmp, predict1,predict2, get_position_difference_per_step, prepare_data_for_pmp_all_frames
import numpy as np
import matplotlib
from outlier_detection import detect_outlier
import random
from tqdm import trange

if __name__ == '__main__':

    PLOT = True
    n_tests = 10

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
    individuals = ['cup', 'pitcher']
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
    test_demos_pool = [demo for demo in demos[0:10] if demo not in bad_demos]
    # Train model
    print(f'The number of training pool is: {len(train_demos_pool)}')
    print(f'The number of outliers is: {len(outliers)}')
    n_dims = len(data.dims)
    n_train = 12

    d1_pos, d2_pos, d3_pos, d4_pos = [], [], [], []
    max_n_components = 4
    max_iter = 50
    result1 = []
    result2 = []
    # result3 = []
    # result4 = []
    for i in trange(n_tests):
        # train_demos = train_demos_pool[:int(n_train/2)] + train_demos_pool[10:10+int(n_train/2)]
        train1 = train_demos_pool[:int(n_train / 2)]
        train2 = train_demos_pool[10:10 + int(n_train / 2)]
        train_demos = train1

        test_demos_pool_updated = [demo for demo in test_demos_pool if demo not in train_demos]
        test_demos = random.sample(test_demos_pool_updated, k=1)

        test_demo = test_demos[0]
        ground_truth = gripper_traj_in_obj['global'][test_demo][data.dims].to_numpy()
        tp_pmps = {}

        data_temp = []
        times_temp = []
        data_all_frames = {}
        for individual in sorted(individuals):
            data_frame, times = prepare_data_for_pmp(gripper_traj_in_generalized_obj, individual, train_demos, data.dims)
            data_all_frames[individual] = data_frame
        temp1 = []
        temp2 = []
        # temp3 = []
        # temp4 = []

        for n in range(max_n_components):
            n_components = n + 1
            model_pmp = pmp.PMP(data_all_frames, times, n_dims, sigma=0.035, n_components=n_components, covariance_type = 'diag',  max_iter = max_iter)
            model_pmp.train(print_lowerbound=False)
            # model_pmp.refine()
            tp_pmps = model_pmp.model
            temp1.append(tp_pmps.bic(with_prior = False))
            temp2.append(tp_pmps.bic(with_prior=True))
            # temp3.append(tp_pmps.aic(with_prior=False))
            # temp4.append(tp_pmps.aic(with_prior=True))
        result1.append(temp1)
        result2.append(temp2)
        # result3.append(temp3)
        # result4.append(temp4)
    result_mean1 = np.mean(result1, axis = 0)
    result_mean2 = np.mean(result2, axis=0)
    plt.figure()
    plt.plot(result_mean1, label = 'bic, no prior')
    plt.plot(result_mean2, label = 'bic, with prior')
    plt.legend()
    plt.show()
    print(result_mean1)
    print(result_mean2)


