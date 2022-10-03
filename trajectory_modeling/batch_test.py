from utils import *
from TP_PMP import pmp
from test import *
import pickle
import TP_GMM.gmm as gmm
import os
import yaml


if __name__ == '__main__':
    # read config file
    with open('../task_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    project_dir = config["project_path"]  # Modify this to your need
    base_dir = os.path.join(project_dir, config["postprocessed_dir"])
    template_dir = os.path.join(project_dir, config["postprocessed_dir"], 'transformations/dlc3d')
    individuals = config["individuals"]  # The objects that we will place a reference frame on
    objs = config["objects"]
    d = Task_data(base_dir, template_dir, individuals, objs)
    gripper_trajs_truncated = d.get_gripper_trajectories_for_each_action()
    # Load data
    base_dir = os.path.join(config["project_path"], config["postprocessed_dir"])
    with open(os.path.join(base_dir, 'processed', 'gripper_trajs_in_obj_aligned_filtered.pickle',), 'rb') as f1:
        gripper_trajs_in_obj = pickle.load(f1)
    with open(os.path.join(base_dir, 'processed', 'HTs_obj_in_ndi.pickle',), 'rb') as f2:
        HTs_obj_in_ndi = pickle.load(f2)

    n_train = 6
    n_dims = len(d.dims)
    bad_demos = ['740521', '506373', '506365', '648027', '781806', '318539']
    demos = [demo for demo in d.demos if demo not in bad_demos]
    n_test = 10
    n_actions = d.get_number_of_actions()
    for i in range(n_actions):
        if i!= 1:
            continue
        print(f'Testing the {i + 1}th. action')
        gripper_traj_in_ndi = gripper_trajs_truncated[i]
        gripper_traj_in_obj = gripper_trajs_in_obj[i]
        HT_obj_in_ndi = HTs_obj_in_ndi[i]

        pos_dists_gmm = []
        pos_dists_pmp = []
        ori_dists_gmm = []
        ori_dists_pmp = []
        # Train model
        for j in range(n_test):
            train_demos = random.sample(demos, k=n_train)
            test_demos = [demo for demo in demos if demo not in train_demos and demo not in bad_demos]
            test_demo = random.sample(test_demos, k=1)[0]
            gmms = {}
            pmps = {}
            for individual in d.individuals:
                data_pmp, times = prepare_data_for_pmp(gripper_traj_in_obj, individual, train_demos, d.dims)
                data_gmm = prepare_data_for_gmm(gripper_traj_in_obj, individual, train_demos, ['Time'] + d.dims)
                n_states = 15
                n_data = data_gmm.shape[0]
                model_gmm = gmm.GMM(nb_states=n_states, nb_dim=n_dims + 1)
                model_gmm.em(data_gmm, reg=1e-3, maxiter=100, verbose=False)
                gmms[individual] = model_gmm
                model_pmp = pmp.PMP(data_pmp, times, n_dims)
                model_pmp.train()
                pmps[individual] = model_pmp
            ### Test on test traj
            ground_truth = gripper_traj_in_ndi[test_demo][d.dims].to_numpy()
            t_pmp = np.linspace(0, 1, ground_truth.shape[0])
            t_gmm = gripper_traj_in_ndi[test_demo]['Time'].to_numpy()
            mu_mean_gmm, sigma_mean_gmm = predict(gmms, t_gmm, test_demo, HT_obj_in_ndi, individuals)
            mu_mean_pmp, sigma_mean_pmp = predict(pmps, t_pmp, test_demo, HT_obj_in_ndi, individuals)

            ### Test on test traj
            ground_truth = gripper_traj_in_ndi[test_demo][d.dims].to_numpy()
            t_pmp = np.linspace(0, 1, ground_truth.shape[0])
            t_gmm = gripper_traj_in_ndi[test_demo]['Time'].to_numpy()

            mu_mean_gmm, sigma_mean_gmm = predict(gmms, t_gmm, test_demo, HT_obj_in_ndi, individuals)
            mu_mean_pmp, sigma_mean_pmp = predict(pmps, t_pmp, test_demo, HT_obj_in_ndi, individuals)

            ground_truth_pos = ground_truth[:, :3]
            ground_truth_ori = ground_truth[:, 3:]
            gmm_prediction_pos = mu_mean_gmm[:, :3]
            gmm_prediction_ori = mu_mean_gmm[:, 3:]
            pmp_prediction_pos = mu_mean_pmp[:, :3]
            pmp_prediction_ori = mu_mean_pmp[:, 3:]

            mid_ind = int(0.8 * len(ground_truth))

            dist_gmm = get_position_difference_per_step(ground_truth_pos, gmm_prediction_pos)
            dist_start_gmm = dist_gmm[0]
            dist_mid_gmm = dist_gmm[mid_ind]
            dist_end_gmm = dist_gmm[-1]
            dist_average_gmm = np.sum(dist_gmm)
            pos_dists_gmm.append([dist_average_gmm, dist_start_gmm, dist_mid_gmm, dist_end_gmm])

            dist_pmp = get_position_difference_per_step(ground_truth_pos, pmp_prediction_pos)
            dist_start_pmp = dist_pmp[0]
            dist_mid_pmp = dist_pmp[mid_ind]
            dist_end_pmp = dist_pmp[-1]
            dist_average_pmp = np.sum(dist_pmp)
            pos_dists_pmp.append([dist_average_pmp, dist_start_pmp, dist_mid_pmp, dist_end_pmp])

            ori_dist_gmm = np.mean(get_orientation_difference_per_step(ground_truth_ori, gmm_prediction_ori))
            ori_dists_gmm.append(ori_dist_gmm)

            ori_dist_pmp = np.mean(get_orientation_difference_per_step(ground_truth_ori, pmp_prediction_ori))
            ori_dists_pmp.append(ori_dist_pmp)

        pos_dists_gmm_average = np.mean(np.array(pos_dists_gmm), axis = 0)
        pos_dists_pmp_average = np.mean(np.array(pos_dists_pmp), axis = 0)
        ori_dists_gmm_average = np.mean(np.array(ori_dist_gmm))
        ori_dists_pmp_average = np.mean(np.array(ori_dist_pmp))
        print(f'The average position distances of gmm model for {i + 1}th. action are: {pos_dists_gmm_average}')
        print(f'The average position distances of pmp model for {i + 1}th. action are: {pos_dists_pmp_average}')
        print(f'The average orientation distances of gmm model for {i + 1}th. action are: {ori_dists_gmm_average}')
        print(f'The average orientation distances of pmp model for {i + 1}th. action are: {ori_dists_pmp_average}')
