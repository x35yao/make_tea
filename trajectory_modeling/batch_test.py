from utils import *
from TP_PMP import pmp
from test import *
import pickle
import TP_GMM.gmm as gmm
from tqdm import trange

if __name__ == '__main__':
    # Load data
    project_dir = '/home/luke/Desktop/project/make_tea/'  # Modify this to your need
    base_dir = os.path.join(project_dir, 'Process_data/postprocessed/2022-08-(17-21)')
    template_dir = os.path.join(project_dir, 'Process_data/postprocessed/2022-08-(17-21)/transformations/dlc3d')
    individuals = ['teabag1', 'teabag2', 'pitcher', 'cup', 'tap']  # The objects that we will place a reference frame on
    objs = ['teabag', 'pitcher', 'cup', 'tap']
    d = Task_data(base_dir, template_dir, individuals, objs)
    gripper_trajs_full = d.load_gripper_trajectories()
    n_actions = d.get_number_of_actions()
    gripper_trajs_truncated = d.get_gripper_trajectories_for_each_action()
    # Load data
    with open(os.path.join(base_dir, 'processed', 'gripper_trajs_in_obj_aligned_filtered.pickle',), 'rb') as f1:
        gripper_trajs_in_obj = pickle.load(f1)
    with open(os.path.join(base_dir, 'processed', 'HTs_obj_in_ndi.pickle',), 'rb') as f2:
        HTs_obj_in_ndi = pickle.load(f2)

    # Train model
    n_train = 6
    n_dims = len(d.dims)
    # bad_demos = ['463678', '636936', '636938', '463675']
    bad_demos = ['740521', '506373', '506365']
    demos = [demo for demo in d.demos if demo not in bad_demos]
    n_test = 1

    for i in range(n_actions):
        print(f'Testing the {i + 1}th. action')
        gripper_trajs_in_obj_action = gripper_trajs_in_obj[i]
        gripper_traj_in_ndi = gripper_trajs_truncated[i]
        HT_obj_in_ndi = HTs_obj_in_ndi[i]
        for j in trange(n_test):
            dists_gmm = []
            dists_pmp = []
            train_demos = random.sample(demos, k=n_train)
            test_demos = [demo for demo in demos if demo not in train_demos and demo not in bad_demos]
            test_demo = random.sample(test_demos, k=1)[0]
            gmms = {}
            pmps = {}

            for individual in individuals:
                data_pmp, times = prepare_data_for_pmp(gripper_trajs_in_obj_action, individual, train_demos, d.dims)
                data_gmm = prepare_data_for_gmm(gripper_trajs_in_obj_action, individual, train_demos, ['Time'] + d.dims)
                model_gmm = gmm.GMM(nb_states=15, nb_dim=n_dims + 1)
                model_gmm.em(data_gmm, reg=1e-3, maxiter=100)
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

            ground_truth_pos = ground_truth[:, :3]
            ground_truth_ori = ground_truth[:, 3:]
            gmm_prediction_pos = mu_mean_gmm[:, :3]
            gmm_prediction_ori = mu_mean_gmm[:, 3:]
            pmp_prediction_pos = mu_mean_pmp[:, :3]
            pmp_prediction_ori = mu_mean_pmp[:, 3:]

            mid_ind = int(0.8 * len(ground_truth))

            # dist_start_gmm = np.linalg.norm(ground_truth_pos[0] - gmm_prediction_pos[0])
            # dist_mid_gmm = np.linalg.norm(ground_truth_pos[mid_ind] - gmm_prediction_pos[mid_ind])
            # dist_end_gmm = np.linalg.norm(ground_truth_pos[-1] - gmm_prediction_pos[-1])
            dist_gmm = get_position_difference_per_step(ground_truth_pos, gmm_prediction_pos)
            dist_start_gmm = dist_gmm[0]
            dist_mid_gmm = dist_gmm[mid_ind]
            dist_end_gmm = dist_gmm[-1]
            dist_average_gmm = np.sum(dist_gmm)
            dists_gmm.append([dist_average_gmm, dist_start_gmm, dist_mid_gmm, dist_end_gmm])

            # dist_start_pmp = np.linalg.norm(ground_truth_pos[0] - pmp_prediction_pos[0])
            # dist_mid_pmp = np.linalg.norm(ground_truth_pos[mid_ind] - pmp_prediction_pos[mid_ind])
            # dist_end_pmp = np.linalg.norm(ground_truth_pos[-1] - pmp_prediction_pos[-1])
            dist_pmp = get_position_difference_per_step(ground_truth_pos, pmp_prediction_pos)
            dist_start_pmp = dist_pmp[0]
            dist_mid_pmp = dist_pmp[mid_ind]
            dist_end_pmp = dist_pmp[-1]
            dist_average_pmp = np.sum(dist_pmp)
            dists_pmp.append([dist_average_pmp, dist_start_pmp, dist_mid_pmp, dist_end_pmp])

        dists_gmm_average = np.mean(np.array(dists_gmm), axis = 0)
        dists_pmp_average = np.mean(np.array(dists_pmp), axis = 0)
        print(f'The average distances of gmm model for {i + 1}th. action are: {dists_gmm_average}')
        print(f'The average distances of pmp model for {i + 1}th. action are: {dists_pmp_average}')

    #     dists_gmm_all_actions.append(dists_gmm_average)
    #     dists_pmp_all_actions.append(dists_pmp_average)
    #
    # for i in range(n_actions):
    #     print(f'The average distances of gmm model for {i}th. action are: {dists_gmm_all_actions[i]}')
    #     print(f'The average distances of pmp model for {i}th. action are: {dists_pmp_all_actions[i]}')