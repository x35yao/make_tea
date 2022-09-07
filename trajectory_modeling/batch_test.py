from utils import *
from TP_PMP import pmp
from test import *
import pickle
import TP_GMM.gmm as gmm


if __name__ == '__main__':
    # Load data
    base_dir = '/home/luke/Desktop/project/make_tea/Process_data/postprocessed/2022-05-26'
    with open(os.path.join(base_dir, 'processed', 'gripper_trajs_in_obj_aligned_filtered.pickle',), 'rb') as f1:
        gripper_trajs_in_obj = pickle.load(f1)
    with open(os.path.join(base_dir, 'processed', 'HTs_obj_in_ndi.pickle',), 'rb') as f2:
        HTs_obj_in_ndi = pickle.load(f2)

    gripper_trajs_full = load_gripper_trajectories(base_dir)
    n_actions = get_number_of_actions(base_dir)
    gripper_trajs_truncated = get_gripper_trajectories_for_each_action(base_dir, gripper_trajs_full, n_actions)
    ind = 0
    gripper_traj_in_ndi = gripper_trajs_truncated[ind]

    gripper_traj_in_obj = gripper_trajs_in_obj[ind]
    HT_obj_in_ndi = HTs_obj_in_ndi[ind]
    demos = gripper_traj_in_ndi.keys()

    # Train model
    n_train = 6
    dims= [ 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    n_dims = len(dims)
    individuals = ['cup', 'teabag1', 'tap', 'teabag2', 'pitcher']
    bad_demos = ['463678', '636936', '636938', '463675']
    demos = [demo for demo in demos if demo not in bad_demos]
    dists_gmm = []
    dists_pmp = []
    for i, demo in enumerate(demos):
        test_demo = demo
        train_demos = [d for d in demos if d != demo]

        gmms = {}
        pmps = {}

        for individual in individuals:
            data_pmp, times = prepare_data_for_pmp(gripper_traj_in_obj, individual, train_demos, dims)
            data_gmm = prepare_data_for_gmm(gripper_traj_in_obj, individual, train_demos, ['Time'] + dims)
            model_gmm = gmm.GMM(nb_states=15, nb_dim=n_dims + 1)
            model_gmm.em(data_gmm, reg=1e-3, maxiter=100)
            gmms[individual] = model_gmm

            model_pmp = pmp.PMP(data_pmp, times, n_dims)
            model_pmp.train()
            pmps[individual] = model_pmp

        ### Test on test traj
        ground_truth = gripper_traj_in_ndi[test_demo][dims].to_numpy()
        t_pmp = np.linspace(0, 1, ground_truth.shape[0])
        t_gmm = gripper_traj_in_ndi[test_demo]['Time'].to_numpy()

        mu_mean_gmm, sigma_mean_gmm = predict(gmms, t_gmm, test_demo, HT_obj_in_ndi, individuals)
        mu_mean_pmp, sigma_mean_pmp = predict(pmps, t_pmp, test_demo, HT_obj_in_ndi, individuals)

        mid_ind = int(0.62 * len(ground_truth))

        dist_start_gmm = np.linalg.norm(ground_truth[0, :3] - mu_mean_gmm[0, :3])
        dist_mid_gmm = np.linalg.norm(ground_truth[mid_ind, :3] - mu_mean_gmm[mid_ind, :3])
        dist_end_gmm = np.linalg.norm(ground_truth[-1, :3] - mu_mean_gmm[-1, :3])
        dists_gmm.append([dist_start_gmm, dist_mid_gmm, dist_end_gmm])

        dist_start_pmp = np.linalg.norm(ground_truth[0, :3] - mu_mean_pmp[0, :3])
        dist_mid_pmp = np.linalg.norm(ground_truth[mid_ind, :3] - mu_mean_pmp[mid_ind, :3])
        dist_end_pmp = np.linalg.norm(ground_truth[-1, :3] - mu_mean_pmp[-1, :3])
        dists_pmp.append([dist_start_pmp, dist_mid_pmp, dist_end_pmp])
    dists_gmm = np.array(dists_gmm)
    dists_pmp = np.array(dists_pmp)
    print(f'GMM average distances are {np.mean(dists_gmm, axis = 0)}')
    print(f'PMP average distances are {np.mean(dists_pmp, axis=0)}')