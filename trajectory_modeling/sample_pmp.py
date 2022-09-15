from utils import *
from TP_PMP import utils
from TP_PMP.pmp import PMP
from plot import plot_position
from matplotlib import pyplot as plt
import random
from test import predict
import pickle
import yaml

if __name__ == '__main__':
    # Load data
    with open('../task_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    base_dir = os.path.join(config["project_path"], config["postprocessed_dir"])
    with open(os.path.join(base_dir, 'processed', 'gripper_trajs_in_obj_aligned_filtered.pickle', ), 'rb') as f1:
        gripper_trajs_in_obj = pickle.load(f1)
    with open(os.path.join(base_dir, 'processed', 'HTs_obj_in_ndi.pickle', ), 'rb') as f2:
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
    dims = [ 'x', 'y', 'z']
    individuals = config["individuals"]
    n_dim = len(dims)
    bad_demos = ['463678', '636936', '636938', '463675']
    demos = [demo for demo in demos if demo not in bad_demos]
    train_demos = random.sample(demos, k=n_train)
    test_demo = [demo for demo in demos if demo not in train_demos and demo not in bad_demos][0]
    gripper_trajs_in_obj_train = {individual: {'pose': [], 'time': []} for individual in individuals}
    for individual in individuals:
        for d in train_demos:
            t = gripper_traj_in_obj[individual][d]['Time'].to_numpy().flatten()
            t = t / t[-1]
            gripper_trajs_in_obj_train[individual]['pose'].append(
                gripper_traj_in_obj[individual][d].loc[:, dims].to_numpy())
            gripper_trajs_in_obj_train[individual]['time'].append(t)

    dof = int(len(dims))
    dim_basis_fun = 30
    inv_whis_mean = lambda v, Sigma: 9e-1 * utils.make_block_diag(Sigma, dof) + 1e-1 * np.eye(dof * dim_basis_fun)
    prior_Sigma_w = {'v': dim_basis_fun * dof, 'mean_cov_mle': inv_whis_mean}

    # Every object has one pmp model
    pmps = {}
    for individual in individuals:
        Q = gripper_trajs_in_obj_train[individual]['pose']
        times = gripper_trajs_in_obj_train[individual]['time']
        model_pmp = PMP(Q, times, 3)
        model_pmp.train()
        pmps[individual] = model_pmp

    ### Test on test traj
    ground_truth = gripper_traj_in_ndi[test_demo][dims].to_numpy()
    t_pmp = np.linspace(0, 1, ground_truth.shape[0])

    mu_mean_pmp, sigma_mean_pmp = predict(pmps, t_pmp, test_demo, HT_obj_in_ndi, individuals)

    mid = 0.62
    fig = plt.figure(figsize=(12, 10))
    ax2 = fig.add_subplot(1, 1, 1, projection='3d')
    plot_position(ax2, mu_mean_pmp[:, :3], ground_truth[:, :3], mid,
                  title=f'PMP position prediction for demo {test_demo}')
    fig.legend()
    plt.show()