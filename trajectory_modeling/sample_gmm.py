
from utils import *
import TP_GMM.gmm as gmm
from TP_GMM.plot import plot_gmm
from matplotlib import pyplot as plt
import random
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
    dims = ['Time', 'x', 'y', 'z']
    n_dim = len(dims)
    bad_demos = ['463678', '636936', '636938', '463675']
    demos = [demo for demo in demos if demo not in bad_demos]
    train_demos = random.sample(demos, k=n_train)
    test_demo = [demo for demo in demos if demo not in train_demos and demo not in bad_demos][0]
    gmms = []
    demos_x_f = []
    individuals = ['cup', 'teabag1', 'tap']
    for individual in individuals:
        temp = []
        for d in train_demos:
            data_demo = gripper_traj_in_obj[individual][d][dims].to_numpy()
            temp.append(data_demo)
        demos_x_f.append(np.array(temp))
        data = np.concatenate(temp)
        model_gmm = gmm.GMM(nb_states = 17, nb_dim = n_dim)
        model_gmm.em(data, reg=1e-3, maxiter = 1000)
        gmms.append(model_gmm)

    fig, ax = plt.subplots(nrows=len(gmms))
    fig.set_size_inches(5, 15)

    for j, individual in enumerate(individuals):
        # position plotting
        ax[j].set_title(f'pos - coord in {individual} frame')
        for k, p in enumerate(demos_x_f[j]):
            ax[j].plot(p[:, 2], p[:, 3])
        model = gmms[j]
        plot_gmm(model.mu[:, 2:4], model.sigma[:, 2:4, 2:4], ax=ax[j], color='orangered');
    plt.tight_layout()
    plt.show()