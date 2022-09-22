from outlier_detection import *
import yaml
from test import *
from math import ceil
from quaternion_metric import *
from sklearn.preprocessing import normalize
from statistics import mean

def plot_orientation( prediction, ground_truth , axes):
    '''
    This function will plot the ground truth orientation and the predicted orientation.

    Parameters
    ----------
    prediction: array
        N by 4 array, where N is the number of datapoints. The 4 dimensions are for the quaternions.
    ground_truth: array
        N by 4 array, where N is the number of datapoints. The 4 dimensions are for the quaternions.
    axes: list
        axes that will be used to plot the quaternions.


    '''
    prediction_normalized = normalize(prediction, axis=1)
    q = ['qx', 'qy', 'qz', 'qw']
    for i, ax in enumerate(axes):
        ax.plot(ground_truth[:, i], 'r', label='ground_truth')
        ax.plot(prediction[:, i], 'b', label='prediction')
        ax.plot(prediction_normalized[:, i], 'g', label = 'prediction_normalized')
        # ax.set_ylim(-1, 1)
        ax.set_title(f'{q[i]}')

if __name__ =='__main__':
    # load config
    with open('../task_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    project_dir = config["project_path"]  # Modify this to your need
    base_dir = os.path.join(project_dir, config["postprocessed_dir"])
    template_dir = os.path.join(project_dir, config["postprocessed_dir"], 'transformations/dlc3d')
    individuals = config["individuals"]  # The objects that we will place a reference frame on
    objs = config["objects"]
    d = Task_data(base_dir, template_dir, individuals, objs)
    ripper_trajs_full = d.load_gripper_trajectories()
    n_actions = d.get_number_of_actions()
    gripper_trajs_truncated = d.get_gripper_trajectories_for_each_action()
    n_dims = len(d.dims)
    # load trajectory data
    with open(os.path.join(base_dir, 'processed', 'gripper_trajs_in_obj_aligned_filtered.pickle',), 'rb') as f1:
        gripper_trajs_in_obj = pickle.load(f1)
    with open(os.path.join(base_dir, 'processed', 'HTs_obj_in_ndi.pickle',), 'rb') as f2:
        HTs_obj_in_ndi = pickle.load(f2)

    ind = 1 # index of action to be tested
    gripper_traj_in_ndi = gripper_trajs_truncated[ind]

    gripper_traj_in_obj = gripper_trajs_in_obj[ind]
    HT_obj_in_ndi = HTs_obj_in_ndi[ind]
    demos = gripper_traj_in_ndi.keys()
    # collect and delete all the outlier trajectories.
    bad_demos = set()
    for i in range(n_actions):
        gripper_traj_in_obj = gripper_trajs_in_obj[i]
        outliers = set()
        for obj_temp in config['individuals']:
            obj_outliers = detect_outlier(gripper_traj_in_obj[obj_temp], n=2.9,
                                          dims=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
            outliers.update(obj_outliers)
        bad_demos.update(outliers)
    demos = [demo for demo in demos if demo not in bad_demos]
    random.shuffle(demos)
    pmp_score = []
    gmm_score = []
    test_size = ceil(len(demos)*0.2)
    iterations = int(len(demos)/test_size)

    # do training and testing data split
    gripper_traj_in_obj = gripper_trajs_in_obj[-1]
    for it in range(1):
        # Train model
        test_demos = demos[it:it+2]
        train_demos = [d for d in demos if d not in test_demos]

        gmms = {}
        pmps = {}

        for individual in d.individuals:
            data_pmp, times = prepare_data_for_pmp(gripper_traj_in_obj, individual, train_demos, d.dims)
            data_gmm = prepare_data_for_gmm(gripper_traj_in_obj, individual, train_demos, ['Time'] + d.dims)
            model_gmm = gmm.GMM(nb_states=15, nb_dim=n_dims + 1)
            model_gmm.em(data_gmm, reg=1e-3, maxiter=200, verbose=True)
            gmms[individual] = model_gmm

            model_pmp = pmp.PMP(data_pmp, times, n_dims)
            model_pmp.train()
            pmps[individual] = model_pmp

        temp_pmp_score = []
        temp_gmm_score = []
        for test_demo in test_demos:
            ground_truth = gripper_traj_in_obj['cup'][test_demo][d.dims].to_numpy()
            t_pmp = np.linspace(0, 1, ground_truth.shape[0])
            t_gmm = gripper_traj_in_obj['cup'][test_demo]['Time'].to_numpy()
            gmm_mu, gmm_sigma = marginal_t(gmms['cup'], t_gmm)
            pmp_mu, pmp_sigma = marginal_t(pmps['cup'], t_pmp)
            gmm_score = gmm_score + [inner_prod_quat(np.array(qua[3:]), ground_truth[j, 3:]) for j, qua in enumerate(gmm_mu)]
            pmp_score = pmp_score + [inner_prod_quat(np.array(qua[3:]), ground_truth[j, 3:]) for j, qua in enumerate(pmp_mu)]
            fig3, axes = plt.subplots(4, 1, figsize=(10, 10))
            fig3.suptitle(f'$Test Demo: {test_demo} --- Model: TP-GMM')
            plot_orientation(np.array(gmm_mu)[:, 3:], ground_truth[:, 3:], axes)

            plt.legend()
            fig4, axes = plt.subplots(4, 1, figsize=(10, 10))
            fig4.suptitle(f'$Test Demo: {test_demo} --- Model: TP-PMP')
            plot_orientation(np.array(pmp_mu)[:, 3:], ground_truth[:, 3:], axes)
            plt.legend()
        print(f"Mean score GMM: {mean(gmm_score)}, PMP: {mean(pmp_score)}")
        plt.show()


        # for test_demo in test_demos:
        #     ground_truth = gripper_traj_in_ndi[test_demo][d.dims].to_numpy()
        #     t_pmp = np.linspace(0, 1, ground_truth.shape[0])
        #     t_gmm = gripper_traj_in_ndi[test_demo]['Time'].to_numpy()
        #     mu_mean_gmm, sigma_mean_gmm = predict(gmms, t_gmm, test_demo, HT_obj_in_ndi, individuals)
        #     mu_mean_pmp, sigma_mean_pmp = predict(pmps, t_pmp, test_demo, HT_obj_in_ndi, individuals)
        #     gmm_score = gmm_score + [inner_prod_quat(np.array(qua[3:]), ground_truth[j, 3:]) for j, qua in enumerate(mu_mean_gmm)]
        #     pmp_score = pmp_score + [inner_prod_quat(np.array(qua[3:]), ground_truth[j, 3:]) for j, qua in enumerate(mu_mean_pmp)]
        #     fig3, axes = plt.subplots(4, 1, figsize=(10, 10))
        #     fig3.suptitle(f'$Test Demo: {test_demo} --- Model: TP-GMM')
        #     plot_orientation(mu_mean_gmm[:, 3:], ground_truth[:, 3:], axes)
        #
        #     plt.legend()
        #     fig4, axes = plt.subplots(4, 1, figsize=(10, 10))
        #     fig4.suptitle(f'$Test Demo: {test_demo} --- Model: TP-PMP')
        #     plot_orientation(mu_mean_pmp[:, 3:], ground_truth[:, 3:], axes)
        #     plt.legend()
        # plt.show()
