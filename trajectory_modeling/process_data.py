import pickle
from scipy import ndimage
from utils import *

def align_trajectoires(gripper_trajs):
    '''
    This function will align gripper trajectories with Dynamic time warping based on speed.

    Parameters:
    ----------
    gripper_trajs: dict
        A dictionary that contains the gripper trajectories, where the keys are the demo ids and the values are Dataframes.

    Returns:
    --------
    gripper_trajs_aligned: The gripper trajectories that are aligned with median length trajectori.

    '''

    demos = [demo for demo in gripper_trajs.keys()]
    trajs_len = []
    gripper_trajs_aligned = {}
    for demo in demos:
        df = gripper_trajs[demo]
        time_diff = df['Time'].diff(1)
        temp = ((np.sqrt(np.square(df.loc[:, ['x', 'y', 'z']].diff(1)).sum(axis=1)))) / time_diff
        gripper_trajs[demo]['Speed'] = np.array(temp)
        gripper_trajs[demo].dropna(inplace=True)
        trajs_len.append(len(gripper_trajs[demo]))
    # get demos with median duration
    median_len_ind = trajs_len.index(int(np.median(trajs_len)))
    median_len_demo = demos[median_len_ind]

    ref_demo_speed = gripper_trajs[median_len_demo]['Speed'].to_numpy()
    ref_demo_traj = gripper_trajs[median_len_demo].loc[:,
                    gripper_trajs[median_len_demo].columns != 'Speed']

    min_cost_demos = {}
    for demo in demos:
        test_demo_speed = gripper_trajs[demo]['Speed'].to_numpy()
        test_demo_traj = gripper_trajs[demo].loc[:,
                         gripper_trajs[demo].columns != 'Speed'].copy().to_numpy()
        match_indices, min_cost = dynamic_time_warp(ref_demo_speed, test_demo_speed)
        match_indices = np.array(match_indices)
        min_cost_demos[demo] = min_cost
        new_demo = np.zeros(ref_demo_traj.shape)
        for match in match_indices:
            new_demo[match[0]] = test_demo_traj[match[1]]
        new_demo[-1] = test_demo_traj[-1]
        new_demo[0] = test_demo_traj[0]
        demo_aligned = ref_demo_traj.copy()
        demo_aligned.at[:, :] = new_demo
        gripper_trajs_aligned[demo] = demo_aligned
    return gripper_trajs_aligned

def filter_data(gripper_trajs, sigma = 2, dims = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']):
    sigmas = [sigma, 0]
    demos = gripper_trajs.keys()
    gripper_trajs_filtered = {demo: {} for demo in demos}
    for demo in demos:
        q = gripper_trajs[demo].loc[:, dims]
        q_filtered = ndimage.gaussian_filter(q, sigma=sigmas)
        temp = gripper_trajs[demo].copy()
        temp.loc[:, ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']] = q_filtered
        gripper_trajs_filtered[demo] = temp
    return gripper_trajs_filtered

def process_and_save_data(base_dir, template_dir, individuals = ['teabag1', 'teabag2', 'pitcher', 'cup', 'tap']):
    gripper_trajs_full = load_gripper_trajectories(base_dir)
    obj_trajs_full = load_obj_trajectories(base_dir)
    n_actions = get_number_of_actions(base_dir)
    gripper_trajs_truncated = get_gripper_trajectories_for_each_action(base_dir,gripper_trajs_full,  n_actions)
    obj_trajs_truncated = get_obj_trajectories_for_for_each_action(base_dir, obj_trajs_full, n_actions)
    gripper_trajs_in_obj = []
    gripper_trajs_in_obj_aligned = []
    gripper_trajs_in_obj_aligned_filtered = []
    HTs_obj_in_ndi = []
    for i in range(n_actions):
        gripper_action_traj = gripper_trajs_truncated[i]
        obj_action_traj = obj_trajs_truncated[i]
        gripper_action_traj_aligned = align_trajectoires(gripper_action_traj)
        gripper_action_traj_aligned_filtered = filter_data(gripper_action_traj_aligned)

        HT_obj_in_ndi, bad_demos = get_HT_obj_in_ndi(obj_action_traj, individuals, template_dir)
        HTs_obj_in_ndi.append(HT_obj_in_ndi)
        gripper_traj_in_obj = convert_trajectories_to_objects_reference_frame(gripper_action_traj, HT_obj_in_ndi, individuals)
        gripper_traj_in_obj_aligned = convert_trajectories_to_objects_reference_frame(gripper_action_traj_aligned, HT_obj_in_ndi,
                                                                          individuals)
        gripper_traj_in_obj_aligned_filtered = convert_trajectories_to_objects_reference_frame(gripper_action_traj_aligned_filtered, HT_obj_in_ndi,
                                                                              individuals)
        gripper_trajs_in_obj.append(gripper_traj_in_obj)
        gripper_trajs_in_obj_aligned.append(gripper_traj_in_obj_aligned)
        gripper_trajs_in_obj_aligned_filtered.append(gripper_traj_in_obj_aligned_filtered)
    destfolder = os.path.join(base_dir, 'processed')
    if not os.path.isdir(destfolder):
        os.makedirs(destfolder)
    with open(os.path.join(destfolder, 'gripper_trajs_in_obj.pickle'), 'wb') as f1:
        pickle.dump(gripper_trajs_in_obj, f1)
    with open(os.path.join(destfolder, 'gripper_trajs_in_obj_aligned.pickle'), 'wb') as f2:
        pickle.dump(gripper_trajs_in_obj_aligned, f2)
    with open(os.path.join(destfolder, 'gripper_trajs_in_obj_aligned_filtered.pickle'), 'wb') as f3:
        pickle.dump(gripper_trajs_in_obj_aligned_filtered, f3)
    with open(os.path.join(destfolder, 'HTs_obj_in_ndi.pickle'), 'wb') as f4:
        pickle.dump(HTs_obj_in_ndi, f4)

    return gripper_trajs_in_obj, gripper_trajs_in_obj_aligned, gripper_trajs_in_obj_aligned_filtered

if __name__ == '__main__':
    base_dir = '/home/luke/Desktop/project/make_tea/Process_data/postprocessed/2022-05-26'
    template_dir = '/home/luke/Desktop/project/make_tea/Process_data/postprocessed/2022-05-26/transformations/dlc3d'
    individuals = ['teabag1', 'teabag2', 'pitcher', 'cup', 'tap'] # The objects that we will place a reference frame on
    a,b,c = process_and_save_data(base_dir, template_dir, individuals)
    print(c)