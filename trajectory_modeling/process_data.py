import pickle
from scipy import ndimage
from dtw_util import *
import os
from glob import glob
import re
import pandas as pd
from transformations import get_HT_objs_in_ndi, lintrans, inverse_homogenous_transform, pairwise_constrained_axis3d, axis3d_to_quat, axis3d_to_rotmatrix, homogenous_transform,get_HT_for_grouped_object
import yaml
import numpy as np
from itertools import permutations
from quaternion_metric import process_quaternions

class Task_data:
    def __init__(self, base_dir, triangulation, individuals):
        all_files = os.listdir(base_dir)
        r = re.compile("^[0-9]+$")
        self.base_dir = base_dir
        self.triangulation = triangulation
        self.template_dir = os.path.join(base_dir, 'transformations', triangulation)
        self.obj_templates = self.get_obj_templates()
        self.obj_templates_valid = self.remove_invisible_markers_in_template()
        self.individuals = individuals + ['global']
        self.objs = self.obj_templates.keys()
        self.demos = list(filter(r.match, all_files))
        self.bad_demos = []
        self.servos = glob(os.path.join(base_dir, '*', '*Servo*'))
        self.ndis = glob(os.path.join(base_dir, '*', '*NDI*'))
        self.n_actions = self.get_number_of_actions()
        self.start_times = self.get_start_times()
        self.end_times = self.get_end_times()
        self.durations = self.get_durations()
        self.dims = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        self.gripper_trajs_full = self.load_gripper_trajectories()
        self.obj_trajs_full = self.load_obj_trajectories()



    def get_obj_templates(self,):
        with open(os.path.join(self.template_dir, 'obj_templates.pickle'), 'rb') as f:
            obj_templates = pickle.load(f)
        return obj_templates

    def get_number_of_actions(self):
        n_actions = []
        for servo_file in self.servos:
            n_action = 0
            with open(servo_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if 'closed' in line:
                        n_action += 1
            n_actions.append(n_action)
        return int(np.median(n_actions))

    def get_start_times(self):
        start_times = [{} for i in range(self.n_actions)]
        for demo, servo_file in zip(self.demos, self.servos):
            j = 0
            with open(servo_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if 'closed' in line:
                    start_times[j][demo] = line.split(',')[0]
                    j += 1
        return start_times

    def get_end_times(self):
        end_times = [{} for i in range(self.n_actions)]
        for demo, servo_file in zip(self.demos, self.servos):
            j = 0
            with open(servo_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if 'open' in line:
                    end_times[j][demo] = line.split(',')[0]
                    j += 1
        return end_times

    def get_durations(self):
        durations = []
        for demo, servo_file in zip(self.demos, self.servos):
            with open(servo_file, 'r') as f:
                lines = f.readlines()
                end_time = lines[-1].split(',')[0]
                durations.append(end_time)
        return durations

    def load_gripper_trajectories(self):
        '''
        This function look for the demos in base_dir, and load the ndi file.

        Parameters
        ----------
        base_dir: string
            The path to the folder where all the demonstrations are saved.

        Returns
        -------
        gripper_trajs: dict
            A dictionary whose keys are the demonstrations' ids.
            The values are the dataframes of the corresponding
            demonstration's gripper trajectory data in ndi reference frame.
        '''

        gripper_trajs = {}
        for demo, ndi_file in zip(self.demos, self.ndis):
            df_temp = pd.read_csv(ndi_file)
            gripper_trajs[demo] = df_temp
        return gripper_trajs

    def load_obj_trajectories(self):
        '''
        This function look for the demos in base_dir, and load marker_3d file.

        Parameters
        ----------
        base_dir: string
            The path to the folder where all the demonstrations are saved.

        Returns
        -------
        markers_trajs: dict
            A dictionary whose keys are the demonstrations' ids. The values are the dataframes of the corresponding
            demonstration's objects' pose trajectories in camera reference frame.
        '''

        markers_trajs = {}
        for demo in self.demos:
            demo_dir = os.path.join(self.base_dir, demo)
            markers_traj_file = os.path.join(demo_dir, self.triangulation, 'markers_trajectory_3d.h5')
            df_temp = pd.read_hdf(markers_traj_file)
            markers_trajs[demo] = df_temp.droplevel('scorer', axis=1)
        return markers_trajs


    def get_gripper_trajectories_for_each_action(self):
        '''
            Given the gripper trajs this function will go in basedir and look for the Servo file
            to chunk the trajs and output each action's traj.

            Parameters
            ---------
            dfs_ndi: dict
                A dictionary that contains the full gripper trajectories for different demos.
            base_dir: string
                The path to the directory that contains the demos
            i: int
                The index of the action

            Returns
            gripper_trajs: The gripper trajectories for each action from different demonstrations.
            -------
            '''
        gripper_trajs = {}
        demos = [demo for demo in self.demos if demo not in self.bad_demos]
        for demo, servo_file in zip(demos, self.servos):
            gripper_trajs[demo] = []
            df_ndi = self.gripper_trajs_full[demo]
            with open(servo_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if 'closed' in line:
                        t_start = float(lines[i].split(',')[0])
                        t_end = float(lines[1 + i].split(',')[0])
                        idx_start = df_ndi['Time'].sub(t_start).abs().idxmin()
                        idx_end = df_ndi['Time'].sub(t_end).abs().idxmin()
                        df_gripper = df_ndi.copy()[idx_start: idx_end]
                        df_gripper = df_gripper.drop(columns=df_gripper.columns[0])
                        # Set the time to start from 0
                        df_gripper.loc[:, 'Time'] = df_gripper.loc[:, 'Time'] - df_gripper.loc[:, 'Time'].iloc[0]
                        gripper_trajs[demo].append(df_gripper)
        action_trajs = []
        for i in range(int(self.n_actions)):
            temp = {}
            for demo in demos:
                temp[demo] = gripper_trajs[demo][i]
            action_trajs.append(temp)
        return action_trajs

    def get_obj_trajectories_for_each_action(base_dir, dfs_camera, n_actions, slack=3):
        '''
        Given the markers_trajs loaded, this function will go in basedir and look for the Servo file
        to chunk the trajs and output the each action's traj.

        Parameters
        ---------
        dfs_camera: dict
            A dictionary that contains the full object markers trajectories for different demos.
        basedir: string
            The path to the directory that contains the demos
        slack: The amount of time to look back at the beginning of an action and to look forward at the end.
        '''
        markers_trajs = {}
        demos = dfs_camera.keys()
        for demo in demos:
            markers_trajs[demo] = []
            df_camera = dfs_camera[demo]
            demo_dir = os.path.join(base_dir, demo)
            servo_file = glob(os.path.join(demo_dir, '*Servo*'))[0]
            with open(servo_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if 'closed' in line:
                        t_start = float(lines[i].split(',')[0]) - slack
                        t_end = float(lines[1 + i].split(',')[0]) + slack
                        task_duration = float(lines[-1].split(',')[0])
                        if t_start < 0:
                            t_start = 0
                        if t_end > task_duration:
                            t_end = task_duration
                        action_duration = t_end - t_start
                        idx_start = int(len(df_camera) * t_start / task_duration)
                        idx_end = int(len(df_camera) * t_end / task_duration)
                        df_markers = df_camera[idx_start: idx_end].reset_index(drop=True)
                        df_markers.loc[:, 'Time'] = np.arange(len(df_markers)) / len(df_markers) * action_duration
                        markers_trajs[demo].append(df_markers)
        action_trajs = []
        for i in range(int(n_actions)):
            temp = {}
            for demo in demos:
                temp[demo] = markers_trajs[demo][i]
            action_trajs.append(temp)
        return action_trajs

    def convert_trajectories_to_objects_reference_frame(self, gripper_trajs_in_ndi, HTs, individuals,
                                                        ignore_orientation=True):
        '''
        This function will convert the gripper trajectories from NDI reference frame to objects' frames.

        Parameters:
        ----------
        gripper_trajs_in_ndi: dict
            A dictionary that contains the gripper trajectory in each demonstration.
        HTs: dict
            A dictionary that contains the homogeneous transformation matrix that will convert the trajectories from NDI to object's reference frames.
        individuals: list
            objects that are relative to the task.
        dims: list
            The dimensions that will be converted.
        ignore_orientation: bool
            Whether or not take the orientation of the objects into consideration when covert trajectories to objects reference frames.
        '''
        gripper_trajs_in_obj = {}
        for individual in individuals:
            gripper_trajs_in_obj[individual] = {}
            for demo in sorted(gripper_trajs_in_ndi.keys()):
                obj_in_ndi = HTs[individual][demo]
                if ignore_orientation:
                    obj_in_ndi[:3, :3] = np.eye(3)
                ndi_in_obj = inverse_homogenous_transform(obj_in_ndi)
                # print(ignore_orientation, obj_in_ndi, ndi_in_obj)
                original_traj = gripper_trajs_in_ndi[demo][self.dims].to_numpy()
                traj_transformed = lintrans(original_traj, ndi_in_obj)
                quats = traj_transformed[:, 3:]
                quats_new = process_quaternions(quats, sigma = 0)
                traj_transformed[:, 3:] = quats_new
                df_temp = gripper_trajs_in_ndi[demo].copy()
                df_temp[self.dims] = traj_transformed
                gripper_trajs_in_obj[individual][demo] = df_temp
        return gripper_trajs_in_obj

    def align_trajectoires(self, gripper_trajs, alignment_func):
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

        trajs_len = []
        gripper_trajs_aligned = {}
        for demo in self.demos:
            df = gripper_trajs[demo]
            time_diff = df['Time'].diff(1)
            temp = ((np.sqrt(np.square(df.loc[:, ['x', 'y', 'z']].diff(1)).sum(axis=1)))) / time_diff
            gripper_trajs[demo]['Speed'] = np.array(temp)
            gripper_trajs[demo].dropna(inplace=True)
            trajs_len.append(len(gripper_trajs[demo]))
        # get demos with median duration
        median = int(np.median(trajs_len))
        if median not in trajs_len:
            # Deal with the case that there are even number of trajs
            median = min(trajs_len, key=lambda x:abs(x-median))
        median_len_ind = trajs_len.index(median)
        median_len_demo = self.demos[median_len_ind]

        ref_demo_speed = gripper_trajs[median_len_demo]['Speed'].to_numpy()
        ref_demo_traj = gripper_trajs[median_len_demo].loc[:,
                        gripper_trajs[median_len_demo].columns != 'Speed']

        min_cost_demos = {}
        for demo in self.demos:
            test_demo_speed = gripper_trajs[demo]['Speed'].to_numpy()
            test_demo_traj = gripper_trajs[demo].loc[:,
                             gripper_trajs[demo].columns != 'Speed'].copy().to_numpy()
            match_indices, min_cost = alignment_func(ref_demo_speed, test_demo_speed)
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

    def filter_data(self, gripper_trajs, sigma = 2):
        sigmas = [sigma, 0]
        gripper_trajs_filtered = {demo: {} for demo in self.demos}
        for demo in self.demos:
            q = gripper_trajs[demo].loc[:, self.dims]
            q_filtered = ndimage.gaussian_filter(q, sigma=sigmas)
            temp = gripper_trajs[demo].copy()
            temp.loc[:, self.dims] = q_filtered
            gripper_trajs_filtered[demo] = temp
        return gripper_trajs_filtered

    def remove_invisible_markers_in_template(self):
        objs = self.obj_templates.keys()
        obj_templates_valid = {}
        for obj in objs:
            obj_templates_valid[obj] = {}
            bps = self.obj_templates[obj].keys()
            for bp in bps:
                if not np.isnan(self.obj_templates[obj][bp]).any():
                    obj_templates_valid[obj][bp] = self.obj_templates[obj][bp]
        return obj_templates_valid


    def process_data(self, alignment_func, thres = 80,markers_average = False):
        self.grouped_objects = self.get_grouped_objects()
        self.gripper_trajs_truncated = self.get_gripper_trajectories_for_each_action()
        self.gripper_trajs_in_obj_for_all_actions = []
        self.gripper_trajs_in_obj_aligned_for_all_actions = []
        self.gripper_trajs_in_obj_aligned_filtered_for_all_actions = []
        self.HT_objs_in_ndi_for_all_actions = []
        self.HT_grouped_objs_in_ndi_for_all_actions = []
        self.gripper_traj_in_grouped_objs_aligned_filtered_for_all_actions = []
        for i in range(self.n_actions): # Loop through actions
            self.gripper_trajs_truncated = self.get_gripper_trajectories_for_each_action()
            gripper_action_traj = self.gripper_trajs_truncated[i]
            gripper_action_traj_aligned = self.align_trajectoires(gripper_action_traj, alignment_func=alignment_func)
            gripper_action_traj_aligned_filtered = self.filter_data(gripper_action_traj_aligned)
            bad_demos_action = []
            HT_objs_in_ndi_action = {}
            HT_grouped_objs_in_ndi_action = {}
            for ind in self.individuals:
                HT_objs_in_ndi_action[ind] = {}
            for ind in self.grouped_objects:
                HT_grouped_objs_in_ndi_action[ind] = {}
            for demo in self.demos: # Loop through demos
                start_time = self.start_times[i][demo]
                duration = self.durations[i]
                obj_trajs_demo = self.obj_trajs_full[demo]
                start_ind = int((float(start_time) / float(duration)) * len(obj_trajs_demo))
                if i == 0: # The first action
                    obj_trajs_demo_action = obj_trajs_demo.iloc[:start_ind]
                else:
                    previous_end_time = self.end_times[i -1][demo]
                    previous_end_ind = int((float(previous_end_time) / float(duration)) * len(obj_trajs_demo))
                    obj_trajs_demo_action = obj_trajs_demo.iloc[previous_end_ind: start_ind]
                print(f'Action ind: {i}, Demo #: {demo}')
                HT_objs_in_ndi_demo, dists = get_HT_objs_in_ndi(obj_trajs_demo_action, self.obj_templates_valid, CAMERA_IN_NDI , self.individuals, markers_average = markers_average)
                if np.max(dists) > thres and demo not in bad_demos_action:#Unmatched demo if the dists higher than the threshold
                    bad_demos_action.append(demo)
                else: ## Found a match and got the homogeneous transformation for each object
                    for ind in self.individuals:
                        HT_objs_in_ndi_action[ind][demo] = HT_objs_in_ndi_demo[ind]
                    for ind in self.grouped_objects:
                        HT_grouped_objs_in_ndi_demo = get_HT_for_grouped_object(ind, HT_objs_in_ndi_demo)
                        HT_grouped_objs_in_ndi_action[ind][demo] = HT_grouped_objs_in_ndi_demo
            print(f'The bad demos for action {i} are: {bad_demos_action}')
            self.HT_objs_in_ndi_for_all_actions.append(HT_objs_in_ndi_action)
            self.HT_grouped_objs_in_ndi_for_all_actions.append(HT_grouped_objs_in_ndi_action)
            self.bad_demos.append(bad_demos_action)
            for demo in bad_demos_action: # Remove the unmatched demo
                del gripper_action_traj[demo]
                del gripper_action_traj_aligned[demo]
                del gripper_action_traj_aligned_filtered[demo]
            gripper_traj_in_obj = self.convert_trajectories_to_objects_reference_frame(gripper_action_traj, HT_objs_in_ndi_action, self.individuals)
            gripper_traj_in_obj_aligned = self.convert_trajectories_to_objects_reference_frame(gripper_action_traj_aligned, HT_objs_in_ndi_action, self. individuals)
            gripper_traj_in_obj_aligned_filtered = self.convert_trajectories_to_objects_reference_frame(gripper_action_traj_aligned_filtered, HT_objs_in_ndi_action, self.individuals)
            gripper_traj_in_grouped_obj_aligned_filtered = self.convert_trajectories_to_objects_reference_frame(gripper_action_traj_aligned_filtered, HT_grouped_objs_in_ndi_action, self.grouped_objects, ignore_orientation=False)
            self.gripper_trajs_in_obj_for_all_actions.append(gripper_traj_in_obj)
            self.gripper_trajs_in_obj_aligned_for_all_actions.append(gripper_traj_in_obj_aligned)
            self.gripper_trajs_in_obj_aligned_filtered_for_all_actions.append(gripper_traj_in_obj_aligned_filtered)
            self.gripper_traj_in_grouped_objs_aligned_filtered_for_all_actions.append(gripper_traj_in_grouped_obj_aligned_filtered)
        return

    def save_data(self, destfolder = None):
        if destfolder == None:
            destfolder = os.path.join(self.base_dir, 'processed', self.triangulation)
        if not os.path.isdir(destfolder):
            os.makedirs(destfolder)
        with open(os.path.join(destfolder, 'gripper_trajs_in_obj.pickle'), 'wb') as f1:
            pickle.dump(self.gripper_trajs_in_obj_for_all_actions, f1)
        with open(os.path.join(destfolder, 'gripper_trajs_in_obj_aligned.pickle'), 'wb') as f2:
            pickle.dump(self.gripper_trajs_in_obj_aligned_for_all_actions, f2)
        with open(os.path.join(destfolder, 'gripper_trajs_in_obj_aligned_filtered.pickle'), 'wb') as f3:
            pickle.dump(self.gripper_trajs_in_obj_aligned_filtered_for_all_actions, f3)
        with open(os.path.join(destfolder, 'HTs_obj_in_ndi.pickle'), 'wb') as f4:
            pickle.dump(self.HT_objs_in_ndi_for_all_actions, f4)
        with open(os.path.join(destfolder, 'HTs_grouped_objs_in_ndi.pickle'), 'wb') as f5:
            pickle.dump(self.HT_grouped_objs_in_ndi_for_all_actions, f5)
        with open(os.path.join(destfolder, 'gripper_traj_in_grouped_objs_aligned_filtered.pickle'), 'wb') as f6:
            pickle.dump(self.gripper_traj_in_grouped_objs_aligned_filtered_for_all_actions, f6)
        return

    def get_grouped_objects(self):
        grouped_objects = []
        individuals = [ind for ind in self.individuals if ind != 'global']
        perms = permutations(individuals, 2)
        for perm in perms:
            obj0 = list(perm)[0]
            obj1 = list(perm)[1]
            grouped_obj = obj0 + '-' + obj1
            if grouped_obj not in grouped_objects:
                grouped_objects.append(grouped_obj)
        return grouped_objects

if __name__ == '__main__':
    task_config_dir = '../Process_data/postprocessed/2022-10-27'
    with open(os.path.join(task_config_dir, 'task_config.yaml')) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    camera_in_ndi_path = os.path.join(config["project_path"], config["camera_in_ndi"])
    with open(os.path.join(camera_in_ndi_path), 'rb') as f:
        CAMERA_IN_NDI = pickle.load(f)
    base_dir = os.path.join(config["project_path"], config["postprocessed_dir"])
    triangulation = 'dlc3d'
    individuals = config["individuals"]# The objects that we will place a reference frame on
    d = Task_data(base_dir, triangulation, individuals)
    d.process_data(alignment_func=dtw_funcs[config["alignment_method"]])
    d.save_data()


