import pickle
from scipy import ndimage
from dtw_util import *
import os
from glob import glob
import re
import pandas as pd
from transformations import get_HT_objs_in_ndi, lintrans, inverse_homogenous_transform

class Task_data:
    def __init__(self, base_dir, template_dir, individuals, objs):
        all_files = os.listdir(base_dir)
        r = re.compile("^[0-9]+$")
        self.base_dir = base_dir
        self.template_dir = template_dir
        self.individuals = individuals
        self.objs = objs
        self.demos = list(filter(r.match, all_files))
        self.servos = glob(os.path.join(base_dir, '*', '*Servo*'))
        self.ndis = glob(os.path.join(base_dir, '*', '*NDI*'))
        self.n_actions = self.get_number_of_actions()
        self.start_times = self.get_start_times()
        self.durations = self.get_durations()
        self.dims = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        self.gripper_trajs_full = self.load_gripper_trajectories()
        self.obj_trajs_full = self.load_obj_trajectories()



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

    def load_obj_trajectories(self, triangulation='dlc3d'):
        '''
        This function look for the demos in base_dir, and load marker_3d file.

        Parameters
        ----------
        base_dir: string
            The path to the folder where all the demonstrations are saved.
        triangulation: string
            'leastereo' or 'dlc3d', which corresponds to which triangulation method is used to get the

        Returns
        -------
        markers_trajs: dict
            A dictionary whose keys are the demonstrations' ids. The values are the dataframes of the corresponding
            demonstration's objects' pose trajectories in camera reference frame.
        '''

        markers_trajs = {}
        for demo in self.demos:
            demo_dir = os.path.join(self.base_dir, demo)
            markers_traj_file = os.path.join(demo_dir, triangulation, 'markers_trajectory_3d.h5')
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
        for demo, servo_file in zip(self.demos, self.servos):
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
            for demo in self.demos:
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

    def convert_trajectories_to_objects_reference_frame(self, gripper_trajs_in_ndi, HTs,
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
            for demo in gripper_trajs_in_ndi.keys():
                obj_in_ndi = HTs[demo][individual]
                if ignore_orientation:
                    obj_in_ndi[:3, :3] = np.eye(3)
                ndi_in_obj = inverse_homogenous_transform(obj_in_ndi)
                original_traj = gripper_trajs_in_ndi[demo][self.dims].to_numpy()
                traj_transformed = lintrans(original_traj, ndi_in_obj)
                df_temp = gripper_trajs_in_ndi[demo].copy()
                df_temp[self.dims] = traj_transformed
                gripper_trajs_in_obj[individual][demo] = df_temp
        return gripper_trajs_in_obj

    def align_trajectoires(self, gripper_trajs):
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

    def process_and_save_data(self):
        self.gripper_trajs_truncated = self.get_gripper_trajectories_for_each_action()
        gripper_trajs_in_obj_for_all_actions = []
        gripper_trajs_in_obj_aligned_for_all_actions = []
        gripper_trajs_in_obj_aligned_filtered_for_all_actions = []
        HT_objs_in_ndi_for_all_actions = []
        for i in range(self.n_actions): # Loop through actions
            HT_objs_in_ndi_action = {}
            gripper_action_traj = self.gripper_trajs_truncated[i]
            gripper_action_traj_aligned = self.align_trajectoires(gripper_action_traj)
            gripper_action_traj_aligned_filtered = self.filter_data(gripper_action_traj_aligned)
            for demo in self.demos: # Loop through demos
                start_time = self.start_times[i][demo]
                duration = self.durations[i]
                obj_trajs_demo = self.obj_trajs_full[demo]
                action_ind = int((float(start_time) / float(duration)) * len(obj_trajs_demo))
                obj_trajs_demo_action = obj_trajs_demo.iloc[:action_ind]
                print(i, demo)
                HT_objs_in_ndi_demo, dists = get_HT_objs_in_ndi(obj_trajs_demo_action, self.objs, self.template_dir)
                HT_objs_in_ndi_action[demo] = HT_objs_in_ndi_demo
            HT_objs_in_ndi_for_all_actions.append(HT_objs_in_ndi_action)
            gripper_traj_in_obj = self.convert_trajectories_to_objects_reference_frame(gripper_action_traj, HT_objs_in_ndi_action)
            gripper_traj_in_obj_aligned = self.convert_trajectories_to_objects_reference_frame(gripper_action_traj_aligned, HT_objs_in_ndi_action)
            gripper_traj_in_obj_aligned_filtered = self.convert_trajectories_to_objects_reference_frame(gripper_action_traj_aligned_filtered, HT_objs_in_ndi_action)

            gripper_trajs_in_obj_for_all_actions.append(gripper_traj_in_obj)
            gripper_trajs_in_obj_aligned_for_all_actions.append(gripper_traj_in_obj_aligned)
            gripper_trajs_in_obj_aligned_filtered_for_all_actions.append(gripper_traj_in_obj_aligned_filtered)
        destfolder = os.path.join(base_dir, 'processed')
        if not os.path.isdir(destfolder):
            os.makedirs(destfolder)
        with open(os.path.join(destfolder, 'gripper_trajs_in_obj.pickle'), 'wb') as f1:
            pickle.dump(gripper_trajs_in_obj_for_all_actions, f1)
        with open(os.path.join(destfolder, 'gripper_trajs_in_obj_aligned.pickle'), 'wb') as f2:
            pickle.dump(gripper_trajs_in_obj_aligned_for_all_actions, f2)
        with open(os.path.join(destfolder, 'gripper_trajs_in_obj_aligned_filtered.pickle'), 'wb') as f3:
            pickle.dump(gripper_trajs_in_obj_aligned_filtered_for_all_actions, f3)
        with open(os.path.join(destfolder, 'HTs_obj_in_ndi.pickle'), 'wb') as f4:
            pickle.dump(HT_objs_in_ndi_for_all_actions, f4)

        return gripper_trajs_in_obj_for_all_actions, gripper_trajs_in_obj_aligned_for_all_actions, gripper_trajs_in_obj_aligned_filtered_for_all_actions

if __name__ == '__main__':
    base_dir = '/home/luke/Desktop/project/make_tea/Process_data/postprocessed/2022-08-(17-21)'
    template_dir = '/home/luke/Desktop/project/make_tea/Process_data/postprocessed/2022-08-(17-21)/transformations/dlc3d'
    individuals = ['teabag1', 'teabag2', 'pitcher', 'cup', 'tap'] # The objects that we will place a reference frame on
    objs = ['teabag', 'pitcher', 'cup', 'tap']
    d = Task_data(base_dir, template_dir, individuals, objs)
    d.process_and_save_data()