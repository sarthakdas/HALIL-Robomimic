"""
Implementation of Behavioral Cloning (BC).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.exps.models.base_nets as BaseNets
import robomimic.exps.models.obs_nets as ObsNets
import robomimic.exps.models.policy_nets as PolicyNets
import robomimic.exps.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import llminterface.llm_utils as LLMUtils
import katinterface.kat_utils as KATUtils
import katinterface.pidcontroller as PIDController
from katinterface.kat_func_utils import *

import json
import time
import random
import h5py
import numpy as np
import pickle

@register_algo_factory_func("kat")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    algo_class, algo_kwargs = KAT, {}

    return algo_class, algo_kwargs


class KAT(PolicyAlgo):
    """
    Normal KAT training.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.number_of_demonstations = self.algo_config.number_of_demonstrations
        self.decimal_places = self.algo_config.decimal_places
        self.sample_frequency = self.algo_config.sample_frequency
        self.run_length = self.algo_config.run_length
        self.demonstation_data_path = self.algo_config.demonstation_data_path

        with h5py.File(self.demonstation_data_path, 'r') as hdf5_file:
            # List all groups that start with 'demo_' in the 'data' group
            demos = [key for key in hdf5_file['data'].keys() if key.startswith('demo_')]
            
            # Count the number of demos
            self.total_number_of_demos = len(demos)


        self.demo_data = {}
        
        self.open_ai_client = LLMUtils.OpenAIClient()
        self.llm_queried = False
        self.llm_response = None
        self.demo_ids = []
        self.prompt_path = "_demo_data.txt"
        self.action_iteration = 0
        self.action_to_execute = []

        self.data_capture = {}
        self._setup()

    def _setup(self):
        """
        Setup function that is called in the constructor. Can be overridden by subclasses.
        """
        # create prompt file 
        with open(self.prompt_path, "w") as f:
            f.write("You are a robot that is to generate possible waypoints to control the robot, the input is the object on the table and the ouput is the waypoints")
            f.write("\n")

        self.controller = PIDController.PController6D(5,0.1,0.01)
        pass

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"]
        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def _get_demonstation_data(self, demo_id):
        '''
        Get the demonstration data from the hdf5 file
        '''
        trajectory = {}
        demo_key = f"data/demo_{demo_id}"

        with h5py.File(self.demonstation_data_path, 'r') as file:
            if demo_key not in file:
                raise ValueError(f"Demo {demo_id} not found in the file.")

            demo_group = file[demo_key]

            trajectory['actions'] = demo_group['actions'][:]
            trajectory['dones'] = demo_group['dones'][:]
            trajectory['rewards'] = demo_group['rewards'][:]
            trajectory['states'] = demo_group['states'][:]

            trajectory['obs'] = {
                name: demo_group['obs'][name][:]
                for name in demo_group['obs']
            }
            trajectory['next_obs'] = {
                name: demo_group['next_obs'][name][:]
                for name in demo_group['next_obs']
            }

            trajectory['actions'] = trajectory['actions'].round(self.decimal_places)

        return trajectory

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        # with TorchUtils.maybe_no_grad(no_grad=validate):
        info = super(KAT, self).train_on_batch(batch, epoch, validate=validate)

        for demo_iter in range(self.number_of_demonstations):
            # generate a random number from 0 to X and reroll if number is in the demo_ids
            demo_id = random.randint(0, self.total_number_of_demos - 1)
            if len(self.demo_ids) == self.total_number_of_demos:
                assert False, "All demos have been queried"
            while demo_id in self.demo_ids :
                demo_id = random.randint(0, self.total_number_of_demos - 1)
            self.demo_ids.append(demo_id)

            # get the demo data
            demo_data = self._get_demonstation_data(demo_id)

            # subsample the data to self.sample_frequency
            starting_obs, waypoint_trajectory, action_token_trajectory = demo_data_to_prompt_format(demo_data, self.sample_frequency, self.decimal_places)
            

            if self.data_capture.get("demonstation_data") is None:
                self.data_capture["demonstation_data"] = []

            self.data_capture["demonstation_data"].append({
                "starting_obs": starting_obs,
                "waypoint_trajectory": waypoint_trajectory,
                "action_token_trajectory": action_token_trajectory
            })


            with open(self.prompt_path, "a") as f:
                f.write(f"Example {demo_iter}:\n")
                f.write(f"Input: {starting_obs}\n")
                # f.write(f"Output: {actions.tolist()}\n")
                f.write(f"Output: {action_token_trajectory}\n")

        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(KAT, self).log_info(info)

        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training


        if self.llm_queried == False:
            context_description = "You are a robot and you are to genereate the possible waypoits in triplets formed of (dx, dy, dz), the format is [[front waypoint], [right waypoint], [right waypoint], gripper value] based on the input in json format with the key: 'waypoints' and to 2 decimal places"
            scene_objects = rollout_action_to_input_format(obs_dict["object"], self.decimal_places)
            instruction = "Give the full answer. Generate waypoint paths that you can do based on the objects in the scene in json format as a list of waypoints do not split up into position and orientation: "

            self.kat_action_list = self.open_ai_client.process_test(self.prompt_path, instruction, scene_objects, context_description)
            self.llm_queried = True

            self.waypoint_list_compresssed = action_token_to_waypoint(self.kat_action_list)

            # multiply each action by the run length
            self.waypoint_list = [x for x in self.waypoint_list_compresssed for _ in range(self.run_length)]

        # make sure that enough time is given for the robot to reach the desired state
        if not self.waypoint_list or len(self.waypoint_list) == 0:
            with open("_actions_executed.txt", "a") as f:
                f.write(f"[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n")
            return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.state_to_execute = self.waypoint_list.pop(0)
        
        # calculate action to exectute to get to the desired state 
        current_pos_orn = obs_dict["robot0_eef_pos"][0].tolist() + obs_dict["robot0_eef_quat"][0].tolist() # x,y,z,quaternion
        self.controller.set_target(self.state_to_execute[:6])
        x_action, y_action, z_action, roll_action, pitch_action, yaw_action = self.controller(np.array(current_pos_orn))

        # fix no roll
        # roll_action = 0
        # pitch_action = 0
        # yaw_action = 0

        self.rollout_capture.append({
            "observation": obs_dict,
            "current_eef_pos_orn": current_pos_orn,
            "target_state": self.state_to_execute,
        })
        
        # save the action to execute
        with open("_actions_executed.txt", "a") as f:
            f.write(f"{x_action, y_action, z_action, roll_action, pitch_action, yaw_action}\n")

        self.action_to_execute = [x_action, y_action, z_action, roll_action, pitch_action, yaw_action, self.state_to_execute[6]]
        return torch.tensor([self.action_to_execute])

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self.llm_queried = False
        self.llm_response = None
        self.action_iteration = self.run_length
        with open("_action_sequence.txt", "a") as f:
            f.write("=========================\n")

        if self.data_capture.get("rollout") is None:
            self.data_capture["rollout"] = []
            self.rollout_capture = []
        else:
            self.data_capture["rollout"].append(self.rollout_capture)
            self.rollout_capture = []

        # pick save data_capture to file
        with open("_data_capture_kat.pkl", "wb") as f:
            pickle.dump(self.data_capture, f)

    def on_epoch_end(self, epoch):
        with open(self.prompt_path, "a") as f:
            f.write("\n")
            f.write("Context: {context_description}\n")
            f.write("Your task: {task}\n")
            f.write("Now do it for this input {scene_objects}\n")
        return super().on_epoch_end(epoch)

