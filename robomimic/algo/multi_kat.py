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
from katinterface.kat_func_utils import *

import json
import time
import random
import h5py
import numpy as np
import pickle

@register_algo_factory_func("multi_kat")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    algo_class, algo_kwargs = Multi_KAT, {}

    return algo_class, algo_kwargs


class Multi_KAT(PolicyAlgo):
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
        self.ensemble_size = self.algo_config.ensemble_size
        self.garanteed_demonstation_data_ids = self.algo_config.garanteed_demonstration_data_ids
        self.requestable_demonstation_data_ids = self.algo_config.requestable_demonstration_data_ids

        assert len(self.garanteed_demonstation_data_ids) + len(self.requestable_demonstation_data_ids) == self.number_of_demonstations, "Make sure demonsation_data_ids len is equal to number_of_demonstrations"
        assert len(self.garanteed_demonstation_data_ids) >= 1, "Make sure there is at least one guaranteed demonstation data id"

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

    def _demonstation_data(self, demo_id):
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

    def _obs_to_obsdict(self, obs):
        obs_position = obs[0:3]
        obs_orientation = obs[3:7] # quaterion
        obs_obj_to_ee_position = obs[7:10]
        obs_obj_to_ee_orientation = obs[10:14] # quaternion

        return {
            "object": obs_position.tolist(),
            "object_orientation": obs_orientation.tolist(),
            "object_to_eef_position": obs_obj_to_ee_position.tolist(),
            "object_to_eef_orientation": obs_obj_to_ee_orientation.tolist()
        }
    def _add_garenteed_demonstration_data(self):
         for demo_iter, demo_id in enumerate(self.garanteed_demonstation_data_ids):
            # generate a random number from 0 to X and reroll if number is in the demo_ids
            if len(self.demo_ids) == self.total_number_of_demos:
                assert False, "All demos have been queried"

            while demo_id in self.demo_ids :
                assert False, "Demo has already been queried, you have duplicate IDs"
            self.demo_ids.append(demo_id)

            # get the demo data
            demo_data = self._demonstation_data(demo_id)

            starting_obs, waypoint_trajectory, action_token_trajectory = demo_data_to_prompt_format(demo_data, self.sample_frequency, self.decimal_places)

            if self.data_capture.get("demonsration_data") is None:
                self.data_capture["demonsration_data"] = []
            self.data_capture["demonsration_data"].append({"starting_obs": starting_obs, "waypoint_trajectory": waypoint_trajectory, "action_token_trajectory": action_token_trajectory})


            with open(self.prompt_path, "a") as f:
                f.write(f"Example {demo_iter}:\n")
                f.write(f"Input: {starting_obs}\n")
                # f.write(f"Output: {actions.tolist()}\n")
                f.write(f"Output: {action_token_trajectory}\n")

    def _add_requestable_demonstration_data(self):
        for demo_iter, demo_id in enumerate(self.requestable_demonstation_data_ids):
            if len(self.demo_ids) == self.total_number_of_demos:
                assert False, "All demos have been queried"

            while demo_id in self.demo_ids :
                assert False, "Demo has already been queried, you have duplicate IDs that are"
            self.demo_ids.append(demo_id)

            # get the demo data
            demo_data = self._demonstation_data(demo_id)

            # get the first observation
            starting_obs, waypoint_trajectory, action_token_trajectory = demo_data_to_prompt_format(demo_data, self.sample_frequency, self.decimal_places)
        

            context_description = "You are a robot and you are to genereate the possible waypoits in triplets formed of (dx, dy, dz), the format is [[front waypoint], [right waypoint], [right waypoint], gripper value] based on the input in json format with the key: 'waypoints' and to 2 decimal places"
            scene_objects = str(starting_obs)
            instruction = "Give the full answer. Generate waypoint paths that you can do based on the objects in the scene in json format as a list of waypoints do not split up into position and orientation: "
            
            self.prompt_path = "_demo_data.txt"
            # load the prompt file
            with open(self.prompt_path, "r") as f:
                prompt_data = f.read()
            prompt_data += "\n"
            prompt_data += "Context: {context_description}\n"
            prompt_data += "Your task: {task}\n"
            prompt_data += "Now do it for this input {scene_objects}\n"

            with open("_demo_data_temp.txt", "w") as f:
                f.write(prompt_data)
            help_needed, generated_pathways = self.open_ai_client.process_ensemble_training("_demo_data_temp.txt", instruction, scene_objects, context_description, self.ensemble_size, temperature=0.2)

            self.data_capture["object"] = starting_obs
            # generated pathways kat conver to waypoints and store in data_capture
            # ..... for loop through all the generated path ways convert to waypoints then add to generated_pathways dictionary then save to data capture.
            for i, generated_pathway in enumerate(generated_pathways):
                waypoint_trajectory = action_token_to_waypoint(generated_pathways[i]["action_token_trajectory"])
                generated_pathways[i]["waypoint_trajectory"] = waypoint_trajectory

            self.data_capture["generated_pathways"] = generated_pathways
            self.data_capture["expert_trajectory"] = waypoint_trajectory

            # save data capture to a pickle file
            with open(f"_data_capture_{demo_id}.pkl", "wb") as f:
                pickle.dump(self.data_capture, f)


            if help_needed:
                # subsample the data to self.sample_frequency
                demo_data["actions"] = demo_data["actions"][::self.sample_frequency]

                actions = demo_data["actions"].round(self.decimal_places)
                actions_size = len(actions)
                actions_deomonstation_legnth = len(actions[0])

                kat_actions = []
                for action in actions:
                    left_position, right_position, front_position, gripper = KATUtils.generate_waypoints(action)
                    
                    # mulitply all values by 100 to eliminate decimal place 
                    front_position = [int(coord * 100) for coord in front_position]
                    right_position = [int(coord * 100) for coord in right_position]
                    left_position = [int(coord * 100) for coord in left_position]
                    # remove decial place values 
                    front_position = [round(coord, 0) for coord in front_position]
                    right_position = [round(coord, 0) for coord in right_position]
                    left_position = [round(coord, 0) for coord in left_position]

                    # combine as a list
                    kat_action = [front_position, right_position, left_position, gripper]
                    kat_actions.append(kat_action)

                with open(self.prompt_path, "a") as f:
                    f.write(f"Example {demo_iter}:\n")
                    f.write(f"Input: {starting_obs}\n")
                    # f.write(f"Output: {actions.tolist()}\n")
                    f.write(f"Output: {kat_actions}\n")

        pass
        

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
        info = super(Multi_KAT, self).train_on_batch(batch, epoch, validate=validate)

        self._add_garenteed_demonstration_data()
        self._add_requestable_demonstration_data()

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
        log = super(Multi_KAT, self).log_info(info)
        # log["Loss"] = info["losses"]["action_loss"].item()
        # if "l2_loss" in info["losses"]:
        #     log["L2_Loss"] = info["losses"]["l2_loss"].item()
        # if "l1_loss" in info["losses"]:
        #     log["L1_Loss"] = info["losses"]["l1_loss"].item()
        # if "cos_loss" in info["losses"]:
        #     log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        # if "policy_grad_norms" in info:
        #     log["Policy_Grad_Norms"] = info["policy_grad_norms"]
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

        # with open("_obs_dictionary.txt", "a") as f:
        #     f.write(str(obs_dict))
        #     f.write("\n")
        return self.nets["policy"](obs_dict, goal_dict=goal_dict) #Remove to activate GPT Querying


        # TODO update this to match KAT.py
        if self.llm_queried == False:
            context_description = "You are a robot and you are to genereate the possible waypoits in triplets formed of (dx, dy, dz), the format is [[front waypoint], [right waypoint], [right waypoint], gripper value] based on the input in json format with the key: 'waypoints' and to 2 decimal places"
            obs_dictionary = self._obs_to_obsdict(obs_dict["object"][0])
            scene_objects = str(obs_dictionary)
            instruction = "Give the full answer. Generate waypoint paths that you can do based on the objects in the scene in json format as a list of waypoints do not split up into position and orientation: "

            self.kat_action_list = self.open_ai_client.process_test(self.prompt_path, instruction, scene_objects, context_description)

            self.action_list = []
            for kat_action in self.kat_action_list:
                # with open("_action_sequence_XXX.txt", "a") as f:
                #     f.write(str(kat_action))
                #     f.write("\n")
                # divide all values by 100 to get the decimal value
                kat_action[0] = [coord / 100 for coord in kat_action[0]]
                kat_action[1] = [coord / 100 for coord in kat_action[1]]
                kat_action[2] = [coord / 100 for coord in kat_action[2]]

                # print(kat_action)
                # with open("_action_sequence_XX.txt", "a") as f:
                #     f.write(str(kat_action))
                #     f.write("\n")

                action = KATUtils.inverse_generate_waypoints(np.array(kat_action[0]), np.array(kat_action[1]), np.array(kat_action[2])) # returns the action in the form of [dx, dy, dz, droll, dpitch, dyaw]
                x, y, z, roll, pitch, yaw = action


                self.action_list.append([x, y, z, roll, pitch, yaw, kat_action[3]]) # append the action and the gripper value
            self.llm_queried = True

            with open("_action_sequence_raw.txt", "a") as f:
                f.write(str(self.action_list))
                f.write("\n")

        if self.action_iteration < self.run_length:
            self.action_to_execute = []

            # Loop until an action with the desired length is found
            while len(self.action_to_execute) != 7:
                if not self.action_list or len(self.action_list) == 0:
                    return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
                self.action_to_execute = self.action_list.pop(0)

            # create a file with the action appended on a new line
        with open("_action_sequence.txt", "a") as f:
            f.write(str(self.action_list))
            f.write("\n")  

        return torch.tensor([self.action_to_execute])

        # 
        # action = self.nets["policy"](obs_dict, goal_dict=goal_dict)
        # # save action to a file
        # with open("_action_sequence.txt", "a") as f:
        #     f.write(str(action.tolist()))
        #     f.write("\n")

        # return self.nets["policy"](obs_dict, goal_dict=goal_dict)

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self.llm_queried = False
        self.llm_response = None
        with open("_action_sequence.txt", "a") as f:
            f.write("=========================\n")
        time.sleep(60)

    def on_epoch_end(self, epoch):
        with open(self.prompt_path, "a") as f:
            f.write("\n")
            f.write("Context: {context_description}\n")
            f.write("Your task: {task}\n")
            f.write("Now do it for this input {scene_objects}\n")
        return super().on_epoch_end(epoch)

    def _gripper_0_to_1(self, gripper, mode="encode"):
        if mode == "encode":
            if gripper == -1:
                return 0
            else:
                return 1
        elif mode == "decode":
            if gripper == 0:
                return -1
            else:
                return 1