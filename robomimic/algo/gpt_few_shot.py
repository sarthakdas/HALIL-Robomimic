"""
Implementation of TD3-BC. 
Based on https://github.com/sfujim/TD3_BC
(Paper - https://arxiv.org/abs/1812.02900).

Note that several parts are exactly the same as the BCQ implementation,
such as @_create_critics, @process_batch_for_training, and 
@_train_critic_on_batch. They are replicated here (instead of subclassing 
from the BCQ algo class) to be explicit and have implementation details 
self-contained in this file.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.exps.models.obs_nets as ObsNets
import robomimic.exps.models.policy_nets as PolicyNets
import robomimic.exps.models.value_nets as ValueNets
import robomimic.exps.models.vae_nets as VAENets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.loss_utils as LossUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo, ValueAlgo


@register_algo_factory_func("gpt_few_shot")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the GPT_Few_Shot algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    # only one variant of GPT_Few_Shot for now
    return GPT_Few_Shot, {}


class GPT_Few_Shot(PolicyAlgo, ValueAlgo):
    """
    Default TD3_BC training, based on https://arxiv.org/abs/2106.06860 and
    https://github.com/sfujim/TD3_BC.
    """
    def __init__(self, **kwargs):
        PolicyAlgo.__init__(self, **kwargs)

        # save the discount factor - it may be overriden later
        self.set_discount(self.algo_config.discount)

        # initialize actor update counter. This is used to train the actor at a lower freq than critic
        self.actor_update_counter = 0

        self.demo_data = []

    def _create_networks(self):
        """
        Called on class initialization - should construct networks and place them into the self.nets ModuleDict.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    # =============== #
    # TRAIN
    # =============== #
    def set_train(self):
        '''
        Prepares network modules for training. By default, just calls self.nets.train(), 
        but certain algorithms may always want a subset of the networks in evaluation mode (such as target networks for BCQ).
        In this case they should override this method.
        '''
        return super().set_train()
    
    def process_batch_for_training(self, batch):
        '''
        Takes a batch sampled from the data loader, and filters out the relevant portions needed for the algorithm.
        It should also send the batch to the correct device (cpu or gpu).
        '''
        return super().process_batch_for_training(batch)
    
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
        # Assert that epoch is not greater then 1
        assert epoch == 1

        # save the demo data to self.demo_data
        self.demo_data.append(batch)

        return super().train_on_batch(batch, epoch, validate)
    
    def log_info(self, info):
        '''
        Takes the output of train_on_batch and returns a new processed dictionary for tensorboard logging.
        '''
        return super().log_info(info)
    
    def serialize(self):
        '''
        Returns the state dictionary that contains the current model parameters. This is used to produce agent checkpoints. By default, 
        returns self.nets.state_dict() - usually only needs to be overriden by hierarchical algorithms like HBC and IRIS to
        collect state dictionaries from sub-algorithms.
        '''
        return super().serialize()
    
    def on_epoch_end(self, epoch):
        '''Called at the end of each training epoch. Usually consists of stepping learning rate schedulers (if they are being used).'''

        # Assert that epoch is not greater then 1
        assert epoch == 1

        # save the self.demo_data to as a txt file
        with open("demo_data.txt", "w") as f:
            for data in self.demo_data:
                f.write(str(data))
                f.write("\n")

        return super().on_epoch_end(epoch)
    

    # =============== #
    # TEST
    # =============== #
    def deserialize(self, model_dict):
        '''
        Inverse operation of serialize - load model weights. Used at test-time to restore model weights.
        '''
        return super().deserialize(model_dict)
    
    def set_eval(self):
        '''
        Prepares network modules for evaluation. By default, just calls self.nets.eval(), 
        but certain hierarchical algorithms like HBC and IRIS override this to call set_eval on their sub-algorithms.
        '''
        return super().set_eval()
    
    def get_action(self, obs_dict, goal_dict=None):
        '''
        The primary method that is called at test-time to return one or more actions, given observations.
        '''
        return super().get_action(obs_dict, goal_dict)
    
    def reset(self):
        '''
        Called at the beginning of each rollout episode to clear internal agent state before starting a rollout.
        As an example, BC_RNN resets the step counter and hidden state.
        '''
        return super().reset()

    

    
