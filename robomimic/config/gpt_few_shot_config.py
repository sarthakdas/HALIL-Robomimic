"""
Config for TD3_BC.
"""

from robomimic.config.base_config import BaseConfig


class GPT_Few_ShotConfig(BaseConfig):
    ALGO_NAME = "gpt_few_shot"

    def experiment_config(self):
        """
        Update from subclass to set paper defaults for gym envs.
        """
        super(GPT_Few_ShotConfig, self).experiment_config()

        # 
        

        # no validation and no video rendering
        self.experiment.validate = False
        self.experiment.render_video = False

        # save 10 checkpoints throughout training
        self.experiment.save.every_n_epochs = 20 

        # save models that achieve best rollout return instead of best success rate
        self.experiment.save.on_best_rollout_return = True
        self.experiment.save.on_best_rollout_success_rate = False

        # epoch definition - 5000 gradient steps per epoch, with 200 epochs = 1M gradient steps, and eval every 1 epochs
        self.experiment.epoch_every_n_steps = 5000

        # evaluate with normal environment rollouts
        self.experiment.rollout.enabled = True
        self.experiment.rollout.n = 50              # paper uses 10, but we can afford to do 50
        self.experiment.rollout.horizon = 1000
        self.experiment.rollout.rate = 1            # rollout every epoch to match paper

    def train_config(self):
        """
        Update from subclass to set paper defaults for gym envs.
        """
        super(GPT_Few_ShotConfig, self).train_config()

        # update to normalize observations
        self.train.hdf5_normalize_obs = True 

        # increase batch size to 256
        self.train.batch_size = 256

        # 200 epochs, with each epoch lasting 5000 gradient steps, for 1M total steps
        self.train.num_epochs = 200

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        self.algo.model = "gpt"

        # Prompt configuration
        self.algo.prompt.enviroment = "You are a robot in a room with a red object. do what the prompt says."
        self.algo.prompt.intruction = "Pick up the red object"

        # Examples configuration
        self.algo.optim_params.examples.num_examples = 5
        self.algo.optim_params.examples.sample_freq = 0.25

        # Demonstration configuration
        self.algo.examples.demonstation.EE_pos = True
        self.algo.examples.demonstation.EE_ori = False


    def observation_config(self):
        """
        Update from superclass to use flat observations from gym envs.
        """
        super(GPT_Few_ShotConfig, self).observation_config()
        self.observation.modalities.obs.low_dim = ["flat"]
