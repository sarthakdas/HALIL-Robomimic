"""
Config for KAT algorithm.
"""

from robomimic.config.base_config import BaseConfig


class Multi_KAT_Config(BaseConfig):
    ALGO_NAME = "multi_kat"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
       
        # optimization parameters
        self.algo.sample_frequency = 1
        self.algo.run_length = 20
        self.algo.decimal_places = 3
        self.algo.demonstation_data_path = "tmp/square.hdf5"

        # Make sure demonsation_data_ids len is equal to number_of_demonstrations
        self.algo.garanteed_demonstration_data_ids = [1,2,3,4,5]
        self.algo.requestable_demonstration_data_ids = [6,7,8,9,10]
        self.algo.number_of_demonstrations = 10  # number of demonstrations to use for prompt

        self.algo.ensemble_size = 5

        # optimization parameters REQUIRED FOR ROBOMIMIC
        self.algo.optim_params = {}