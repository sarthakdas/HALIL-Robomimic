"""
Config for KAT algorithm.
"""

from robomimic.config.base_config import BaseConfig


class KAT_Config(BaseConfig):
    ALGO_NAME = "kat"

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
        self.algo.number_of_demonstrations = 10  # number of demonstrations to use for prompt

         # optimization parameters REQUIRED FOR ROBOMIMIC
        self.algo.optim_params = {}