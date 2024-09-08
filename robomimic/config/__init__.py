from robomimic.config.config import Config
from robomimic.config.base_config import config_factory, get_all_registered_configs

# note: these imports are needed to register these classes in the global config registry
from robomimic.config.bc_config import BCConfig
from robomimic.config.bcq_config import BCQConfig
from robomimic.config.cql_config import CQLConfig
from robomimic.config.iql_config import IQLConfig
from robomimic.config.gl_config import GLConfig
from robomimic.config.hbc_config import HBCConfig
from robomimic.config.iris_config import IRISConfig
from robomimic.config.td3_bc_config import TD3_BCConfig
from robomimic.config.gpt_few_shot_config import GPT_Few_ShotConfig

from robomimic.config.kat_config import KAT_Config
from robomimic.config.multi_kat_config import Multi_KAT_Config