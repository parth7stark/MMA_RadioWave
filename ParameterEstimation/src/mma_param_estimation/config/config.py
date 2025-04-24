from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf

@dataclass
class ServerAgentConfig:
    param_estimator_configs: DictConfig = OmegaConf.create({})