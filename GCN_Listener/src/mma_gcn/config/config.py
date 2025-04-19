from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf

@dataclass
class ServerAgentConfig:
    gcn_listener_configs: DictConfig = OmegaConf.create({})