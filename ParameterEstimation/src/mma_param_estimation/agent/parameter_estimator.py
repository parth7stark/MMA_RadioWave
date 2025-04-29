from mma_param_estimation.config import ServerAgentConfig
from mma_param_estimation.logger import ServerAgentFileLogger
from mma_param_estimation.downloader import DataDownloader
from mma_param_estimation.results import AnalyzeResults
from omegaconf import OmegaConf, DictConfig

import subprocess
from pathlib import Path

class ParamEstimatorAgent:
    """
    Contain functions that Parameter Estimator performs like
    - Downloading dataset
    - Running dingo pipeline
   
    User can overwrite any class method to customize the behavior of the server agent.
    """
    def __init__(
        self,
        estimator_config: ServerAgentConfig = ServerAgentConfig()
    ) -> None:

        self.estimator_config = estimator_config
        self._create_logger()
        self._load_downloader()
        self._load_result_analyzer()
        
    def run_dingo_pipe(self, ini_file: Path):
        print(f" Running Dingo-BNS inference using config: {ini_file}")
        subprocess.run(["dingo_pipe", str(ini_file)], check=True)

    def _create_logger(self) -> None:
        kwargs = {}
        if hasattr(self.estimator_config.bns_parameter_estimation_configs, "logging_output_dirname"):
            kwargs["file_dir"] = self.estimator_config.bns_parameter_estimation_configs.logging_output_dirname
        if hasattr(self.estimator_config.bns_parameter_estimation_configs, "logging_output_filename"):
            kwargs["file_name"] = self.estimator_config.bns_parameter_estimation_configs.logging_output_filename
        self.logger = ServerAgentFileLogger(**kwargs)


    def _load_downloader(self) -> None:
        """
        Load downloader and initialize parameters
        """

        self.downloader: DataDownloader = DataDownloader(
            self.estimator_config,
            self.logger,
        )
        
    def _load_result_analyzer(self) -> None:
        """
        Load result analyzer object and initialize parameters
        """

        self.analyzer: AnalyzeResults = AnalyzeResults(
            self.estimator_config,
            self.logger,
        )
    