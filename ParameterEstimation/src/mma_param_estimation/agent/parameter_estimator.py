from mma_param_estimation.config import ServerAgentConfig
from mma_param_estimation.logger import ServerAgentFileLogger
from mma_param_estimation.downloader import DataDownloader
from mma_param_estimation.results import AnalyzeResults
from omegaconf import OmegaConf, DictConfig
from proxystore.store import Store

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
        self._load_proxystore()
        
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
    
    def clean_up(self) -> None:
        """Clean up the client agent."""
        if hasattr(self, "proxystore") and self.proxystore is not None:
            try:
                self.proxystore.close(clear=True)
            except:
                self.proxystore.close()

    def _load_proxystore(self) -> None:
        """
        Create the proxystore for storing and sending the local mcmc posterior samples from each site to the server.
        """
        if hasattr(self, "proxystore") and self.proxystore is not None:
            return
        self.proxystore = None
        self.use_proxystore = False
        if not hasattr(self.estimator_config, "comm_configs"):
            return
        if not hasattr(self.estimator_config.comm_configs, "proxystore_configs"):
            return
        if getattr(self.estimator_config.comm_configs.proxystore_configs, "enable_proxystore", False):
            self.use_proxystore = True
            self.proxystore = Store(
                name="mma-param-estimation-proxystore",
                connector=self.get_proxystore_connector(
                    self.estimator_config.comm_configs.proxystore_configs.connector_type,
                    self.estimator_config.comm_configs.proxystore_configs.connector_configs,
                ),
            )
            self.logger.info(
                f"Site using proxystore for local MCMC chain transfer with store: {self.estimator_config.comm_configs.proxystore_configs.connector_type}."
            )


    def get_proxystore_connector(self,
        connector_name,
        connector_args,
    ):
        assert connector_name in ["RedisConnector", "FileConnector", "EndpointConnector"], (
            f"Invalid connector name: {connector_name}, only RedisConnector, FileConnector, and EndpointConnector are supported"
        )
        if connector_name == "RedisConnector":
            from proxystore.connectors.redis import RedisConnector

            connector = RedisConnector(**connector_args)
        elif connector_name == "FileConnector":
            from proxystore.connectors.file import FileConnector

            connector = FileConnector(**connector_args)
        elif connector_name == "EndpointConnector":
            from proxystore.connectors.endpoint import EndpointConnector

            connector = EndpointConnector(**connector_args)
        return connector
