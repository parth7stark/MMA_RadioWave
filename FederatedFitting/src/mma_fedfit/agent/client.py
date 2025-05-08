import uuid
from proxystore.store import Store
from mma_fedfit.compressor import *
from mma_fedfit.generator import LocalGenerator
from mma_fedfit.config import ClientAgentConfig
from omegaconf import DictConfig, OmegaConf
from typing import Union, Dict, OrderedDict, Tuple, Optional
from mma_fedfit.logger import ClientAgentFileLogger


class ClientAgent:
    """
    Contain functions that client performs
    The `ClientAgent` should act on behalf of the FL client to:
    - load configurations received from the server `ClientAgent.load_config`
    - compute local posterior samples  `ClientAgent.run_local_mcmc`
    - prepare data for communication `ClientAgent.get_parameters`
    - get a unique client id for server to distinguish clients `ClientAgent.get_id`


    Users can overwrite any class method to add custom functionalities of the client agent.
    """
    def __init__(
        self, 
        client_agent_config: ClientAgentConfig = ClientAgentConfig()
    ) -> None:
        self.client_agent_config = client_agent_config
        self._create_logger()
        self._load_generator()
        self._load_compressor()
        self._load_proxystore()

    def load_config(self, config: DictConfig) -> None:
        """Load additional configurations provided by the server."""
        self.client_agent_config = OmegaConf.merge(self.client_agent_config, config)
        # Initialize again so that they get updated configs
        self._load_generator()
        self._load_compressor()

    def get_id(self) -> str:
        """Return a unique client id for server to distinguish clients."""
        if not hasattr(self, 'client_id'):
            if hasattr(self.client_agent_config, "client_id"):
                self.client_id = self.client_agent_config.client_id
            else:
                self.client_id = str(uuid.uuid4())
        return self.client_id
    

    def run_local_mcmc(self, preprocessed_local_data, pos, niter, nwalkers, ndim) -> None:
        """Compute local embedding using the local data."""
        self.generator.run_local_mcmc(preprocessed_local_data, pos, niter, nwalkers, ndim)

    
    def compute_log_likelihood(self, theta, fitting_data) -> None:
        """Compute local embedding using the local data."""
        return self.generator.compute_log_likelihood(theta, fitting_data)


    def get_parameters(self) -> Union[Dict, OrderedDict, bytes, Tuple[Union[Dict, OrderedDict, bytes], Dict]]:
        """
        Get local embeddings for sending to server
        Return parameters for communication
        """
        params = self.generator.get_parameters()
        # params = {k: v.cpu() for k, v in params.items()}
        if isinstance(params, tuple):
            params, metadata = params
        else:
            metadata = None
        if self.enable_compression:
            params = self.compressor.compress_model(params)

        return params if metadata is None else (params, metadata)
        # return self.proxy(params)[0] if metadata is None else (self.proxy(params)[0], metadata)
    

    # def proxy(self, obj):
    #     """
    #     Create the proxy of the object.
    #     :param obj: the object to be proxied.
    #     :return: the proxied object and a boolean value indicating whether the object is proxied.
    #     """
    #     if self.enable_proxystore:
    #         return self.proxystore.proxy(obj), True
    #     else:
    #         return obj, False
        
    def clean_up(self) -> None:
        """Clean up the client agent."""
        if hasattr(self, "proxystore") and self.proxystore is not None:
            try:
                self.proxystore.close(clear=True)
            except:
                self.proxystore.close()

    def _create_logger(self):
        """
        Create logger for the client agent to log local training process.
        You can modify or overwrite this method to create your own logger.
        """
        if hasattr(self, "logger"):
            return
        kwargs = {}
        if not hasattr(self.client_agent_config, "fitting_configs"):
            kwargs["logging_id"] = self.get_id()
            kwargs["file_dir"] = "./output"
            kwargs["file_name"] = "result"
        else:
            kwargs["logging_id"] = self.client_agent_config.fitting_configs.get("logging_id", self.get_id())
            kwargs["file_dir"] = self.client_agent_config.fitting_configs.get("logging_output_dirname", "./output")
            kwargs["file_name"] = self.client_agent_config.fitting_configs.get("logging_output_filename", "result")
        if hasattr(self.client_agent_config, "experiment_id"):
            kwargs["experiment_id"] = self.client_agent_config.experiment_id
        self.logger = ClientAgentFileLogger(**kwargs)


    def _load_generator(self) -> None:
        """
        LocalGenerator for FL clients, which computes/generates the local posterior samples using the given data
        """

        self.generator: LocalGenerator = LocalGenerator(
            client_agent_config=self.client_agent_config,
            logger=self.logger,
        )

        
    def _load_compressor(self) -> None:
        """
        Create a compressor for compressing the model parameters.
        """
        if hasattr(self, "compressor") and self.compressor is not None:
            return
        self.compressor = None
        self.enable_compression = False
        if not hasattr(self.client_agent_config, "comm_configs"):
            return
        if not hasattr(self.client_agent_config.comm_configs, "compressor_configs"):
            return
        if getattr(self.client_agent_config.comm_configs.compressor_configs, "enable_compression", False):
            self.enable_compression = True
            self.compressor = eval(self.client_agent_config.comm_configs.compressor_configs.lossy_compressor)(
               self.client_agent_config.comm_configs.compressor_configs
            )

    def _load_proxystore(self) -> None:
        """
        Create the proxystore for storing and sending the local mcmc posterior samples from each site to the server.
        """
        if hasattr(self, "proxystore") and self.proxystore is not None:
            return
        self.proxystore = None
        self.use_proxystore = False
        if not hasattr(self.client_agent_config, "comm_configs"):
            return
        if not hasattr(self.client_agent_config.comm_configs, "proxystore_configs"):
            return
        if getattr(self.client_agent_config.comm_configs.proxystore_configs, "enable_proxystore", False):
            self.use_proxystore = True
            self.proxystore = Store(
                name="mma-rw-proxystore",
                connector=self.get_proxystore_connector(
                    self.client_agent_config.comm_configs.proxystore_configs.connector_type,
                    self.client_agent_config.comm_configs.proxystore_configs.connector_configs,
                ),
            )
            self.logger.info(
                f"Site using proxystore for local MCMC chain transfer with store: {self.client_agent_config.comm_configs.proxystore_configs.connector_type}."
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


            


