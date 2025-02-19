import uuid
from proxystore.store import Store
from mma_fedfit.compressor import *
from mma_fedfit.generator import GWGenerator
from mma_fedfit.config import ClientAgentConfig
from omegaconf import DictConfig, OmegaConf
from typing import Union, Dict, OrderedDict, Tuple, Optional
from mma_fedfit.logger import ClientAgentFileLogger
from mma_fedfit.model.GW_client_model import ClientModel
import h5py


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
        return self.proxy(params)[0] if metadata is None else (self.proxy(params)[0], metadata)
    

    def proxy(self, obj):
        """
        Create the proxy of the object.
        :param obj: the object to be proxied.
        :return: the proxied object and a boolean value indicating whether the object is proxied.
        """
        if self.enable_proxystore:
            return self.proxystore.proxy(obj), True
        else:
            return obj, False
        
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
        if not hasattr(self.client_agent_config, "generator_configs"):
            kwargs["logging_id"] = self.get_id()
            kwargs["file_dir"] = "./output"
            kwargs["file_name"] = "result"
        else:
            kwargs["logging_id"] = self.client_agent_config.generator_configs.get("logging_id", self.get_id())
            kwargs["file_dir"] = self.client_agent_config.generator_configs.get("logging_output_dirname", "./output")
            kwargs["file_name"] = self.client_agent_config.generator_configs.get("logging_output_filename", "result")
        if hasattr(self.client_agent_config, "experiment_id"):
            kwargs["experiment_id"] = self.client_agent_config.experiment_id
        self.logger = ClientAgentFileLogger(**kwargs)


    def _load_generator(self) -> None:
        """
        do what load_trainer is doing
        Load embeddings generator and initialize parameters
        """

        self.generator: GWGenerator = GWGenerator(
            model=self.model, 
            generator_configs=self.client_agent_config.generator_configs,
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
        Create the proxystore for storing and sending the model parameters from the client to the server.
        """
        if hasattr(self, "proxystore") and self.proxystore is not None:
            return
        self.proxystore = None
        self.enable_proxystore = False
        if not hasattr(self.client_agent_config, "comm_configs"):
            return
        if not hasattr(self.client_agent_config.comm_configs, "proxystore_configs"):
            return
        if getattr(self.client_agent_config.comm_configs.proxystore_configs, "enable_proxystore", False):
            self.enable_proxystore = True
            from proxystore.connectors.redis import RedisConnector
            from proxystore.connectors.file import FileConnector
            from proxystore.connectors.endpoint import EndpointConnector
            # from appfl.communicator.connector.s3 import S3Connector
            self.proxystore = Store(
                self.get_id(),
                eval(self.client_agent_config.comm_configs.proxystore_configs.connector_type)(
                    **self.client_agent_config.comm_configs.proxystore_configs.connector_configs
                ),
            )

            


