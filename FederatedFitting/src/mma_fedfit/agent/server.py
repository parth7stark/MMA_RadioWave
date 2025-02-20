import threading
from mma_fedfit.aggregator import *
from mma_fedfit.compressor import *
from mma_fedfit.config import ServerAgentConfig
from mma_fedfit.logger import ServerAgentFileLogger
from omegaconf import OmegaConf, DictConfig
from typing import Union, Dict, OrderedDict, Tuple, Optional
from proxystore.store import Store
from proxystore.proxy import Proxy, extract


class ServerAgent:
    """
    Contain functions that Server performs
    `ServerAgent` should act on behalf of the FL server to:
    - provide configurations that are shared among all clients to the clients (e.g. which approach to use? 1 or 2? etc.) `ServerAgent.get_client_configs`
    - send initial parameter guess 
    - take the local posterior samples from a client and aggregate using the server-side model

    User can overwrite any class method to customize the behavior of the server agent.
    """
    def __init__(
        self,
        server_agent_config: ServerAgentConfig = ServerAgentConfig()
    ) -> None:

        self.server_agent_config = server_agent_config

        if hasattr(self.server_agent_config.client_configs, "comm_configs"):
            self.server_agent_config.server_configs.comm_configs = (OmegaConf.merge(
                self.server_agent_config.server_configs.comm_configs,
                self.server_agent_config.client_configs.comm_configs
            ) if hasattr(self.server_agent_config.server_configs, "comm_configs") 
            else self.server_agent_config.client_configs.comm_configs
            )
        
        self._create_logger()
        # self._load_model()   # load server side model with best GNN weights
        self._load_aggregator()  # Initialize parameters used by aggregator
        
        self._load_compressor()
        self._load_proxystore()

    def _create_logger(self) -> None:
        kwargs = {}
        if hasattr(self.server_agent_config.server_configs, "logging_output_dirname"):
            kwargs["file_dir"] = self.server_agent_config.server_configs.logging_output_dirname
        if hasattr(self.server_agent_config.server_configs, "logging_output_filename"):
            kwargs["file_name"] = self.server_agent_config.server_configs.logging_output_filename
        self.logger = ServerAgentFileLogger(**kwargs)


    def _load_aggregator(self) -> None:
        """
        Load aggregator and initialize parameters
        """

        self.aggregator: GlobalAggregator = GlobalAggregator(
            OmegaConf.create(
                self.server_agent_config.server_configs.aggregator_kwargs if
                hasattr(self.server_agent_config.server_configs, "aggregator_kwargs") else {}
            ),
            self.logger,
        )
        

    def _load_compressor(self) -> None:
        """Obtain the compressor."""
        self.compressor = None
        self.enable_compression = False
        if not hasattr(self.server_agent_config.server_configs, "comm_configs"):
            return
        if not hasattr(self.server_agent_config.server_configs.comm_configs, "compressor_configs"):
            return
        if getattr(self.server_agent_config.server_configs.comm_configs.compressor_configs, "enable_compression", False):
            self.enable_compression = True
            self.compressor = eval(self.server_agent_config.server_configs.comm_configs.compressor_configs.lossy_compressor)(
                self.server_agent_config.server_configs.comm_configs.compressor_configs
            )

    def proxy(self, obj) -> Tuple[Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict], Proxy], bool]:
        """
        Create the proxy of the object.
        """
        if self.enable_proxystore:
            return self.proxystore.proxy(obj), True
        else:
            return obj, False
        
    def _load_proxystore(self) -> None:
        """
        Create the proxystore for storing and sending model parameters from the server to the clients.
        """
        self.proxystore = None
        self.enable_proxystore = False
        if not hasattr(self.server_agent_config.server_configs, "comm_configs"):
            return
        if not hasattr(self.server_agent_config.server_configs.comm_configs, "proxystore_configs"):
            return
        if getattr(self.server_agent_config.server_configs.comm_configs.proxystore_configs, "enable_proxystore", False):
            self.enable_proxystore = True
            from proxystore.connectors.redis import RedisConnector
            from proxystore.connectors.file import FileConnector
            from proxystore.connectors.endpoint import EndpointConnector
            # from appfl.communicator.connector import S3Connector
            self.proxystore = Store(
                'server-proxystore',
                eval(self.server_agent_config.server_configs.comm_configs.proxystore_configs.connector_type)(
                    **self.server_agent_config.server_configs.comm_configs.proxystore_configs.connector_configs
                ),
            )
    
    
    def get_client_configs(self, **kwargs) -> DictConfig:
        """Return the FL configurations that are shared among all clients."""
        return self.server_agent_config.client_configs
    

    def close_connection(self, client_id: Union[int, str]) -> None:
        """Record the client that has finished the communication with the server."""
        if not hasattr(self, 'closed_clients'):
            self.closed_clients = set()
            self._close_connection_lock = threading.Lock()
        with self._close_connection_lock:
            self.closed_clients.add(client_id)

    def server_terminated(self):
        """Indicate whether the server can be terminated from listening to the clients."""
        if not hasattr(self, "closed_clients"):
            return False
        num_clients = (
            self.server_agent_config.server_configs.num_clients if 
            hasattr(self.server_agent_config.server_configs, "num_clients") else
            self.server_agent_config.server_configs.scheduler_kwargs.num_clients if
            hasattr(self.server_agent_config.server_configs.scheduler_kwargs, "num_clients") else
            self.server_agent_config.server_configs.aggregator_kwargs.num_clients
        )
        with self._close_connection_lock:
            terminated = len(self.closed_clients) >= num_clients
        if terminated:
            self.clean_up()
        return terminated
    
    def clean_up(self) -> None:
        """
        Nececessary clean-up operations.
        No need to call this method if using `server_terminated` to check the termination status.
        """
        if not hasattr(self, "cleaned"):
            self.cleaned = False
        if not self.cleaned:
            self.cleaned = True
            if hasattr(self, "proxystore") and self.proxystore is not None:
                try:
                    self.proxystore.close(clear=True)
                except:
                    self.proxystore.close()
