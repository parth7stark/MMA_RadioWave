from mma_gcn.config import ServerAgentConfig
from mma_gcn.storage import EventStorage
from mma_gcn.logger import ServerAgentFileLogger


class MergerListenerAgent:
    """
    Contain functions that MergerListenerAgent listener performs like 
    - listen to potential merger events from GW module 
    - storing it in a file
    
    User can overwrite any class method to customize the behavior of the this agent.
    """
    def __init__(
        self,
        merger_agent_config: ServerAgentConfig = ServerAgentConfig()
    ) -> None:

        self.merger_agent_config = merger_agent_config       
        self._create_logger()
        self._load_storage()  # Initialize parameters used by parser


    def _create_logger(self) -> None:
        kwargs = {}
        if hasattr(self.merger_agent_config.gcn_listener_configs, "merger_logging_output_dirname"):
            kwargs["file_dir"] = self.merger_agent_config.gcn_listener_configs.merger_logging_output_dirname
        if hasattr(self.merger_agent_config.gcn_listener_configs, "merger_logging_output_filename"):
            kwargs["file_name"] = self.merger_agent_config.gcn_listener_configs.merger_logging_output_filename
        self.logger = ServerAgentFileLogger(**kwargs)


    def _load_storage(self) -> None:
        """
        Load storage object and initialize parameters
        """

        self.storage: EventStorage = EventStorage(
            self.merger_agent_config,
            self.logger,
        )
        
        

    