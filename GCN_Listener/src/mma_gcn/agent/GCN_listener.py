from mma_gcn.config import ServerAgentConfig
from mma_gcn.logger import ServerAgentFileLogger
from mma_gcn.parser import GCNParser
from mma_gcn.storage import EventStorage
from omegaconf import OmegaConf, DictConfig


class GCNAgent:
    """
    Contain functions that GCN listener performs like
    - listening to incoming gcn alerts like notices and circulars
    - parsing and storing it in file
   
    User can overwrite any class method to customize the behavior of the server agent.
    """
    def __init__(
        self,
        gcn_agent_config: ServerAgentConfig = ServerAgentConfig()
    ) -> None:

        self.gcn_agent_config = gcn_agent_config
        self._create_logger()
        self._load_storage()
        self._load_parser()  # Initialize parameters used by parser


    def _create_logger(self) -> None:
        kwargs = {}
        if hasattr(self.gcn_agent_config.gcn_listener_configs, "gcn_logging_output_dirname"):
            kwargs["file_dir"] = self.gcn_agent_config.gcn_listener_configs.gcn_logging_output_dirname
        if hasattr(self.gcn_agent_config.gcn_listener_configs, "gcn_logging_output_filename"):
            kwargs["file_name"] = self.gcn_agent_config.gcn_listener_configs.gcn_logging_output_filename
        self.logger = ServerAgentFileLogger(**kwargs)


    def _load_parser(self) -> None:
        """
        Load parser and initialize parameters
        """

        self.parser: GCNParser = GCNParser(
            self.gcn_agent_config,
            self.storage,
            self.logger,
        )
        
    def _load_storage(self) -> None:
        """
        Load storage object and initialize parameters
        """

        self.storage: EventStorage = EventStorage(
            self.gcn_agent_config,
            self.logger,
        )
    