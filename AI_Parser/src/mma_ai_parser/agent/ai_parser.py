from mma_ai_parser.config import ServerAgentConfig
from mma_ai_parser.logger import ServerAgentFileLogger
from mma_ai_parser.parser import GCNParser
from omegaconf import OmegaConf, DictConfig


class AIParserAgent:
    """
    Contain functions that AI GCN parser performs like
    - Collecting GCN
    - Parsing and extracting flux-time data point
    - Storing information in CSV
   
    User can overwrite any class method to customize the behavior of the server agent.
    """
    def __init__(
        self,
        ai_parser_config: ServerAgentConfig = ServerAgentConfig()
    ) -> None:

        self.ai_parser_config = ai_parser_config
        self._create_logger()
        self._load_gcn_parser()


    def _create_logger(self) -> None:
        kwargs = {}
        if hasattr(self.ai_parser_config.ai_parser_configs, "logging_output_dirname"):
            kwargs["file_dir"] = self.ai_parser_config.ai_parser_configs.logging_output_dirname
        if hasattr(self.ai_parser_config.ai_parser_configs, "logging_output_filename"):
            kwargs["file_name"] = self.ai_parser_config.ai_parser_configs.logging_output_filename
        self.logger = ServerAgentFileLogger(**kwargs)


    def _load_gcn_parser(self) -> None:
        """
        Load result analyzer object and initialize parameters
        """

        self.parser: GCNParser = GCNParser(
            self.ai_parser_config,
            self.logger,
        )
    