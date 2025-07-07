import argparse
from omegaconf import OmegaConf
from mma_ai_parser.agent import AIParserAgent
from mma_ai_parser.communicator.octopus import OctopusAIParserCommunicator
from openai import OpenAI
from pathlib import Path
import pickle
from datetime import datetime



argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", 
    type=str, 
    default="examples/configs/ai_parser_config.yaml",
    help="Path to the configuration file."
)

args = argparser.parse_args()

# Load config from YAML (via OmegaConf)
ai_parser_config = OmegaConf.load(args.config)

# Initialize ai_parser-side modules
ai_parser_agent = AIParserAgent(ai_parser_config=ai_parser_config)

# Create Octopus communicator for publishing events to Radio topic - 
octopuscommunicator = OctopusAIParserCommunicator(
    ai_parser_agent,
    logger=ai_parser_agent.logger,
)

# Commenting as Octopus doesn't work on Polaris compute node
octopuscommunicator.publish_ai_parser_started_event()

# print("[GCN Listener] Started listening for LVK notices and circulars...", flush=True)
ai_parser_agent.logger.info("[AI Parser] Started AI Parser...")

# Start the AI Parser process
if ai_parser_config.ai_parser_configs.llm_parameters.simulate_events == "yes":
    print("[Simulation Mode] Reading and loading GCNs from a folder...")
    ai_parser_agent.logger.info("[Simulation Mode] Reading and loading GCNs from a folder...")
    
    # Create the client and pass it the api_key
    client = OpenAI(
        api_key=ai_parser_config.ai_parser_configs.llm_parameters.gpt_api_key,
    )

    """
    This is the system prompt - these are the instructions which are passed to the LLM reader which pertain to all instances of the 
    """


    # SYSTEM_PROMPT = (
    #     "You are an expert API for extracting structured radio astronomy data from GCN circulars. "
    #     "You respond strictly in valid, parsable JSON, with no explanatory text, no Markdown, and no comments. "
    #     "If information is missing, omit that key."
    # )


    data_path = ai_parser_config.ai_parser_configs.llm_parameters.Gw170817_GCN_data_path
    save_folder = ai_parser_config.ai_parser_configs.llm_parameters.result_dir
    run_name = ai_parser_config.ai_parser_configs.llm_parameters.run_name

    radio_filenames = ai_parser_agent.parser.find_and_copy_radio_gcns(octopuscommunicator, data_path, radio_gcn_folder = save_folder + run_name + '_radiofiles')


    #change the order
    dir_data = Path(save_folder + run_name + '_radiofiles')
    radio_filenames = ai_parser_agent.parser.get_txt_filenames(dir_data)

    print('radio filenames:')
    print(radio_filenames)
    ai_parser_agent.logger.info(f"radio filenames: {radio_filenames}")
    

    alldata = []

    for filename in radio_filenames:
        data_from_thisgcn = ai_parser_agent.parser.load_and_process_GCN_rephrase(
            path_GCN = dir_data / filename,
            client=client,
        )
        
        alldata.extend(data_from_thisgcn)



    with open(save_folder + 'alldata_noflags_' + run_name + '.pkl', 'wb') as f:
        pickle.dump(alldata, f)

    cleaned_data = [ai_parser_agent.parser.clean_data(data) for data in alldata]

    print(cleaned_data)
    ai_parser_agent.logger.info(f"cleaned data: {cleaned_data}")


    with open(save_folder + 'cleaned_data_' + run_name + '.pkl', 'wb') as f:
        pickle.dump(cleaned_data, f)


    #make a datetime object for the GW
    yyyy = ai_parser_config.ai_parser_configs.llm_parameters.yyyy
    mm = ai_parser_config.ai_parser_configs.llm_parameters.mm
    dd = ai_parser_config.ai_parser_configs.llm_parameters.dd
    hh = ai_parser_config.ai_parser_configs.llm_parameters.hh
    minute = ai_parser_config.ai_parser_configs.llm_parameters.minute
    sec = ai_parser_config.ai_parser_configs.llm_parameters.sec

    GW_dt = datetime(int(yyyy), int(mm), int(dd), int(hh), int(minute), int(sec))


    ai_parser_agent.parser.dicts_to_CSV_for_fitting(octopuscommunicator, cleaned_data, output_csv = save_folder +run_name + '_for_fitting.csv', GW_dt = GW_dt)
