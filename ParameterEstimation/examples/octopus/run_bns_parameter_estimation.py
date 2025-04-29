import argparse
from omegaconf import OmegaConf
from mma_param_estimation.agent import ParamEstimatorAgent
from mma_param_estimation.communicator.octopus import OctopusEstimatorCommunicator


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", 
    type=str, 
    default="examples/configs/bns_parameter_estimation_config.yaml",
    help="Path to the configuration file."
)

args = argparser.parse_args()

# Load config from YAML (via OmegaConf)
estimator_config = OmegaConf.load(args.config)

# Initialize estimator-side modules
estimator_agent = ParamEstimatorAgent(estimator_config=estimator_config)

# Create Octopus communicator for publishing events to Radio topic - 
octopuscommunicator = OctopusEstimatorCommunicator(
    estimator_agent,
    logger=estimator_agent.logger,
)

# Commenting as Octopus doesn't work on Polaris compute node
# octopuscommunicator.publish_estimator_started_event()

# print("[GCN Listener] Started listening for LVK notices and circulars...", flush=True)
estimator_agent.logger.info("[Parameter Estimator] Started Parameter Estimation Process...")


# Step 1: Download or verify that the model & data are present
# downloader = DataDownloader(download_dir="downloads")
downloaded_paths = estimator_agent.downloader.ensure_all()

model_path = downloaded_paths["model"]
psd_files = downloaded_paths["psds"]
strain_files = downloaded_paths["frames"]

# Ensure these same paths are mentioned in ini file

estimator_agent.logger.info("\n Ensure following data paths are used in the Dingo INI file:")
estimator_agent.logger.info(f" Model path: {model_path}")
for det, psd_path in psd_files.items():
    estimator_agent.logger.info(f"  PSD path for {det}: {psd_path}")
for det, strain_path in strain_files.items():
    estimator_agent.logger.info(f" Strain frame for {det}: {strain_path}")

estimator_agent.logger.info("\n Please verify that these paths match those specified in your Dingo .ini configuration file.")


# Step 2: Run Dingo‐BNS using ini file
ini_filepath =  estimator_config.bns_parameter_estimation_configs.dingo_configs.ini_filepath
estimator_agent.run_dingo_pipe(ini_filepath)

# analyzer = AnalyzeResults(result_path)
# move this to agent file

# a) Print summary stats
summary = estimator_agent.analyzer.compute_summary_statistics()
for k, (med, low, high) in summary.items():
    print(f"{k}: {med:.3f} [{low:.3f}, {high:.3f}, 90%CI]")

# b) Full corner plot
# Full corner plot with all parameters
result_dir =  estimator_config.bns_parameter_estimation_configs.dingo_configs.result_dir
estimator_agent.analyzer.plot_corner(output_path=result_dir + "/corner_plot_all_parameters.png", style="dingo")

# Corner plot with a selected subset
param_list = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'theta_jn', 'luminosity_distance']
estimator_agent.analyzer.plot_corner(
    param_names=param_list,
    output_path=result_dir + "/corner_plot_subset_parameters.png"    
)

# c) plot corner plot between Distance vs angle
estimator_agent.analyzer.plot_corner(
    param_names=["theta_jn", "luminosity_distance"],
    output_path=result_dir + "/corner_theta_distance.png"
)

# Send posterior samples to mma topic for overlap analysis
