import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
from dingo.gw.result import Result

class AnalyzeResults():
    """
    AnalyzeResults:
        This class contains functions to:
        - estimate best paramaters values using posterior samples
        - 
    """  
    def __init__(
        self,
        estimator_config,
        logger,
        **kwargs
    ):
        
        self.estimator_config = estimator_config
        self.logger = logger
        self.__dict__.update(kwargs)
        
        # For now hardcode the result path in ini file
        # result_path = self.estimator_config.bns_parameter_estimation_configs.dingo_configs.result_dir
        # self.result = Result(file_name=result_path)
        # self.samples = pd.DataFrame(self.result.samples)
        self.summary = {}

    def _get_median_ci(self, data: pd.Series, q_low=5, q_high=95):
        median = np.median(data)
        low, high = np.percentile(data, [q_low, q_high])
        return median, low, high

    def compute_summary_statistics(self):
        # For now hardcode the result path in ini file
        samples_filepath = self.estimator_config.bns_parameter_estimation_configs.dingo_configs.samples_filepath
        self.result = Result(file_name=samples_filepath)
        self.samples = pd.DataFrame(self.result.samples)

        df = self.samples
        print("Columns/parameters:", df.columns)

        """
        COlumns/parameters: Index(['delta_chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2',
        'phi_12', 'phi_jl', 'theta_jn', 'luminosity_distance', 'geocent_time',
        'psi', 'lambda_1', 'lambda_2', 'log_prob', 'delta_log_prob_target',
        'chirp_mass_proxy', 'ra', 'dec', 'chirp_mass', '_log_likelihood',
        'phase', 'log_likelihood', 'log_prior', 'weights'],
            dtype='object')

        """

        # Inclination angle
        theta_m, theta_l, theta_h = self._get_median_ci(df['theta_jn'])
        self.summary["Inclination (deg)"] = (np.degrees(theta_m), np.degrees(theta_l), np.degrees(theta_h))

        # Luminosity distance
        dl_m, dl_l, dl_h = self._get_median_ci(df['luminosity_distance'])
        self.summary["Luminosity Distance (Mpc)"] = (dl_m, dl_l, dl_h)

        # Sky location
        ra_m, ra_l, ra_h = self._get_median_ci(df['ra'])
        dec_m, dec_l, dec_h = self._get_median_ci(df['dec'])
        self.summary["RA (deg)"] = (np.degrees(ra_m), np.degrees(ra_l), np.degrees(ra_h))
        self.summary["Dec (deg)"] = (np.degrees(dec_m), np.degrees(dec_l), np.degrees(dec_h))

        # Masses
        m1, m2 = self._chirp_mass_and_q_to_masses(df['chirp_mass'], df['mass_ratio'])
        m1_m, m1_l, m1_h = self._get_median_ci(m1)
        m2_m, m2_l, m2_h = self._get_median_ci(m2)
        self.summary["Primary Mass (M☉)"] = (m1_m, m1_l, m1_h)
        self.summary["Secondary Mass (M☉)"] = (m2_m, m2_l, m2_h)

        # Effective spin
        chi_eff = (m1 * df['a_1'] + m2 * df['a_2']) / (m1 + m2)
        chi_m, chi_l, chi_h = self._get_median_ci(chi_eff)
        self.summary["Effective Spin"] = (chi_m, chi_l, chi_h)

        return self.summary

    def _chirp_mass_and_q_to_masses(self, chirp_mass, q):
        m1 = chirp_mass * (1 + q)**(1/5) * q**(-3/5)
        m2 = m1 * q
        return m1, m2

    def plot_corner(self, param_names=None, output_path="corner_plot_all.png", style="custom", use_weights=True):
        """
        Plot a corner plot using the full sample set.

        Args:
            param_names (list[str], optional): List of parameter names to include in the plot.
                                            If None, all columns in the samples DataFrame are used.
            output_path (str): File path to save the plot.
            style (str): 'custom' (your default style) or 'dingo' (Dingo BNS style).
             use_weights (bool): If True and 'weights' column exists, use importance sampling weights in the corner plot.
        """
        df = self.samples.copy()
        if param_names is None:
            # param_names = list(self.samples.columns)
           # Filter: exclude constant columns and columns with NaN or Inf
            param_names = [
            col for col in df.columns
            if df[col].nunique() > 1 and np.isfinite(df[col]).all()
        ]
        self.logger.info(f"Excluding constant or invalid columns. Using: {param_names}")

       
        sample_subset = df[param_names]

        # Check and extract weights
        weights = None
        if use_weights and "weights" in df.columns:
            weights = df["weights"].to_numpy()
            self.logger.info("Using importance sampling weights in plot.")

        # ----- Set corner plot parameters based on style -----
        if style == "dingo":
            corner_params = {
                "smooth": 1.0,
                "smooth1d": 1.0,
                "plot_datapoints": False,
                "plot_density": False,
                "plot_contours": True,
                "levels": [0.5, 0.9],
                "bins": 30,
                "no_fill_contours": True,
            }
        else:  # Default to 'custom' (your style)
            corner_params = {
                "smooth": 1.0,
                "smooth1d": 1.0,
                "plot_datapoints": True,
                "fill_contours": True,
                "plot_density": True,
                "levels": (0.5, 0.9),
                "show_titles": True,
                "title_fmt": ".2f",
                "title_kwargs": {"fontsize": 12},
                "label_kwargs": {"fontsize": 14},
                "figsize": (4, 4)
            }

        fig = corner.corner(
            sample_subset,
            labels=param_names,
            weights=weights,
            **corner_params
        )
        fig.savefig(output_path)
        print(f" Saved corner plot: {output_path}")
        # self.logger.info(f"Saved corner plot ({style} style): {output_path}")
        plt.close()
    
    def get_posterior_samples(self, param_names):
        """
        Extract posterior samples for specified parameters.

        Args:
            param_names (list[str]): List of parameter names to extract.

        Returns:
            dict: A dictionary where keys are parameter names and values are lists of sampled values.
        """
        df = self.samples.copy()

        # Check that all requested parameters exist
        missing = [p for p in param_names if p not in df.columns]
        if missing:
            raise ValueError(f"Requested parameters not found in samples: {missing}")

        selected_samples_df = df[param_names]

        # Convert DataFrame to dictionary (column: list of values)
        # posterior_df = {col: selected_samples[col].tolist() for col in selected_samples.columns}

        self.logger.info(f" Extracted posterior samples for parameters: {param_names}")
        return selected_samples_df