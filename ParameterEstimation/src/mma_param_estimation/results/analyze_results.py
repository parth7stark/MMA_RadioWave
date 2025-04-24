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
        result_path = self.estimator_config.bns_parameter_estimation_configs.dingo_configs.result_dir
        self.result = Result(file_name=result_path)
        self.samples = pd.DataFrame(self.result.samples)

        df = self.samples

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

    def plot_corner(self, param_names=None, output_path="corner_plot_all.png"):
        """
        Plot a corner plot using the full sample set.

        Args:
            param_names (list[str], optional): List of parameter names to include in the plot.
                                            If None, all columns in the samples DataFrame are used.
            output_path (str): File path to save the plot.
            true_values (list[float], optional): Ground truth values for overlay.
        """
        if param_names is None:
            param_names = list(self.samples.columns)

        sample_subset = self.samples[param_names]

        fig = corner.corner(
            sample_subset,
            labels=param_names,
            show_titles=True,
            title_fmt=".2f",
            quantiles=[0.05, 0.5, 0.95],
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 14},
            plot_datapoints=True,
            fill_contours=True,
            levels=(0.5, 0.9),
            smooth=1.0,
            smooth1d=1.0,
            figsize=(4, 4)
        )

        fig.savefig(output_path)
        print(f" Saved corner plot: {output_path}")
        plt.close()