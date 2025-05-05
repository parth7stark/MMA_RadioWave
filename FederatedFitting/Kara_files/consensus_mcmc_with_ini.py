import afterglowpy as grb
import emcee
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import math
import pickle
import matplotlib.colors as mcolors
import gc
import sys
from emcee import moves as mvs
import corner
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import shutil
import configparser
import ast
import os
from astropy.coordinates import SkyCoord

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from scipy.optimize import brentq

#run command:
#python with_ini.py --ini_file_path ./test_ini.ini


#Function to parse the .ini file:
def parse_value(val):
    # Try to interpret booleans, numbers, and lists
    val = val.strip()
    if val.lower() == 'true':
        return True
    elif val.lower() == 'false':
        return False
    try:
        return ast.literal_eval(val)  # handles lists, numbers, etc.
    except:
        return val  # fallback to string
            
def parse_ini(filename):
    config = configparser.ConfigParser()
    config.read(filename)

    parsed = {}
    for section in config.sections():
        parsed[section] = {}
        for key, value in config[section].items():
            parsed[section][key] = parse_value(value)

    return parsed

def argparser():
    """
    An arg parser such that we can pass in our ini_file_path to the parameters file.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ini_file_path", type=str, required=True,
        help="Path to the .ini configuration file."
    )
    return parser

#parse the parameters file
parser = argparser()
args = parser.parse_args()
ini_file_path = args.ini_file_path
params_from_ini = parse_ini(ini_file_path)

# Unpack sampling_parameters to variables
for key, value in params_from_ini['sampling_parameters'].items():
    globals()[key] = value

# Unpack files to variables
for key, value in params_from_ini['files'].items():
    globals()[key] = value

# Unpack source_and_flag_info to variables
for key, value in params_from_ini['source_and_flag_info'].items():
    globals()[key] = value

#save a copy of the .ini to the save folder:
shutil.copy(ini_file_path, os.path.join(save_folder, run_name + '_params.ini'))

# Function to save sampler state
def save_sampler_state(sampler, filename):
    with open(filename, "wb") as f:
        pickle.dump(sampler, f)

radio_bands = {
    "L-band": (1e9, 2e9, 1.5e9, 'o'),
    "S-band": (2e9, 4e9, 3e9, 's'),
    "C-band": (4e9, 8e9, 6e9, '^'),
    "X-band": (8e9, 12e9, 10e9, '>'),
    "Ku-band": (12e9, 18e9, 15e9, '*'),
    "K-band": (18e9, 26.5e9, 22e9, 'D'),
    "Ka-band": (26.5e9, 40e9, 33e9, 'P')
}

def process_RA_Dec_constraints(data):
    
    """
    Here we examine the data and reject all which falls outside of the user's specified arcsecond_uncertainty
    """
    
    #Collect the indices we keep:
    #If True, we use the data. If False, we exclude it.
    indices_not_flagged_for_exclusion_RA_Dec = []
    
    if "RA" in data.columns and "Dec" in data.columns and exclude_outside_ra_dec_uncertainty == True:
            
        for i in range(data.shape[0]):
            #These are the RA and Dec given by the data.
            RA_str = data.iloc[i]["RA"]
            Dec_str = data.iloc[i]["Dec"]

            #put the units in:
            RA_str = RA_str.split(':')
            RA = RA_str[0] + 'h' + RA_str[1] + 'm' + RA_str[2] + 's' 

            #put the units in:
            Dec_str = Dec_str.split(':')
            Dec = Dec_str[0] + 'd' + Dec_str[1] + 'm' + Dec_str[2] + 's'

            coordinates_from_data = SkyCoord(RA, Dec)
            coordinates_from_user = SkyCoord(ra, dec)

            # Calculate angular separation
            separation = coordinates_from_data.separation(coordinates_from_user)
            separation = separation.to(u.arcsec).value
            
            if separation <= arcseconds_uncertainty:
                indices_not_flagged_for_exclusion_RA_Dec.append(True) #we keep it
            if separation > arcseconds_uncertainty:
                indices_not_flagged_for_exclusion_RA_Dec.append(False) #we toss it
        return indices_not_flagged_for_exclusion_RA_Dec

    else:
        #Then we use all the data
        for i in range(data.shape[0]):
            indices_not_flagged_for_exclusion_RA_Dec.append(True)
        return indices_not_flagged_for_exclusion_RA_Dec

def process_flags(data):
    """
    Process all of the flags given in the .csv file into an array indicating which data points we discard.
    """
    
    #Collect the indices we keep:
    #If True, we use the data. If False, we exclude it.
    indices_not_flagged_for_exclusion = []

    #handle each flag:
    if "time_flag" in data.columns and exclude_time_flag == True:
        use_time_flag = True
    else:
        use_time_flag = False

    if "RA_Dec_flag" in data.columns and exclude_ra_dec_flag == True:
        use_RA_Dec_flag = True
    else:
        use_RA_Dec_flag = False

    if "name_flag" in data.columns and exclude_name_flag == True:
        use_name_flag = True
    else:
        use_name_flag = False
        
    for i in range(data.shape[0]):
        flags = []
        if use_time_flag:
            time_flag = data.iloc[i]["time_flag"]
            flags.append(time_flag)
        if use_RA_Dec_flag:
            RA_Dec_flag = data.iloc[i]["RA_Dec_flag"]
            flags.append(RA_Dec_flag)
        if use_name_flag:
            name_flag = data.iloc[i]["name_flag"]
            flags.append(name_flag)
            
        flags = np.asarray(flags)

        #This means it contains a nonzero flag and we exclude it
        if np.sum(flags) > 0:
            indices_not_flagged_for_exclusion.append(False)
        else:
            indices_not_flagged_for_exclusion.append(True)
    
    return indices_not_flagged_for_exclusion


def interpret(data):
    print('interpret data')
    if "days" in data.columns:
        data["t"] = data["days"] * 86400
    elif "seconds" in data.columns:
        data["t"] = data["seconds"]
    elif "t_delta" in data.columns:
        data["t"] = data["t_delta"]
    
    if "Filter" in data.columns:
        data["filter"] = data["Filter"]
    elif "Band" in data.columns:
        data["filter"] = data["Band"]
    elif "band" in data.columns:
        data["filter"] = data["band"]
    
    if "GHz" in data.columns:
        data["frequency"] = data["GHz"]
        freq_correct = 1e9
    if "Hz" in data.columns:
        data["frequency"] = data["Hz"]
        freq_correct = 1
    
    if "microJy" in data.columns:
        data["flux"] = data["microJy"]
        flux_correct = 1e-3
    elif "Jy" in data.columns:
        data["flux"] = data["Jy"]
        flux_correct = 1e3
    elif "mJy" in data.columns:
        data["flux"] = data["mJy"]
        flux_correct = 1
    elif "mag" in data.columns:
        data["flux"] = data["mag"]
        flux_correct = "mag"

    #use the RA and Dec radius:
    indices_not_flagged_for_exclusion_RA_Dec = process_RA_Dec_constraints(data)
    
    #use the flags:
    indices_not_flagged_for_exclusion = process_flags(data)
    
    freq, new_flux, err, tvals = [], [], [], []

    for i in range(data.shape[0]):
        if indices_not_flagged_for_exclusion[i] == True and indices_not_flagged_for_exclusion_RA_Dec[i] == True:
            try:
                this_freq = float(data.iloc[i]["frequency"]) * freq_correct
            except:
                continue  # skip this row entirely if frequency can't be parsed
    
            if "err" in data.columns:
                flux = data.iloc[i]["flux"]
                error = float(data.iloc[i]["err"])
                if "<" in flux:
                    print('skipping UL')
                    #new_flux.append("UL")
                    #err.append(0)
                elif ">" in flux:
                    print('skipping UL')
                    #new_flux.append("UL")
                    #err.append(0)
                else:
                    flux = float(flux)
                    new_flux.append(flux)
                    err.append(float(error))
                    freq.append(this_freq)
                    tvals.append(data.iloc[i]["t"])
            else:
                flux = data.iloc[i]["flux"]
    
                if "<" in flux:
                    continue
                    #new_flux.append("UL")
                    #err.append(0)
                elif ">" in flux:
                    continue
                    #new_flux.append("UL")
                    #err.append(0)
                
                if "±" in flux:
                    splt = flux.split("±")
                    new_flux.append(float(splt[0]))
                    err.append(float(splt[1]))
                    freq.append(this_freq)
                    tvals.append(data.iloc[i]["t"])
                elif "+-" in flux:
                    splt = flux.split("+-")
                    new_flux.append(float(splt[0]))
                    err.append(float(splt[1]))
                    tvals.append(data.iloc[i]["t"])
                    freq.append(this_freq)
                elif "+" in flux and "-" in flux:
                    splt = flux.split("+")
                    new_flux.append(float(splt[0]))
                    err_splt = splt[1].split("-")
                    err.append(max([float(splt[0]), float(err_splt[1])]))
                    freq.append(this_freq)
                    tvals.append(data.iloc[i]["t"])
                else:
                    new_flux.append(float(flux))
                    err.append(0)
                    freq.append(this_freq)
                    tvals.append(data.iloc[i]["t"])
    
    for i in range(len(new_flux)):
        if new_flux[i] != "UL":
            if flux_correct != "mag":
                new_flux[i] = new_flux[i] * flux_correct
                err[i] = err[i] * flux_correct

    # Build a clean DataFrame only from valid, parsed values

    print(len(tvals))
    print(len(freq))
    print(len(new_flux))
    print(len(err))
    
    valid_data = pd.DataFrame({
    "t": tvals,
    "frequency": freq,
    "flux": new_flux,
    "err": err})
    
    # Return just the columns we want, fully cleaned
    valid_data = valid_data[["t", "frequency", "flux", "err"]].astype(np.float64)
    return valid_data


def interpret_ULs(data):
    print('interpret UL data')
    """
    This takes in the same datafile as interpret(), but it picks out the upper limits only. 
    We use this for the plotting.
    """
    if "days" in data.columns:
        data["t"] = data["days"] * 86400
    elif "seconds" in data.columns:
        data["t"] = data["seconds"]
    elif "t_delta" in data.columns:
        data["t"] = data["t_delta"]
    
    if "Filter" in data.columns:
        data["filter"] = data["Filter"]
    elif "Band" in data.columns:
        data["filter"] = data["Band"]
    elif "band" in data.columns:
        data["filter"] = data["band"]
    
    if "GHz" in data.columns:
        data["frequency"] = data["GHz"]
        freq_correct = 1e9
    if "Hz" in data.columns:
        data["frequency"] = data["Hz"]
        freq_correct = 1
    
    if "microJy" in data.columns:
        data["flux"] = data["microJy"]
        flux_correct = 1e-3
    elif "Jy" in data.columns:
        data["flux"] = data["Jy"]
        flux_correct = 1e3
    elif "mJy" in data.columns:
        data["flux"] = data["mJy"]
        flux_correct = 1
    elif "mag" in data.columns:
        data["flux"] = data["mag"]
        flux_correct = "mag"

    #use the RA and Dec radius:
    indices_not_flagged_for_exclusion_RA_Dec = process_RA_Dec_constraints(data)
    
    #use the flags:
    indices_not_flagged_for_exclusion = process_flags(data)

    freq, new_flux, tvals = [], [], []

    for i in range(data.shape[0]):
        if indices_not_flagged_for_exclusion[i] == True and indices_not_flagged_for_exclusion_RA_Dec[i] == True:
            try:
                this_freq = float(data.iloc[i]["frequency"]) * freq_correct
            except:
                continue  # skip this row entirely if frequency can't be parsed
    
            if "err" in data.columns:
                flux = data.iloc[i]["flux"]
                error = float(data.iloc[i]["err"])
                if "<" in flux:
                    flux_str = flux.split('<')
                    flux = flux_str[1]
                    flux = float(flux)
                    new_flux.append(flux)
                    freq.append(this_freq)
                    tvals.append(data.iloc[i]["t"])
                elif ">" in flux:
                    flux_str = flux.split('>')
                    flux = flux_str[1]
                    flux = float(flux)
                    new_flux.append(flux)
                    freq.append(this_freq)
                    tvals.append(data.iloc[i]["t"])
    
            else:
                flux = data.iloc[i]["flux"]
    
                if "<" in flux:
                    flux_str = flux.split('<')
                    flux = flux_str[1]
                    flux = float(flux)
                    new_flux.append(flux)
                    freq.append(this_freq)
                    tvals.append(data.iloc[i]["t"])
                elif ">" in flux:
                    flux_str = flux.split('>')
                    flux = flux_str[1]
                    flux = float(flux)
                    new_flux.append(flux)
                    freq.append(this_freq)
                    tvals.append(data.iloc[i]["t"])
    
    for i in range(len(new_flux)):
        if new_flux[i] != "UL":
            if flux_correct != "mag":
                new_flux[i] = new_flux[i] * flux_correct
                #err[i] = err[i] * flux_correct

    print(len(tvals))
    print(len(freq))
    print(len(new_flux))
    
    valid_data = pd.DataFrame({
    "t": tvals,
    "frequency": freq,
    "flux": new_flux})
    
    # Return just the columns we want, fully cleaned
    valid_data = valid_data[["t", "frequency", "flux"]].astype(np.float64)
    return valid_data


def find_redshift_from_DL(DL_cm, z_min=z_range[0], z_max=z_range[1]):
    # Convert cm to Mpc for comparison with astropy's internal units
    DL_target = DL_cm * u.cm
    def func(z):
        return (cosmo.luminosity_distance(z) - DL_target).value
    return brentq(func, z_min, z_max)


def log_likelihood(theta, nu, x, y, yerr):

    if z_known == True:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing = theta
        z = z_fixed
    else:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z = theta

    DL = cosmo.luminosity_distance(z).to(u.cm)
    DL = DL.value


    E0 = 10 ** logE0
    n0 = 10 ** logn0
    epsilon_e = 10 ** logepsilon_e
    epsilon_B = 10 ** logepsilon_B


    Z = {
    "jetType": grb.jet.Gaussian,  # Jet type
    "specType": 0,  # Basic Synchrotron Emission Spectrum
    "thetaObs": thetaObs, #35.2/180.0 * math.pi,  # Viewing angle in radians
    "E0": E0, #1.5e53*9.8e-3,  # Isotropic-equivalent energy in erg
    "thetaCore": thetaCore, #0.25,  # Half-opening angle in radians
    "thetaWing": thetaWing,  # "wing" truncation angle of the jet, in radians
    "n0": n0, #9.8e-3,  # circumburst density in cm^-3
    "p": p, #2.168,  # electron energy distribution index
    #"b": 6.0,  # power law structure index
    "epsilon_e": epsilon_e, #7.8e-3,  # epsilon_e
    "epsilon_B": epsilon_B, #9.9e-4,  # epsilon_B
    "xi_N": 1.0,  # Fraction of electrons accelerated
    "d_L": DL,  # Luminosity distance in cm of 40Mpc
    "z": z, #40Mpc
    }
    

    try:    

        model = grb.fluxDensity(x, nu, **Z)

        sigma2 = yerr**2
    
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log10(sigma2))
        

    except:

        #Just put in different parameters to make it run.
        print('enters except block')

        Z = {
        "jetType": grb.jet.Gaussian,  # Jet type
        "specType": 0,  # Basic Synchrotron Emission Spectrum
        "thetaObs": math.pi/4.0, #35.2/180.0 * math.pi,  # Viewing angle in radians
        "E0": 1e52, #1.5e53*9.8e-3,  # Isotropic-equivalent energy in erg
        "thetaCore": math.pi/10.0, #0.25,  # Half-opening angle in radians
        "thetaWing": math.pi/3.0,  # "wing" truncation angle of the jet, in radians
        "n0": 5e-3, #9.8e-3,  # circumburst density in cm^-3
        "p": 2.5, #2.168,  # electron energy distribution index
        #"b": 6.0,  # power law structure index
        "epsilon_e": .1, #7.8e-3,  # epsilon_e
        "epsilon_B": .01, #9.9e-4,  # epsilon_B
        "xi_N": 1.0,  # Fraction of electrons accelerated
        "d_L": 1.2344e26,  # Luminosity distance in cm of 40Mpc
        "z": 0.00897, #40Mpc
        }

        model = grb.fluxDensity(x, nu, **Z)

        sigma2 = yerr**2
    
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log10(sigma2))

def log_prior(theta):
    # print("prior theta: ", theta)
    # exit()
    # theta is in log space
    # rename below variables accordingly
    # E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p, thetaWing = theta
    if z_known:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing = theta
    else:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z = theta

    if z_known:
        if (
            loge0_range[0] <= logE0 <= loge0_range[1]
            and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
            and thetawing_range[0] <= thetaWing < thetawing_range[1]
            and thetacore_range[0] <= thetaCore < thetacore_range[1]
            and p_range[0] < p < p_range[1]
            and logepsilon_e_range[0] < logepsilon_e <= logepsilon_e_range[1]
            and logepsilon_e_range[0] < logepsilon_B <= logepsilon_e_range[1]
            and logn0_range[0] < logn0 < logn0_range[1]
        ):
            return 0.0
        return -np.inf

    else:
        if (
            loge0_range[0] <= logE0 <= loge0_range[1]
            and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
            and thetawing_range[0] <= thetaWing < thetawing_range[1]
            and thetacore_range[0] <= thetaCore < thetacore_range[1]
            and p_range[0] < p < p_range[1]
            and logepsilon_e_range[0] < logepsilon_e <= logepsilon_e_range[1]
            and logepsilon_e_range[0] < logepsilon_B <= logepsilon_e_range[1]
            and logn0_range[0] < logn0 < logn0_range[1]
            and z_range[0] < z < z_range[1]
        ):
            return 0.0
        return -np.inf


def log_probability(theta, nu, x, y, yerr):
    # print("1st theta: ", theta)
    # exit()
    # theta is in log space
    lp = log_prior(theta)
    # print("prior return value: ", lp)
    # exit()
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, nu, x, y, yerr)


########################
# CONSENSUS MCMC FUNCTIONS
#########################

def split_data_by_site(data, num_sites=3):
    """
    Split data into num_sites parts, roughly equal in size.
    Adds a 'site_id' column to the data.
    """
    n_samples = len(data)
    data = data.copy()  # Create a copy to avoid modifying the original
    
    # Assign site IDs to each data point (0, 1, 2 for 3 sites)
    data['site_id'] = np.random.randint(0, num_sites, size=n_samples)
    
    # Split the data into a list of dataframes, one per site
    site_data = [data[data['site_id'] == i].copy() for i in range(num_sites)]
    
    return site_data

def run_local_mcmc(data, nwalkers, ndim, niter, burnin, lower_bounds, upper_bounds, nu, t, fnu, err, z_known, site_id):
    """
    Run MCMC for a specific site using only its local data
    """
    print(f"Running MCMC for site {site_id}")
    
    # Initialize walkers with uniform random positions
    pos = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(nwalkers, ndim))
    
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(nu, t, fnu, err), pool=pool,
            moves=[(mvs.StretchMove(a=1.1), 0.7), (mvs.WalkMove(10), 0.3)])
        
        print(f'Site {site_id}: doing sampling')
        state = sampler.run_mcmc(pos, niter, progress=True)
    
    gc.collect()
    
    # Save site-specific results
    site_folder = os.path.join(save_folder, f"site_{site_id}")
    os.makedirs(site_folder, exist_ok=True)
    
    save_sampler_state(sampler, f"{site_folder}/sampler.pkl")
    flat_samples = sampler.get_chain(discard=burnin, flat=True)
    np.save(f"{site_folder}/{run_name}_flat_samples.npy", flat_samples)
    log_prob = sampler.get_log_prob(flat=False)
    np.save(f"{site_folder}/{run_name}_log_prob.npy", log_prob)
    
    # Create site-specific plots
    if z_known:
        params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing']
    else:
        params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing', 'z']
    
    make_posterior_hists(flat_samples, burnin, nwalkers, ndim, params, f"{site_folder}/{run_name}_PosteriorHists.png")
    make_corner_plots(flat_samples, burnin, nwalkers, ndim, params, true_values, display_truths_on_corner, f"{site_folder}/{run_name}_CornerPlots.png")
    make_Log_Likelihood_plot(log_prob, burnin, nwalkers, f"{site_folder}/{run_name}_LogProb.png")
    
    # Return the samples for consensus
    return flat_samples, params

def aggregate_gaussian(chains):
    """
    Approximate each subposterior as Gaussian and compute the
    product of Gaussians:
      Sigma_full = (sum_i inv(Sigma_i))^-1,
      mu_full = Sigma_full * (sum_i inv(Sigma_i)*mu_i)
    """
    means = []
    covs = []
    for chain in chains:
        mu = np.mean(chain, axis=0)
        cov = np.cov(chain, rowvar=False)
        means.append(mu)
        covs.append(cov)
    
    precision_sum = np.zeros_like(covs[0])
    weighted_mean_sum = np.zeros_like(means[0])
    
    for mu, cov in zip(means, covs):
        # try:
        inv_cov = np.linalg.inv(cov)
        precision_sum += inv_cov
        weighted_mean_sum += inv_cov @ mu
        # except np.linalg.LinAlgError:
        #     print("Warning: Singular matrix encountered in covariance inversion")
        #     # Add a small regularization to handle potential singular matrices
        #     reg_cov = cov + np.eye(cov.shape[0]) * 1e-8
        #     inv_cov = np.linalg.inv(reg_cov)
        #     precision_sum += inv_cov
        #     weighted_mean_sum += inv_cov @ mu
    
    Sigma_full = np.linalg.inv(precision_sum)
    mu_full = Sigma_full @ weighted_mean_sum
    
    return mu_full, Sigma_full

def generate_consensus_samples(mu_full, Sigma_full, n_samples=10000):
    """
    Generate samples from the consensus posterior distribution
    """
    return np.random.multivariate_normal(mu_full, Sigma_full, size=n_samples)

#########################
# MODIFIED PLOTTING FUNCTIONS
#########################

def plot_lc_noUL(data_list, theta, site_colors=None):
    """
    Modified to plot data points from different sites with different colors
    """
    # Combine all data for finding min/max times
    all_data = pd.concat(data_list)
    all_data = all_data.sort_values(by=["frequency"], ascending=False)

    # Extract and transform model parameters
    if z_known:
        E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p, thetaWing = theta
        z = z_fixed
    else:
        E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p, thetaWing, z = theta
        
    E0 = 10 ** E0
    n0 = 10 ** n0
    epsilon_e = 10 ** epsilon_e
    epsilon_B = 10 ** epsilon_B

    DL = cosmo.luminosity_distance(z).to(u.cm)
    DL = DL.value

    Z = {
        "jetType": grb.jet.Gaussian,
        "specType": 0,
        "thetaObs": thetaObs,
        "E0": E0,
        "thetaCore": thetaCore,
        "thetaWing": thetaWing,
        "n0": n0,
        "p": p,
        "epsilon_e": epsilon_e,
        "epsilon_B": epsilon_B,
        "xi_N": 1.0,
        "d_L": DL,
        "z": z,
    }

    times = np.geomspace(min(all_data["t"]), max(all_data["t"]), 100)
    fig, ax = plt.subplots(figsize=(10, 7))

    band_colors_list = ['black', 'red', 'blue', 'green', 'magenta']
    site_markers = ['o', 's', '^', 'v', 'D']  # Different marker for each site
    legend_handles = []

    # First plot model curves for all bands
    for i, (band, (fmin, fmax, fcen, marker)) in enumerate(radio_bands.items()):
        # Filter combined data for this band
        band_data_exists = False
        for data in data_list:
            if not data[(data["frequency"] >= fmin) & (data["frequency"] < fmax)].empty:
                band_data_exists = True
                break
        
        if band_data_exists:
            color = band_colors_list[i % len(band_colors_list)]
            # Plot model curve for this band
            model = grb.fluxDensity(times, np.full(times.shape, fcen), **Z)
            ax.plot(times / (24 * 60.0 * 60), model, color=color, linewidth=1.2)
            
            # Create legend handle for band
            legend_label = f"{band} ({fcen/1e9:.1f} GHz)"
            legend_handles.append(Line2D(
                [0], [0],
                marker=None,
                color=color,
                linestyle='-',
                linewidth=1.2,
                label=legend_label
            ))

    # Then plot data points from each site with different markers
    for site_id, data in enumerate(data_list):
        if site_colors is None:
            site_color = 'k'  # Default color
        else:
            site_color = site_colors[site_id % len(site_colors)]
        
        site_marker = site_markers[site_id % len(site_markers)]
        
        for i, (band, (fmin, fmax, fcen, _)) in enumerate(radio_bands.items()):
            band_data = data[(data["frequency"] >= fmin) & (data["frequency"] < fmax)]
            if not band_data.empty:
                color = band_colors_list[i % len(band_colors_list)]
                
                # Plot error bars
                band_t = np.array(band_data["t"]) / (24 * 60.0 * 60)
                band_fnu = np.array(band_data["flux"])
                band_err = np.array(band_data["err"])
                
                ax.errorbar(
                    band_t,
                    band_fnu,
                    yerr=band_err,
                    fmt='none',
                    ecolor=color,
                    elinewidth=0.6,
                    capsize=2,
                    alpha=0.8
                )
                
                # Overlay scatter points with site-specific markers
                ax.scatter(
                    band_t,
                    band_fnu,
                    marker=site_marker,
                    color=color,
                    edgecolor='black',
                    linewidth=0.5,
                    s=40,
                    alpha=0.9
                )
        
        # Add site marker to legend
        legend_handles.append(Line2D(
            [0], [0],
            marker=site_marker,
            color='black',
            linestyle='None',
            markerfacecolor='gray',
            markeredgecolor='black',
            linewidth=0,
            markersize=7,
            label=f"Site {site_id+1}"
        ))

    ax.set(
        xlabel="Time since detection (days)",
        ylabel="Flux density (mJy)",
        xscale="log",
        yscale="log",
        title=f"{plot_names} - Consensus Model"
    )

    # Create legend with two rows: bands in first row, sites in second row
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), 
              ncol=len(legend_handles)//2, frameon=False)
    
    plt.tight_layout()
    plt.savefig(save_folder + '/' + run_name + '_consensus_lightcurve.png', bbox_inches='tight')

def plot_lc_wUL(data_list, data_UL_list, theta, site_colors=None):
    """
    Modified to plot data points from different sites with different colors, including upper limits
    """
    # Combine all data for finding min/max times
    all_data = pd.concat(data_list)
    all_data = all_data.sort_values(by=["frequency"], ascending=False)
    
    all_data_UL = pd.concat(data_UL_list)
    all_data_UL = all_data_UL.sort_values(by=["frequency"], ascending=False)

    # Extract and transform model parameters
    if z_known:
        E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p, thetaWing = theta
        z = z_fixed
    else:
        E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p, thetaWing, z = theta
        
    E0 = 10 ** E0
    n0 = 10 ** n0
    epsilon_e = 10 ** epsilon_e
    epsilon_B = 10 ** epsilon_B

    DL = cosmo.luminosity_distance(z).to(u.cm)
    DL = DL.value

    Z = {
        "jetType": grb.jet.Gaussian,
        "specType": 0,
        "thetaObs": thetaObs,
        "E0": E0,
        "thetaCore": thetaCore,
        "thetaWing": thetaWing,
        "n0": n0,
        "p": p,
        "epsilon_e": epsilon_e,
        "epsilon_B": epsilon_B,
        "xi_N": 1.0,
        "d_L": DL,
        "z": z,
    }

    times = np.geomspace(min(all_data["t"]), max(all_data["t"]), 100)
    fig, ax = plt.subplots(figsize=(10, 7))

    band_colors_list = ['black', 'red', 'blue', 'green', 'magenta']
    site_markers = ['o', 's', '^', 'v', 'D']  # Different marker for each site
    legend_handles = []

    # First plot model curves for all bands
    for i, (band, (fmin, fmax, fcen, marker)) in enumerate(radio_bands.items()):
        # Filter combined data for this band
        band_data_exists = False
        for data in data_list:
            if not data[(data["frequency"] >= fmin) & (data["frequency"] < fmax)].empty:
                band_data_exists = True
                break
        
        if band_data_exists:
            color = band_colors_list[i % len(band_colors_list)]
            # Plot model curve for this band
            model = grb.fluxDensity(times, np.full(times.shape, fcen), **Z)
            ax.plot(times / (24 * 60.0 * 60), model, color=color, linewidth=1.2)
            
            # Create legend handle for band
            legend_label = f"{band} ({fcen/1e9:.1f} GHz)"
            legend_handles.append(Line2D(
                [0], [0],
                marker=None,
                color=color,
                linestyle='-',
                linewidth=1.2,
                label=legend_label
            ))

    # Then plot data points from each site with different markers
    for site_id, (data, data_UL) in enumerate(zip(data_list, data_UL_list)):
        if site_colors is None:
            site_color = 'k'  # Default color
        else:
            site_color = site_colors[site_id % len(site_colors)]
        
        site_marker = site_markers[site_id % len(site_markers)]
        
        # Plot regular data points
        for i, (band, (fmin, fmax, fcen, _)) in enumerate(radio_bands.items()):
            band_data = data[(data["frequency"] >= fmin) & (data["frequency"] < fmax)]
            if not band_data.empty:
                color = band_colors_list[i % len(band_colors_list)]
                
                # Plot error bars
                band_t = np.array(band_data["t"]) / (24 * 60.0 * 60)
                band_fnu = np.array(band_data["flux"])
                band_err = np.array(band_data["err"])
                
                ax.errorbar(
                    band_t,
                    band_fnu,
                    yerr=band_err,
                    fmt='none',
                    ecolor=color,
                    elinewidth=0.6,
                    capsize=2,
                    alpha=0.8
                )
                
                # Overlay scatter points with site-specific markers
                ax.scatter(
                    band_t,
                    band_fnu,
                    marker=site_marker,
                    color=color,
                    edgecolor='black',
                    linewidth=0.5,
                    s=40,
                    alpha=0.9
                )
        
        # Plot upper limits
        for i, (band, (fmin, fmax, fcen, _)) in enumerate(radio_bands.items()):
            band_data = data_UL[(data_UL["frequency"] >= fmin) & (data_UL["frequency"] < fmax)]
            if not band_data.empty:
                color = band_colors_list[i % len(band_colors_list)]
                
                band_t = np.array(band_data["t"]) / (24 * 60.0 * 60)
                band_fnu = np.array(band_data["flux"])
                
                # Overlay upper limit points with downward triangles
                ax.scatter(
                    band_t,
                    band_fnu,
                    marker='v',
                    facecolors='none',
                    edgecolors=color,
                    linewidth=0.5,
                    s=40,
                    alpha=0.9
                )
        
        # Add site marker to legend
        legend_handles.append(Line2D(
            [0], [0],
            marker=site_marker,
            color='black',
            linestyle='None',
            markerfacecolor='gray',
            markeredgecolor='black',
            linewidth=0,
            markersize=7,
            label=f"Site {site_id+1}"
        ))

    # Add upper limit to legend
    legend_handles.append(Line2D(
        [0], [0],
        marker='v',
        color='black',
        markerfacecolor='none',
        markeredgecolor='black',
        linewidth=0,
        markersize=7,
        label="Upper limit"
    ))

    ax.set(
        xlabel="Time since detection (days)",
        ylabel="Flux density (mJy)",
        xscale="log",
        yscale="log",
        title=f"{plot_names} - Consensus Model"
    )

    # Create legend with multiple rows
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), 
              ncol=3, frameon=False)
    
    plt.tight_layout()
    plt.savefig(save_folder + '/' + run_name + '_consensus_lightcurve.png', bbox_inches='tight')

def make_Log_Likelihood_plot(log_prob, burnin, nwalkers, save_path):
    """
    Modified to take save path as a parameter
    """
    plt.figure(figsize=(10, 6))
    for i in range(nwalkers):
        plt.plot(log_prob[:, i], alpha=0.3)
    
    plt.xlabel("Steps")
    plt.ylabel("Log Likelihood")
    plt.title("Log Likelihood Progression " + plot_names + " walkers = " + str(nwalkers))
    plt.ylim(-500, 120)
    plt.axvline(burnin, label='burnin = ' + str(burnin), color='g', lw='.5')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')

def make_corner_plots(samples, burnin, nwalkers, ndim, params, true_values, display_truths_on_corner, save_path):
    """
    Modified to take parameters as arguments
    """
    figure = corner.corner(
        samples[burnin * nwalkers:],
        labels=params,
        show_titles=True,
        title_fmt=".2f",
        quantiles=[0.05, 0.5, 0.95],  # 90% credible interval
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        plot_datapoints=True,
        fill_contours=True,
        levels=(0.50, 0.90,),  # 90% confidence contours
        smooth=1.0,
        smooth1d=1.0,
        truths=true_values if len(true_values) == ndim and display_truths_on_corner else None
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

def make_posterior_hists(samples, burnin, nwalkers, ndim, params, save_path):
    """
    Modified to take parameters as arguments
    """
    medians = np.median(samples[burnin * nwalkers:], axis=0)

    print('median parameter values after burnin:')
    for i in range(ndim):
        print(f"{params[i]}: {medians[i]:.4f}")
    
    # Create subplots
    if ndim == 8:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns
    if ndim == 9:
        fig, axes = plt.subplots(3, 3, figsize=(16, 10))  # 3 rows, 3 columns
        
    axes = axes.flatten()
    
    # Loop over each dimension and create a histogram
    theta = []
    
    for i in range(ndim):
        theta_component = np.asarray(samples[burnin * nwalkers:, i])
        lower, upper = np.percentile(theta_component, [15.865, 84.135])
    
        theta.append(theta_component)
        ax = axes[i]
        ax.hist(samples[burnin * nwalkers:, i], bins=20, color="blue", alpha=0.7, label="Samples")
    
        # Plot mean value as a vertical line
        ax.axvline(medians[i], color="red", linestyle="--", label=f"Median: {medians[i]:.4f}")
        ax.axvline(lower, color="green", linestyle="--", label=f"lower limit: {lower:.4f}")
        ax.axvline(upper, color="green", linestyle="--", label=f"upper limit: {upper:.4f}")
        
        ax.set_title(params[i])
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend(loc=4)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

#########################
# MAIN EXECUTION BLOCK
#########################

if __name__ == "__main__":
    
    # Parse parameters and load data
    data_from_csv = pd.read_csv(datafile)
    if data_from_csv.shape[1] == 1:
        data_from_csv = pd.read_csv(datafile, delim_whitespace=True)
    
    # Create consensys directory structure
    os.makedirs(save_folder, exist_ok=True)
    
    # Main processing pipeline
    data = interpret(data_from_csv)
    data_UL = interpret_ULs(data_from_csv)
    
    # Split data into sites
    num_sites = 3  # Number of sites to distribute data
    site_data_list = split_data_by_site(data, num_sites)
    site_data_UL_list = split_data_by_site(data_UL, num_sites)
    
    # Prepare MCMC parameters
    ndim = 8 if z_known else 9
    nwalkers = 32
    
    # Define parameter bounds
    if z_known:
        lower_bounds = np.array([
            loge0_range[0],        # log10(E0)
            thetaobs_range[0],     # thetaObs
            thetacore_range[0],    # thetaCore
            logn0_range[0],        # log10(n0)
            logepsilon_e_range[0], # log10(eps_e)
            logepsilon_b_range[0], # log10(eps_B)
            p_range[0],            # p
            thetawing_range[0]     # thetaWing
        ])
        
        upper_bounds = np.array([
            loge0_range[1],        # log10(E0)
            thetaobs_range[1],     # thetaObs
            thetacore_range[1],    # thetaCore
            logn0_range[1],        # log10(n0)
            logepsilon_e_range[1], # log10(eps_e)
            logepsilon_b_range[1], # log10(eps_B)
            p_range[1],            # p
            thetawing_range[1]     # thetaWing
        ])
    else:
        lower_bounds = np.array([
            loge0_range[0],        # log10(E0)
            thetaobs_range[0],     # thetaObs
            thetacore_range[0],    # thetaCore
            logn0_range[0],        # log10(n0)
            logepsilon_e_range[0], # log10(eps_e)
            logepsilon_b_range[0], # log10(eps_B)
            p_range[0],            # p
            thetawing_range[0],    # thetaWing
            z_range[0]             # z
        ])
        
        upper_bounds = np.array([
            loge0_range[1],        # log10(E0)
            thetaobs_range[1],     # thetaObs
            thetacore_range[1],    # thetaCore
            logn0_range[1],        # log10(n0)
            logepsilon_e_range[1], # log10(eps_e)
            logepsilon_b_range[1], # log10(eps_B)
            p_range[1],            # p
            thetawing_range[1],    # thetaWing
            z_range[1]             # z
        ])
    
    # Run MCMC for each site and collect samples
    all_site_samples = []
    
    for site_id in range(num_sites):
        site_data = site_data_list[site_id]
        
        # Extract data for this site
        t = np.array(list(site_data["t"]))
        nu = np.array(list(site_data["frequency"]))
        fnu = np.array(list(site_data["flux"]))
        err = np.array(list(site_data["err"]))
        
        # Run MCMC for this site
        site_samples, params = run_local_mcmc(
            site_data, nwalkers, ndim, niter, burnin, 
            lower_bounds, upper_bounds, nu, t, fnu, err, z_known, site_id
        )
        
        all_site_samples.append(site_samples)
    
    # Perform consensus aggregation
    print("Performing consensus aggregation...")
    mu_full, Sigma_full = aggregate_gaussian(all_site_samples)
    
    # Generate samples from consensus posterior
    consensus_samples = generate_consensus_samples(mu_full, Sigma_full, n_samples=10000)
    
    # Save consensus samples
    np.save(f"{save_folder}/{run_name}_consensus_samples.npy", consensus_samples)
    
    # Create consensus plots
    make_posterior_hists(
        consensus_samples, 0, 1, ndim, params, 
        f"{save_folder}/{run_name}_consensus_PosteriorHists.png"
    )
    
    make_corner_plots(
        consensus_samples, 0, 1, ndim, params, 
        true_values, display_truths_on_corner,
        f"{save_folder}/{run_name}_consensus_CornerPlots.png"
    )
    
    # Get mean consensus parameter values
    theta_consensus = np.median(consensus_samples, axis=0)
    
    # Plot consensus light curve
    site_colors = ['red', 'blue', 'green']
    if include_upper_limits_on_lc:
        plot_lc_wUL(site_data_list, site_data_UL_list, theta_consensus, site_colors)
    else:
        plot_lc_noUL(site_data_list, theta_consensus, site_colors)
    
    # Print consensus parameter values
    print("\nConsensus parameter values:")
    for i, param in enumerate(params):
        print(f"{param}: {theta_consensus[i]:.4f}")
    
    print(f"\nConsensus MCMC complete. Results saved to {save_folder}")