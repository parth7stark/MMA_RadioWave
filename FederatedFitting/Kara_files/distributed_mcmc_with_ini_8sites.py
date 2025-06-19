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

# Create save directory
os.makedirs(save_folder, exist_ok=True)

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

# Modified function to calculate log likelihood for a specific site
########################
# NEW DISTRIBUTED MCMC FUNCTIONS
########################

def remote_log_likelihood(theta, site_data):
    """
    Compute log likelihood for a specific site's data
    In a real distributed system, this would be computed remotely at each site
    """
    if site_data is None or len(site_data) == 0:
        return 0.0
    
    nu = np.array(site_data["frequency"])
    x = np.array(site_data["t"])
    y = np.array(site_data["flux"])
    yerr = np.array(site_data["err"])
    
    if z_known == True and dl_known == True:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing = theta
        z = z_fixed
        DL = dl_fixed 

    elif z_known == True and dl_known == False:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, DL = theta
        z = z_fixed

    elif z_known == False and dl_known == True:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z = theta
        DL = dl_fixed
        
    else:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z, DL = theta

    DL = (DL*u.Mpc).to(u.cm).value #Turn it to cm for the fitting
    

    E0 = 10 ** logE0
    n0 = 10 ** logn0
    epsilon_e = 10 ** logepsilon_e
    epsilon_B = 10 ** logepsilon_B

    Z = {
        "jetType": grb.jet.Gaussian,  # Jet type
        "specType": 0,  # Basic Synchrotron Emission Spectrum
        "thetaObs": thetaObs,
        "E0": E0,
        "thetaCore": thetaCore,
        "thetaWing": thetaWing,
        "n0": n0,
        "p": p,
        "epsilon_e": epsilon_e,
        "epsilon_B": epsilon_B,
        "xi_N": 1.0,  # Fraction of electrons accelerated
        "d_L": DL,  # Luminosity distance in cm
        "z": z,
    }

    try:    
        model = grb.fluxDensity(x, nu, **Z)
        sigma2 = yerr**2
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log10(sigma2))
    except:
        #Just put in different parameters to make it run.
        print('enters except block')
        # Fallback parameters to make it run
        Z = {
            "jetType": grb.jet.Gaussian,
            "specType": 0,
            "thetaObs": math.pi/4.0,
            "E0": 1e52,
            "thetaCore": math.pi/10.0,
            "thetaWing": math.pi/3.0,
            "n0": 5e-3,
            "p": 2.5,
            "epsilon_e": .1,
            "epsilon_B": .01,
            "xi_N": 1.0,
            "d_L": 1.2344e26,
            "z": 0.00897,
        }
        model = grb.fluxDensity(x, nu, **Z)
        sigma2 = yerr**2
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log10(sigma2))

def global_log_probability(theta, sites_data):
    """
    Global log probability function that coordinates with all sites
    Aggregates log-likelihood from all sites and adds prior
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    total_ll = 0.0
    for site_data in sites_data:
        # In a real distributed setting, the coordinator would send theta to the site
        # and receive the local log-likelihood. Here we simulate by direct function call.
        site_ll = remote_log_likelihood(theta, site_data)
        total_ll += site_ll
    
    return lp + total_ll

def log_prior(theta):
    # print("prior theta: ", theta)
    # exit()
    # theta is in log space
    # rename below variables accordingly
    # E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p, thetaWing = theta
    if z_known == True and dl_known == True:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing = theta

    elif z_known == True and dl_known == False:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, DL = theta

    elif z_known == False and dl_known == True:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z = theta
        
    else:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z, DL = theta
    

    if z_known == True and dl_known == True:
        if (
            loge0_range[0] <= logE0 <= loge0_range[1]
            and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
            and thetawing_range[0] <= thetaWing < thetawing_range[1]
            and thetacore_range[0] <= thetaCore < thetacore_range[1]
            and p_range[0] < p < p_range[1]
            and logepsilon_e_range[0] < logepsilon_e <= logepsilon_e_range[1]
            and logepsilon_b_range[0] < logepsilon_B <= logepsilon_b_range[1]
            and logn0_range[0] < logn0 < logn0_range[1]
        ):
            return 0.0
        return -np.inf

    elif z_known == True and dl_known == False:
        if (
            loge0_range[0] <= logE0 <= loge0_range[1]
            and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
            and thetawing_range[0] <= thetaWing < thetawing_range[1]
            and thetacore_range[0] <= thetaCore < thetacore_range[1]
            and p_range[0] < p < p_range[1]
            and logepsilon_e_range[0] < logepsilon_e <= logepsilon_e_range[1]
            and logepsilon_b_range[0] < logepsilon_B <= logepsilon_b_range[1]
            and logn0_range[0] < logn0 < logn0_range[1]
            and dl_range[0] < DL < dl_range[1]
        ):
            return 0.0
        return -np.inf

    elif z_known == False and dl_known == True:
        if (
            loge0_range[0] <= logE0 <= loge0_range[1]
            and thetaobs_range[0] <= thetaObs < thetaobs_range[1]
            and thetawing_range[0] <= thetaWing < thetawing_range[1]
            and thetacore_range[0] <= thetaCore < thetacore_range[1]
            and p_range[0] < p < p_range[1]
            and logepsilon_e_range[0] < logepsilon_e <= logepsilon_e_range[1]
            and logepsilon_b_range[0] < logepsilon_B <= logepsilon_b_range[1]
            and logn0_range[0] < logn0 < logn0_range[1]
            and z_range[0] < z < z_range[1]
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
            and logepsilon_b_range[0] < logepsilon_B <= logepsilon_b_range[1]
            and logn0_range[0] < logn0 < logn0_range[1]
            and z_range[0] < z < z_range[1]
            and dl_range[0] < DL < dl_range[1]
        ):
            return 0.0
        return -np.inf

def run_distributed_mcmc(sites_data, nwalkers, ndim, niter, burnin, lower_bounds, upper_bounds):
    """
    Run MCMC in a distributed fashion with a central coordinator
    """
    print("Running distributed MCMC with a central coordinator")
    
    # Initialize walkers with uniform random positions
    pos = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(nwalkers, ndim))
    
    with Pool() as pool:
        # Create the sampler that will coordinate with all sites
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, global_log_probability, args=(sites_data,), pool=pool,
            moves=[(mvs.StretchMove(a=1.1), 0.7), (mvs.WalkMove(10), 0.3)])
        
        print('Central coordinator: starting sampling')
        state = sampler.run_mcmc(pos, niter, progress=True)
    
    gc.collect()
    return sampler

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
# def generate_site_data(num_sites=3):
#     """
#     Split the data into multiple sites for distributed processing
#     """
#     # Load and process the CSV data
#     data_from_csv = pd.read_csv(datafile)
#     if data_from_csv.shape[1] == 1:
#         data_from_csv = pd.read_csv(datafile, delim_whitespace=True)
    
#     # Process the data
#     data = interpret(data_from_csv)
#     data_UL = interpret_ULs(data_from_csv)
    
#     # Split the data by site
#     total_rows = len(data)
#     site_indices = np.random.randint(0, num_sites, size=total_rows)
    
#     sites_data = []
#     sites_data_UL = []
    
#     for site_id in range(num_sites):
#         # Get data for this site
#         site_mask = site_indices == site_id
#         site_data = data[site_mask].copy()
#         site_data['site_id'] = site_id
#         sites_data.append(site_data)
        
#         # Get upper limit data for this site if it exists
#         if len(data_UL) > 0:
#             # For UL data, we'll use the same random assignment to sites
#             # This is a simplification - in reality, UL data might be differently distributed
#             site_UL_data = data_UL[site_indices == site_id].copy() if site_id < len(data_UL) else pd.DataFrame()
#             if not site_UL_data.empty:
#                 site_UL_data['site_id'] = site_id
#             sites_data_UL.append(site_UL_data)
    
#     return sites_data, sites_data_UL

def plot_distributed_data_lc(sites_data, sites_data_UL, theta, include_upper_limits=True):
    """
    Plot light curve with data points colored by their site
    """
    # Extract parameters
    # Extract and transform model parameters
    if z_known == True and dl_known == True:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing = theta
        z = z_fixed
        DL = dl_fixed

    elif z_known == True and dl_known == False:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, DL = theta
        z = z_fixed

    elif z_known == False and dl_known == True:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z = theta
        DL = dl_fixed
        
    else:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z, DL = theta

    DL = (DL * u.Mpc).to(u.cm).value   
    E0 = 10 ** logE0
    n0 = 10 ** logn0
    epsilon_e = 10 ** logepsilon_e
    epsilon_B = 10 ** logepsilon_B

    
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
    
    # Combine all data to find min/max times
    all_data = pd.concat(sites_data)
    min_time = min(all_data["t"])
    max_time = max(all_data["t"])
    
    times = np.geomspace(min_time, max_time, 100)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors and markers for sites and bands
    # site_colors = ['red', 'blue', 'green']
    # site_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
    # site_markers = ['o', 's', '^', 'D', 'P']
    band_colors = ['black', 'darkblue', 'blue', 'green', 'orange', 'red', 'darkred']
    
    site_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan']
    site_markers = ['o', 's', '^', 'D', 'P', 'X', '*', '<']
    
    # Create legend handles
    legend_handles = []
    
    # Plot model curves for each radio band
    for i, (band, (fmin, fmax, fcen, _)) in enumerate(radio_bands.items()):
        # Check if any site has data in this band
        band_data_exists = False
        for site_data in sites_data:
            if not site_data[(site_data["frequency"] >= fmin) & 
                           (site_data["frequency"] < fmax)].empty:
                band_data_exists = True
                break
        
        if band_data_exists:
            color = band_colors[i % len(band_colors)]
            model = grb.fluxDensity(times, np.full(times.shape, fcen), **Z)
            ax.plot(times / (24 * 60.0 * 60), model, color=color, linewidth=1.5, 
                   label=f"{band} ({fcen/1e9:.1f} GHz)")
            
            # Add band to legend
            legend_handles.append(Line2D([0], [0], color=color, linewidth=1.5, 
                                       label=f"{band} ({fcen/1e9:.1f} GHz)"))
    
    # Plot data points from each site
    for site_id, site_data in enumerate(sites_data):
        site_color = site_colors[site_id % len(site_colors)]
        site_marker = site_markers[site_id % len(site_markers)]
        
        # Plot data for each band
        for i, (band, (fmin, fmax, fcen, _)) in enumerate(radio_bands.items()):
            band_data = site_data[(site_data["frequency"] >= fmin) & 
                                (site_data["frequency"] < fmax)]
            
            if not band_data.empty:
                band_color = band_colors[i % len(band_colors)]
                
                # Plot error bars
                band_t = np.array(band_data["t"]) / (24 * 60.0 * 60)
                band_fnu = np.array(band_data["flux"])
                band_err = np.array(band_data["err"])
                
                ax.errorbar(
                    band_t, band_fnu, yerr=band_err, 
                    fmt='none', ecolor=band_color, elinewidth=0.8, 
                    capsize=3, alpha=0.7
                )
                
                # Plot actual data points with site-specific marker
                ax.scatter(
                    band_t, band_fnu, 
                    marker=site_marker, s=50, 
                    color=band_color, edgecolor='black', 
                    linewidth=0.5, alpha=0.9
                )
        
        # Add site to legend
        legend_handles.append(Line2D([0], [0], marker=site_marker, color='gray',
                                    markerfacecolor='gray', markeredgecolor='black',
                                    linestyle='None', markersize=8,
                                    label=f"Site {site_id+1}"))
    
    # Plot upper limits if requested
    if include_upper_limits and sites_data_UL:
        for site_id, site_data_UL in enumerate(sites_data_UL):
            if site_data_UL.empty:
                continue
                
            # Plot UL for each band
            for i, (band, (fmin, fmax, fcen, _)) in enumerate(radio_bands.items()):
                band_data = site_data_UL[(site_data_UL["frequency"] >= fmin) & 
                                      (site_data_UL["frequency"] < fmax)]
                
                if not band_data.empty:
                    band_color = band_colors[i % len(band_colors)]
                    
                    band_t = np.array(band_data["t"]) / (24 * 60.0 * 60)
                    band_fnu = np.array(band_data["flux"])
                    
                    # Plot upper limits with downward arrows
                    ax.scatter(
                        band_t, band_fnu, 
                        marker='v', s=60, 
                        facecolors='none', 
                        edgecolors=band_color, 
                        linewidth=1.0, alpha=0.8
                    )
        
        # Add upper limit to legend
        legend_handles.append(Line2D([0], [0], marker='v', color='none',
                                   markerfacecolor='none', markeredgecolor='black',
                                   linestyle='None', markersize=8,
                                   label="Upper Limit"))
    
    # Set axis properties
    ax.set_xlabel("Time since detection (days)", fontsize=14)
    ax.set_ylabel("Flux density (mJy)", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{plot_names} - Distributed MCMC Fit", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # Create two-row legend
    by_label = dict(zip([h.get_label() for h in legend_handles], legend_handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='upper center', bbox_to_anchor=(0.5, -0.15),
             ncol=len(by_label)//2 + 1, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"{run_name}_distributed_lightcurve.png"), 
               bbox_inches='tight', dpi=300)
    plt.close()



def plot_lc_noUL(data, theta):
    data = data.sort_values(by=["frequency"], ascending=False)

    # Extract and transform model parameters
       # Extract and transform model parameters
    if z_known == True and dl_known == True:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing = theta
        z = z_fixed
        DL = dl_fixed

    elif z_known == True and dl_known == False:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, DL = theta
        z = z_fixed

    elif z_known == False and dl_known == True:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z = theta
        DL = dl_fixed
        
    else:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z, DL = theta

    DL = (DL * u.Mpc).to(u.cm).value   
    E0 = 10 ** logE0
    n0 = 10 ** logn0
    epsilon_e = 10 ** logepsilon_e
    epsilon_B = 10 ** logepsilon_B

    # DL = cosmo.luminosity_distance(z).to(u.cm)
    # DL = DL.value
    DL = (DL * u.Mpc).to(u.cm).value   


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

    times = np.geomspace(min(data["t"]), max(data["t"]), 100)
    fig, ax = plt.subplots()

    # Extract and convert time to days
    t = np.array(data["t"]) / (24 * 60.0 * 60)
    nu = np.array(data["frequency"])
    fnu = np.array(data["flux"])
    err = np.array(data["err"])

    band_colors_list = ['black', 'red', 'blue', 'green', 'magenta']
    legend_handles = []

    for i, (band, (fmin, fmax, fcen, marker)) in enumerate(radio_bands.items()):
        band_data = data[(data["frequency"] >= fmin) & (data["frequency"] < fmax)]
        if not band_data.empty:
            color = band_colors_list[i % len(band_colors_list)]

            # Plot model curve
            model = grb.fluxDensity(times, np.full(times.shape, fcen), **Z)
            ax.plot(times / (24 * 60.0 * 60), model, color=color, linewidth=1.2)

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

            # Overlay scatter points
            ax.scatter(
                band_t,
                band_fnu,
                marker=marker,
                color=color,
                edgecolor='black',
                linewidth=0.5,
                s=40,
                alpha=0.9
            )

            # Create custom legend handle (symbol + freq)
            legend_label = f"{band} ({fcen/1e9:.1f} GHz)"
            legend_handles.append(Line2D(
                [0], [0],
                marker=marker,
                color=color,
                linestyle='-',  # for model
                markerfacecolor=color,
                markeredgecolor='black',
                linewidth=1.2,
                markersize=7,
                label=legend_label
            ))

    ax.set(
        xlabel="Time since detection (days)",
        ylabel="Flux density (mJy)",
        xscale="log",
        yscale="log",
        title=plot_names
    )

    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(save_folder + '/' + run_name + '_' + 'lightcurve.png', bbox_inches = 'tight')

    
def plot_lc_wUL(data, data_UL, theta):
    data = data.sort_values(by=["frequency"], ascending=False)
    data_UL = data_UL.sort_values(by=["frequency"], ascending=False)

    # Extract and transform model parameters
    if z_known == True and dl_known == True:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing = theta
        z = z_fixed
        DL = dl_fixed

    elif z_known == True and dl_known == False:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, DL = theta
        z = z_fixed

    elif z_known == False and dl_known == True:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z = theta
        DL = dl_fixed
        
    else:
        logE0, thetaObs, thetaCore, logn0, logepsilon_e, logepsilon_B, p, thetaWing, z, DL = theta


    DL = (DL * u.Mpc).to(u.cm).value   
    E0 = 10 ** logE0
    n0 = 10 ** logn0
    epsilon_e = 10 ** logepsilon_e
    epsilon_B = 10 ** logepsilon_B

    # DL = cosmo.luminosity_distance(z).to(u.cm)
    # DL = DL.value

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

    times = np.geomspace(min(data["t"]), max(data["t"]), 100)
    fig, ax = plt.subplots()

    # Extract and convert time to days
    t = np.array(data["t"]) / (24 * 60.0 * 60)
    nu = np.array(data["frequency"])
    fnu = np.array(data["flux"])
    err = np.array(data["err"])

    band_colors_list = ['black', 'red', 'blue', 'green', 'magenta']
    legend_handles = []

    for i, (band, (fmin, fmax, fcen, marker)) in enumerate(radio_bands.items()):
        band_data = data[(data["frequency"] >= fmin) & (data["frequency"] < fmax)]
        if not band_data.empty:
            color = band_colors_list[i % len(band_colors_list)]

            # Plot model curve
            model = grb.fluxDensity(times, np.full(times.shape, fcen), **Z)
            ax.plot(times / (24 * 60.0 * 60), model, color=color, linewidth=1.2)

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

            # Overlay scatter points
            ax.scatter(
                band_t,
                band_fnu,
                marker=marker,
                color=color,
                edgecolor='black',
                linewidth=0.5,
                s=40,
                alpha=0.9
            )

            # Create custom legend handle (symbol + freq)
            legend_label = f"{band} ({fcen/1e9:.1f} GHz)"
            legend_handles.append(Line2D(
                [0], [0],
                marker=marker,
                color=color,
                linestyle='-',  # for model
                markerfacecolor=color,
                markeredgecolor='black',
                linewidth=1.2,
                markersize=7,
                label=legend_label
            ))
            
    #plot the upper limits as triangles:
    for i, (band, (fmin, fmax, fcen, marker)) in enumerate(radio_bands.items()):
        band_data = data_UL[(data_UL["frequency"] >= fmin) & (data_UL["frequency"] < fmax)]
        if not band_data.empty:
            color = band_colors_list[i % len(band_colors_list)]

            # Plot error bars
            band_t = np.array(band_data["t"]) / (24 * 60.0 * 60)
            band_fnu = np.array(band_data["flux"])

            # Overlay scatter points
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

    # Create custom legend handle (symbol + freq)
    legend_label = "upper limit"
    legend_handles.append(Line2D(
        [0], [0],
        marker='v',
        color='black',
        markerfacecolor='none',
        markeredgecolor='black',
        linewidth=1.2,
        linestyle='None',
        markersize=7,
        label=legend_label
    ))

    ax.set(
        xlabel="Time since detection (days)",
        ylabel="Flux density (mJy)",
        xscale="log",
        yscale="log",
        title=plot_names
    )

    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(save_folder + '/' + run_name + '_' + 'lightcurve.png', bbox_inches = 'tight')

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
    # Create subplots
    if ndim == 8:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns
    if ndim == 9:
        fig, axes = plt.subplots(3, 3, figsize=(16, 10))  # 3 rows, 3 columns
    if ndim == 10:
        fig, axes = plt.subplots(2, 5, figsize=(16, 10))  # 3 rows, 3 columns
        
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


if __name__ == "__main__":
    
    # Parse parameters and load data
    # data_from_csv = pd.read_csv(datafile)
    # if data_from_csv.shape[1] == 1:
    #     data_from_csv = pd.read_csv(datafile, delim_whitespace=True)
    
    # # Create distributed directory structure
    os.makedirs(save_folder, exist_ok=True)
    
    # # Main processing pipeline
    # data = interpret(data_from_csv)
    # data_UL = interpret_ULs(data_from_csv)

    sites_data = []
    sites_data_UL = []

    num_sites = 8  # Number of sites to distribute data

    for i in range(num_sites):
        site_df = pd.read_csv(params_from_ini['files'][f'datafile_site{i}'])
        if site_df.shape[1] == 1:
            site_df = pd.read_csv(params_from_ini['files'][f'datafile_site{i}'], delim_whitespace=True)

        site_data = interpret(site_df)
        site_data['site_id'] = i
        sites_data.append(site_data)

        site_UL = interpret_ULs(site_df)
        site_UL['site_id'] = i
        sites_data_UL.append(site_UL)
    
    # # Split data into sites
    # sites_data = split_data_by_site(data, num_sites)
    # sites_data_UL = split_data_by_site(data_UL, num_sites)

    """
    if args.niter < 100:
        print("A minimum of 100 iterations must be used for the fitting.")
        exit()
    """
    

    # Simulate distributing data across 3 sites
    # In a real implementation, this would be read from separate files or databases
    np.random.seed(42)

    # Partition data for 3 sites (for simulation)
    # In reality, data would already be distributed across sites
    # Generate site data
    # num_sites = 3
    # sites_data, sites_data_UL = generate_site_data(num_sites)
    
    # Print data distribution info
    print(f"\nData distribution across {num_sites} sites:")
    for site_id, site_data in enumerate(sites_data):
        print(f"Site {site_id+1}: {len(site_data)} data points")

    nwalkers = 32
    # ndim = 8 if z_known else 9

    # define these differently if we need to fit Z also
    nwalkers = 32
    if z_known == True and dl_known == True:
        ndim = 8
        lower_bounds = np.array([
            loge0_range[0],        # log10(E0)
            thetaobs_range[0],       # thetaObs
            thetacore_range[0],      # thetaCore
            logn0_range[0],        # log10(n0)
            logepsilon_e_range[0],        # log10(eps_e)
            logepsilon_b_range[0],        # log10(eps_B)
            p_range[0],       # p
            thetawing_range[0]        # thetaWing
        ])
        
        upper_bounds = np.array([
            loge0_range[1],        # log10(E0)
            thetaobs_range[1],       # thetaObs
            thetacore_range[1],      # thetaCore
            logn0_range[1],        # log10(n0)
            logepsilon_e_range[1],        # log10(eps_e)
            logepsilon_b_range[1],        # log10(eps_B)
            p_range[1],       # p
            thetawing_range[1]        # thetaWing
        ])

    elif z_known == True and dl_known == False:
        ndim = 9
        lower_bounds = np.array([
            loge0_range[0],        # log10(E0)
            thetaobs_range[0],       # thetaObs
            thetacore_range[0],      # thetaCore
            logn0_range[0],        # log10(n0)
            logepsilon_e_range[0],        # log10(eps_e)
            logepsilon_b_range[0],        # log10(eps_B)
            p_range[0],       # p
            thetawing_range[0],        # thetaWing
            dl_range[0]
        ])
        
        upper_bounds = np.array([
            loge0_range[1],        # log10(E0)
            thetaobs_range[1],       # thetaObs
            thetacore_range[1],      # thetaCore
            logn0_range[1],        # log10(n0)
            logepsilon_e_range[1],        # log10(eps_e)
            logepsilon_b_range[1],        # log10(eps_B)
            p_range[1],       # p
            thetawing_range[1],        # thetaWing
            dl_range[1]
        ])

    elif z_known == False and dl_known == True:
        ndim = 9
        lower_bounds = np.array([
            loge0_range[0],        # log10(E0)
            thetaobs_range[0],       # thetaObs
            thetacore_range[0],      # thetaCore
            logn0_range[0],        # log10(n0)
            logepsilon_e_range[0],        # log10(eps_e)
            logepsilon_b_range[0],        # log10(eps_B)
            p_range[0],       # p
            thetawing_range[0],        # thetaWing
            z_range[0]
        ])
        
        upper_bounds = np.array([
            loge0_range[1],        # log10(E0)
            thetaobs_range[1],       # thetaObs
            thetacore_range[1],      # thetaCore
            logn0_range[1],        # log10(n0)
            logepsilon_e_range[1],        # log10(eps_e)
            logepsilon_b_range[1],        # log10(eps_B)
            p_range[1],       # p
            thetawing_range[1],        # thetaWing
            z_range[1]
        ])

    else:
        ndim = 10
        lower_bounds = np.array([
            loge0_range[0],        # log10(E0)
            thetaobs_range[0],       # thetaObs
            thetacore_range[0],      # thetaCore
            logn0_range[0],        # log10(n0)
            logepsilon_e_range[0],        # log10(eps_e)
            logepsilon_b_range[0],        # log10(eps_B)
            p_range[0],       # p
            thetawing_range[0],        # thetaWing
            z_range[0],
            dl_range[0]
        ])
        
        upper_bounds = np.array([
            loge0_range[1],        # log10(E0)
            thetaobs_range[1],       # thetaObs
            thetacore_range[1],      # thetaCore
            logn0_range[1],        # log10(n0)
            logepsilon_e_range[1],        # log10(eps_e)
            logepsilon_b_range[1],        # log10(eps_B)
            p_range[1],       # p
            thetawing_range[1],        # thetaWing
            z_range[1],
            dl_range[1] 
        ])


    # position it locally and uniformly
    pos = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(nwalkers, ndim))

    # Create the distributed MCMC sampler
    # Run distributed MCMC
    sampler = run_distributed_mcmc(
        sites_data, nwalkers, ndim, niter, burnin, 
        lower_bounds, upper_bounds
    )
    
    # Save sampler state
    save_sampler_state(sampler, os.path.join(save_folder, f"{run_name}_distributed_sampler.pkl"))
    
    # Extract samples
    flat_samples = sampler.get_chain(discard=burnin, flat=True)
    np.save(os.path.join(save_folder, f"{run_name}_distributed_flat_samples.npy"), flat_samples)
    
    # Save log probability
    log_prob = sampler.get_log_prob(flat=False)
    np.save(os.path.join(save_folder, f"{run_name}_distributed_log_prob.npy"), log_prob)
    

    #Save everything in sight:
    # Save results
    print(f"Saved results in {save_folder} after {niter} samples with burnin = {burnin}")
    # Update pos for next iteration

    """
    Everything beyond this point can be done by the user at a later time by loading up the sampler, log_prob, samples, etc.

    But we print all of the information and make a first-pass at plotting the outputs. The user can then check for over-fitting and adjust the analysis accordingly.
    """

    try:
        tau = sampler.get_autocorr_time(tol=0)
        print("Autocorrelation time:", tau)
    except emcee.autocorr.AutocorrError:
        print("Chain is too short to estimate autocorrelation time!")

    print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

    theta = []

    if z_known == True and dl_known == True:
        params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing']

    elif z_known == False and dl_known == True:
        params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing', 'z']

    elif z_known == True and dl_known == False:
        params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing', 'DL']

    else:
        params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing', 'z', 'DL']

    # Create standard plots
    make_posterior_hists(
        flat_samples, burnin, nwalkers, ndim, params, 
        os.path.join(save_folder, f"{run_name}_distributed_PosteriorHists.png")
    )
    
    make_corner_plots(
        flat_samples, burnin, nwalkers, ndim, params, 
        true_values, display_truths_on_corner,
        os.path.join(save_folder, f"{run_name}_distributed_CornerPlots.png")
    )
    
    make_Log_Likelihood_plot(
        log_prob, burnin, nwalkers, 
        os.path.join(save_folder, f"{run_name}_distributed_LogProb.png")
    )
    
    # Get median parameter values
    theta_distributed = np.median(flat_samples, axis=0)
    
    # Plot distributed data light curve
    plot_distributed_data_lc(
        sites_data, sites_data_UL, theta_distributed, include_upper_limits_on_lc
    )
    
    # Print final parameter values
    print("\nDistributed MCMC parameter values:")
    for i, param in enumerate(params):
        print(f"{param}: {theta_distributed[i]:.4f}")
    
    print(f"\nDistributed MCMC complete. Results saved to {save_folder}")