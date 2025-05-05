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


def plot_lc_noUL(data, theta):
    data = data.sort_values(by=["frequency"], ascending=False)

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



if __name__ == "__main__":
    
    data_from_csv = pd.read_csv(datafile)
    if data_from_csv.shape[1] == 1:
        data_from_csv = pd.read_csv(datafile, delim_whitespace=True)
    data = interpret(data_from_csv)
    data_UL = interpret_ULs(data_from_csv)

    """
    if args.niter < 100:
        print("A minimum of 100 iterations must be used for the fitting.")
        exit()
    """
    
    t = np.array(list(data["t"]))
    nu = np.array(list(data["frequency"]))
    fnu = np.array(list(data["flux"]))
    err = np.array(list(data["err"]))

    np.random.seed(42)

    nwalkers = 32
    ndim = 8 if z_known else 9

    # define these differently if we need to fit Z also
    if z_known:
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
        
    else:
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

    # position it locally and uniformly
    pos = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(nwalkers, ndim))

    
    #print("Pos: ", pos)
    # pos is log space

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(nu, t, fnu, err),
        moves=[(mvs.StretchMove(a=1.1), 0.7), (mvs.WalkMove(10), 0.3)])

    
    print('doing sampling')
   
    state = sampler.run_mcmc(pos, niter, progress=True)

    gc.collect()

    #Save everything in sight:
    save_sampler_state(sampler, f"{save_folder}/sampler.pkl")
    flat_samples = sampler.get_chain(discard = burnin, flat=True)
    np.save(f"{save_folder}/{run_name}_flat_samples.npy", flat_samples)
    log_prob = sampler.get_log_prob(flat=False)
    np.save(f"{save_folder}/{run_name}_log_prob.npy", log_prob)
    print(f"Saved everything in at {save_folder} after {niter} samples with a burnin = {burnin}")
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

    if z_known:
        params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing']

    else:
        params = ['log(E0)','thetaObs','thetaCore','log(n0)','log(eps_e)','log(eps_B)','p', 'thetaWing', 'z']
    
    results_dict = {}

    print('Median posteriors, 1 sigma$')
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        """
        if i in [0, 3, 4, 5]:
            theta.append(10 ** mcmc[1])
        else:
            theta.append(mcmc[1])
        """
        #keep it in log space
        theta.append(mcmc[1])
        q = np.diff(mcmc)
        print(f'{params[i]} = {mcmc[1]:.2f} +{q[1]:.2f} -{q[0]:.2f}')

        results_dict[params[i]] = {}
        results_dict[params[i]]['median'] = mcmc[1]
        results_dict[params[i]]['LL'] = mcmc[0]
        results_dict[params[i]]['UL'] = mcmc[2]


def make_Log_Likelihood_plot(log_prob):
    
    # Plot log-likelihood for each walker
    plt.figure(figsize=(10, 6))
    for i in range(nwalkers):
        
        plt.plot(log_prob[:, i], alpha=0.3)
    
    plt.xlabel("Steps")
    plt.ylabel("Log Likelihood")
    plt.title("Log Likelihood Progression " + plot_names + " walkers = " +str(nwalkers))
    plt.ylim(-500,120)
    plt.axvline(burnin, label = 'burnin = '+str(burnin), color = 'g', lw = '.5')
    plt.legend()
    plt.savefig(save_folder + '/' + run_name + '_' + 'LogProb.png', bbox_inches = 'tight')

def make_corner_plots(samples):
    
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
    plt.savefig(save_folder + '/' + run_name + '_CornerPlots.png', bbox_inches = 'tight')

def make_posterior_hists(samples):

    medians = np.median(samples[burnin * nwalkers :], axis=0)

    print('median parameter values after burnin:')
    for i in range(ndim):
        print(f"{params[i]}: {medians[i]:.4f}")
    
    # Create subplots
    if ndim == 8:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns
    if ndim == 9:
        fig, axes = plt.subplots(3, 3, figsize=(16, 10))  # 2 rows, 4 columns
        
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
        ax.legend(loc = 4)
    
    plt.tight_layout()
    plt.savefig(save_folder + '/' + run_name + '_PosteriorHists.png', bbox_inches = 'tight')


#Make all of the plots:
make_posterior_hists(flat_samples)

make_corner_plots(flat_samples)

make_Log_Likelihood_plot(log_prob)

#plot the light curve and save it in the folder:
if include_upper_limits_on_lc:
    plot_lc_wUL(data, data_UL, theta)

else:
    plot_lc_noUL(data, theta)








