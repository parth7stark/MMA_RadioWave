import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import afterglowpy as grb


#-------------------------------------------------
# Dictionary for filter names (if needed)
#-------------------------------------------------
band_dict = {
    2.418e17: "1 keV",
    1.555e15: "UVW2",
    1.335e15: "UVM2",
    1.153e15: "UVW1",
    8.652e14: "U",
    8.443e14: "u",
    6.826e14: "B",
    6.389e14: "g",
    5.483e14: "V",
    4.862e14: "r",
    4.008e14: "i",
    3.356e14: "z",
    2.398e14: "J",
    1.851e14: "H",
    1.414e14: "Ks",
    1.000e10: "10 GHz",
    6.000e09: "6 GHz",
}

#-------------------------------------------------
# Data interpretation function (as before)
#-------------------------------------------------
def interpret(data):
    # Correct for time units
    if "days" in data.columns:
        data["t"] = data["days"] * 86400
    elif "seconds" in data.columns:
        data["t"] = data["seconds"]
    elif "t_delta" in data.columns:
        data["t"] = data["t_delta"]
    # Rename filter column if needed
    if "Filter" in data.columns:
        data["filter"] = data["Filter"]
    elif "Band" in data.columns:
        data["filter"] = data["Band"]
    elif "band" in data.columns:
        data["filter"] = data["band"]
    # Frequency (assume Hz or GHz)
    if "GHz" in data.columns:
        data["frequency"] = data["GHz"]
        freq_correct = 1e9
    if "Hz" in data.columns:
        data["frequency"] = data["Hz"]
        freq_correct = 1
    # Flux conversion
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
  
    freq, new_flux, err = [], [], []
    for i in range(data.shape[0]):
        try:
            # Either use a band lookup or take the frequency column:
            freq.append(float(data.iloc[i]["frequency"]) * freq_correct)
        except:
            freq.append("Unknown")
        if "err" in data.columns:
            flux = data.iloc[i]["flux"]
            error = float(data.iloc[i]["err"])
            if "<" in str(flux):
                new_flux.append("UL")
                err.append(0)
            elif ">" in str(flux):
                new_flux.append("UL")
                err.append(0)
            else:
                flux = float(flux)
                new_flux.append(flux)
                err.append(float(error))
        else:
            flux = data.iloc[i]["flux"]
            if "<" in str(flux):
                new_flux.append("UL")
                err.append(0)
            elif ">" in str(flux):
                new_flux.append("UL")
                err.append(0)
            elif "±" in str(flux):
                splt = str(flux).split("±")
                new_flux.append(float(splt[0]))
                err.append(float(splt[1]))
            elif "+-" in str(flux):
                splt = str(flux).split("+-")
                new_flux.append(float(splt[1]))
                err.append(float(splt[1]))
            elif ("+" in str(flux)) or ("-" in str(flux)):
                splt = str(flux).split("+")
                new_flux.append(float(splt[0]))
                err_splt = splt[1].split("-")
                err.append(max([float(splt[0]), float(err_splt[1])]))
            else:
                new_flux.append(float(flux))
                err.append(0)

    for i in range(len(new_flux)):
        if new_flux[i] != "UL":
            if flux_correct != "mag":
                new_flux[i] = new_flux[i] * flux_correct
                err[i] = err[i] * flux_correct
            else:
                temp_flux = 1e3 * 3631 * 10 ** (float(new_flux[i]) / -2.5)
                max_flux = 1e3 * 3631 * 10 ** ((float(new_flux[i]) - err[i]) / -2.5)
                min_flux = 1e3 * 3631 * 10 ** ((float(new_flux[i]) + err[i]) / -2.5)
                new_flux[i] = temp_flux
                err[i] = max([max_flux - temp_flux, temp_flux - min_flux])
    data["frequency"] = freq
    data["flux"] = new_flux
    data["err"] = err

    data = data.loc[(data["flux"] != "UL") & (data["frequency"] != "Unknown")]
    data = data[["t", "frequency", "flux", "err"]].astype(np.float64)
    return data

# --- Function to Plot Global Light Curve Locally ---
def plot_global_light_curve(data, theta, global_min, global_max, frequencies, output_filename):
    
    times = np.geomspace(global_min, global_max, 100)
    fig, ax = plt.subplots()
    E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p = theta
    Z = {
        "jetType": grb.jet.Gaussian,
        "specType": 0,
        "thetaObs": thetaObs,
        "E0": E0,
        "thetaCore": thetaCore,
        "thetaWing": 0.6,
        "n0": n0,
        "p": p,
        "epsilon_e": epsilon_e,
        "epsilon_B": epsilon_B,
        "xi_N": 1.0,
        "d_L": 1.2344e26,
        "z": 0.00897,
    }

    colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))
    for i, freq in enumerate(frequencies):
        nus = np.full(times.shape, freq)
        model = grb.fluxDensity(times, nus, **Z)
        ax.plot(times, model, color=colors[i], label=f"Freq: {freq:.2e} Hz")
    ax.set(xlabel="Time since detection (s)", ylabel="Flux density (mJy)",
           xscale="log", yscale="log")
    ax.legend()
    plt.savefig(output_filename)
    plt.close()
    print(f"Global light curve saved to {output_filename}", flush=True)


    # times = np.geomspace(global_min, global_max, 100)
    # fig, ax = plt.subplots()
    # E0, thetaObs, thetaCore, n0, epsilon_e, epsilon_B, p = theta
    # Z = {
    #     "jetType": grb.jet.Gaussian,
    #     "specType": 0,
    #     "thetaObs": thetaObs,
    #     "E0": E0,
    #     "thetaCore": thetaCore,
    #     "thetaWing": 0.6,
    #     "n0": n0,
    #     "p": p,
    #     "epsilon_e": epsilon_e,
    #     "epsilon_B": epsilon_B,
    #     "xi_N": 1.0,
    #     "d_L": 1.2344e26,
    #     "z": 0.00897,
    # }

    # colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))
    
    # filters = sorted(data["frequency"].unique())

    # for i in range(len(filters)):
    #     to_plot = data.loc[data["frequency"] == filters[i]]
    #     nus = np.full(times.shape, filters[i])
    #     model = grb.fluxDensity(times, nus, **Z)
    #     ax.plot(times, model, color=colors[i], linewidth=0.5)
    #     ax.errorbar(
    #         to_plot["t"],
    #         to_plot["flux"],
    #         yerr=to_plot["err"],
    #         capsize=0,
    #         fmt=".",
    #         label=band_dict[filters[i]],
    #         color=colors[i],
    #     )
    """
    File "/u/parthpatel7173/MMA_RadioWave/FederatedFitting/src/mma_fedfit/generator/inference_utils.py", line 170, in plot_global_light_curve
    label=band_dict[filters[i]],
    KeyError: 670000000.0

    """
    # ax.set(
    #     xlabel="Time since detection (s)",
    #     ylabel="Flux density (mJy)",
    #     xscale="log",
    #     yscale="log",
    # )
    # ax.legend(frameon=True)
    # plt.savefig(output_filename)
    # plt.close()
    # print(f"Global light curve saved to {output_filename}", flush=True)
