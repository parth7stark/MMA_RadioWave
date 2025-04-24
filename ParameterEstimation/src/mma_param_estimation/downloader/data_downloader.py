import os
import numpy as np
import requests
from tqdm import tqdm
from typing import List



class DataDownloader():
    """
    DataDownloader:
        Mimics downloads.py present in dingo-bns repo
        This creates a directory containing data
        This class contains functions for downloading the trained Dingo-BNS model and the measured data (a noise PSD and a GW strain file for each of the three detectors H1,  L1, V1). The model is downloaded from zenodo and the data is downloaded from ligo dcc (strain, PSDs).
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

        self.download_dir = self.estimator_config.bns_parameter_estimation_configs.downloader_configs.download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def download_file(self, url: str, filename: str):
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        full_path = os.path.join(self.download_dir, filename)

        with open(full_path, "wb") as f:
            pbar = tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=int(response.headers.get("Content-Length", 0)),
                desc=f"Downloading {filename}"
            )
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    pbar.update(len(chunk))
                    f.write(chunk)
            pbar.close()
        return full_path

    def download_model(self):
        print("\n### Downloading Dingo model ###")
        url = "https://zenodo.org/records/13321251/files/dingo-bns-model_GW170817.pt?download=1"
        filename = "dingo-bns-model_GW170817.pt"
        return self.download_file(url, filename)

    def download_psds(self):
        print("\n### Downloading PSDs ###")
        url = "https://dcc.ligo.org/public/0158/P1900011/001/GWTC1_GW170817_PSDs.dat"
        raw_psd_filename = "GWTC1_GW170817_PSDs.dat"
        raw_psd_path = self.download_file(url, raw_psd_filename)

        print("Repackaging PSDs...")
        psds = np.loadtxt(raw_psd_path)
        ifos = {1: "H1", 2: "L1", 3: "V1"}
        psd_paths = {}
        for idx, ifo in ifos.items():
            out_file = f"GWTC1_GW170817_PSD_{ifo}.txt"
            out_path = os.path.join(self.download_dir, out_file)
            np.savetxt(out_path, psds[:, [0, idx]])
            psd_paths[ifo] = out_path
        return psd_paths

    def download_frame_files(self, detectors: List[str] = ["H1", "L1", "V1"]):
        print("\n### Downloading GW strain frame files ###")
        frame_paths = {}
        for ifo in detectors:
            frame_file = f"{ifo[0]}-{ifo}_LOSC_CLN_4_V1-1187007040-2048.gwf"
            url = f"https://dcc.ligo.org/public/0146/P1700349/001/{frame_file}"
            path = self.download_file(url, frame_file)
            frame_paths[ifo] = path
        return frame_paths

    def ensure_all(self):
        """Download all required components."""
        model_path = self.download_model()
        psd_paths = self.download_psds()
        frame_paths = self.download_frame_files()
        return {
            "model": model_path,
            "psds": psd_paths,
            "frames": frame_paths
        }