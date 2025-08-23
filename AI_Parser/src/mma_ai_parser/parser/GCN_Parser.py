import os
import re
import json
from pathlib import Path
from openai import OpenAI
import copy
import pprint
from astropy.coordinates import Angle
import astropy.units as u
from datetime import datetime, timedelta
from copy import deepcopy
import shutil
from datetime import datetime, timezone
import pandas as pd
import numpy as np


class GCNParser():
    """
    AnalyzeResults:
        This class contains functions to:
        - estimate best paramaters values using posterior samples
        - 
    """  
    def __init__(
        self,
        ai_parser_config,
        logger,
        **kwargs
    ):
        
        self.ai_parser_config = ai_parser_config
        self.logger = logger
        self.__dict__.update(kwargs)
        
        # For now hardcode the result path in ini file
        # result_path = self.ai_parser_config.bns_parameter_estimation_configs.dingo_configs.result_dir
        # self.result = Result(file_name=result_path)
        # self.samples = pd.DataFrame(self.result.samples)
        self.summary = {}

    def find_and_copy_radio_gcns(self, communicator, all_gcn_folder, radio_gcn_folder):

        """
        Parameters:
            all_gcn_folder (str): The path to the folder containing the .txt files.
            radio_gcn_folder (str): If not in existance, it createes this folder in the radio output folder. Saves all of the radio GCNs here.

        Returns:
            list of the radio GCN filenames.

        """
        if not os.path.isdir(all_gcn_folder):
            print('allgcn folder not detected')
            self.logger.info('allgcn folder not detected')

            raise FileNotFoundError(f"{all_gcn_folder} is not a valid directory.")
        if not os.path.isdir(radio_gcn_folder):
            os.makedirs(radio_gcn_folder)  # Create output folder if it doesn't exist

        txt_files = [f for f in os.listdir(all_gcn_folder) if f.endswith('.txt')]

        print(f"Found {len(txt_files)} .txt files in {all_gcn_folder}")
        self.logger.info(f"Found {len(txt_files)} .txt files in {all_gcn_folder}")

        count = 0
        for filename in txt_files:
            file_path = os.path.join(all_gcn_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Jy' in content and 'Hz' in content:
                        shutil.copy(file_path, os.path.join(radio_gcn_folder, filename))
                        count += 1
                        gw =  self.ai_parser_config.ai_parser_configs.llm_parameters.gw
                        communicator.send_NewRadioGCN_event(file_path, gw)
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
                self.logger.info(f"Skipping {filename} due to error: {e}")

        print(f"Copied {count} files containing 'Jy' to {radio_gcn_folder}")
        self.logger.info(f"Copied {count} files containing 'Jy' to {radio_gcn_folder}")
        
        return txt_files

    def get_txt_filenames(self, folder_path):
        """
        Returns a list of .txt filenames in a specified folder.

        Parameters:
            folder_path (str): The path to the folder containing the .txt files.

        Returns:
            list of str: A list of .txt filenames in the folder, sorted alphabetically.
        """
        # Ensure the folder exists
        if not os.path.isdir(folder_path):
            raise ValueError(f"The specified folder does not exist: {folder_path}")

        # Get list of .txt filenames sorted alphabetically
        txt_filenames = sorted(
            [filename for filename in os.listdir(folder_path) if filename.endswith(".txt")]
        )

        return txt_filenames
    
    def extract_gps_time(self, text):
        """
        parameters:
            text: The text of a GCN.
            returns: The GPS time (s) of the GCN posting, the string that encompasses the DATE field of the GCN

            Note: This GPS time is only used when the time/date of observation cannot be recovered in the GCN. We default to this and throw a time flag.

        """
        
        # Regular expression to find the DATE line
        match = re.search(r"DATE:\s+(\d{2})/(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2}) GMT", text)

        if not match:
            raise ValueError("Date format not found in text")

        # Extract matched date components
        year, month, day, hour, minute, second = map(int, match.groups())

        # Convert two-digit year to four-digit year (assuming 2000s)
        year += 2000 if year < 80 else 1900

        # Create a datetime object in UTC
        dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)

        # GPS epoch: January 6, 1980, 00:00:00 UTC
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)

        # Calculate GPS time (seconds since GPS epoch)
        gps_seconds = int((dt - gps_epoch).total_seconds()) + 18  # Add leap second offset

        return gps_seconds, match[0]

    def extract_json(self, text):
        matches = re.findall(r'\{.*?\}', text, re.DOTALL)
        json_data = []
        try:
            assert len(matches) > 0
            for match in matches:
                json_data.append(json.loads(match))
            return json_data
        except:
            raise ValueError("No valid JSON found in the text.")

    def generate_json_formatted_response(
        self,
        prompt: str,
        client: OpenAI,
        expected_keys: list,
        model: str = "gpt-4.1",
        max_retry: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ):
        """
        Generates a JSON-formatted response from the OpenAI API.
        
        Args:
            prompt (str): The prompt to send to the OpenAI API.
            client (OpenAI): The OpenAI API client instance.
            expected_keys (list): A list of expected keys in the JSON response.
            model (str): The model to use for generating the response.
            max_retry (int): The maximum number of retries for the API call if it fails to return a valid JSON response.
            temperature (float): The temperature setting for the API call.
            max_tokens (int): The maximum number of tokens to generate in the response.
            
        Returns:
            dict: The JSON-formatted response from the OpenAI API.
        """

        SYSTEM_PROMPT = (
            "You are an expert API for extracting structured radio astronomy data from GCN circulars. "
            "You respond strictly in valid, parsable JSON, with no explanatory text, no Markdown, and no comments. "
            "If information is missing, omit that key."
        )
        json_response = None
        while max_retry > 0:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                            ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                json_response = response.choices[0].message.content
                json_response_copy = copy.deepcopy(json_response)
                # Attempt to parse the JSON response
                response_list_raw = self.extract_json(json_response)
                response_list = []
                # Make all keys lowercase
                valid_list_response = True
                for response in response_list_raw:
                    response = {key.lower(): value for key, value in response.items()}
                    # Check if all expected keys are present
                    valid_response = True
                    for key in expected_keys:
                        if key not in response:
                            valid_response = False
                            break
                    if valid_response:
                        response_list.append(response)
                    else:
                        valid_list_response = False
                        break
                if valid_list_response:
                    return response_list
                else:
                    print("Invalid response keys, retrying...")
                    self.logger("Invalid response keys, retrying...")
                    
                    max_retry -= 1
            except ValueError as e:
                print(json_response_copy)
                print("Invalid JSON response, retrying...")
                self.logger.info("Invalid JSON response, retrying...")

                max_retry -= 1
            except Exception as e:
                print(f"Error generating response: {e}")
                self.logger.info(f"Error generating response: {e}")

                break
        
        return None


    def load_and_process_GCN_rephrase(
        self,
        path_GCN: Path,
        client: OpenAI,
    ):
        print(f"====Starting to process {path_GCN}====")
        self.logger.info(f"====Starting to process {path_GCN}====")

        all_data_points_from_thisGCN = []
        GCN_text = path_GCN.read_text(encoding="utf-8")
        
        # find the GCN number
        for line in GCN_text.splitlines():
            if "NUMBER" in line:
                parts = line.split()
                if len(parts) > 1:
                    GCN_number = parts[1]
                break
            
        gps_seconds, time_string = self.extract_gps_time(GCN_text)
        # print(f"GPS seconds: {gps_seconds}")
        # print(f"Time string: {time_string}")
        
        optical_counterpart_name =  self.ai_parser_config.ai_parser_configs.llm_parameters.optical_counterpart_name
        gw =  self.ai_parser_config.ai_parser_configs.llm_parameters.gw
        host_galaxy =  self.ai_parser_config.ai_parser_configs.llm_parameters.host_galaxy

        prompt1 = """Return ONLY one of the following two JSON objects, with no extra text or explanation.

            {"optical_transient": "true"}
            or
            {"optical_transient": "false"}
            
            Based on the text below, does this GCN circular report a radio follow-up observation of the optical transient associated with """ + gw +""" (also called """ + optical_counterpart_name + """, the optical transient, the """ + gw+ """ optical counterpart, etc.)? The observation should be explicitly targeted at the optical transient, not the host galaxy (""" + host_galaxy + """) or any other object.
            
            Text to analyze:
            ==================================
            """  +GCN_text +  """
            ==================================
            
            Rules:
            - If any part of the circular describes a radio follow-up targeted at """+ optical_counterpart_name + """ or any of its synonyms, respond "true".
            - Otherwise, respond "false".
            - Return strictly valid JSON as shown above. Do NOT include any other text or explanation.
            
            """

        response1 = self.generate_json_formatted_response(
            prompt=prompt1,
            client=client,
            expected_keys=["optical_transient"],
        )[0]
        print(response1)
        
        prompt2 = """Return ONLY one of the following two JSON objects, with no extra text or explanation.

            {"non_emission_statement": "true"}
            or
            {"non_emission_statement": "false"}
            
            Based on the text below, does this GCN circular include any statement claiming there is no significant radio emission detected from the optical transient (""" +optical_counterpart_name+ """ or its synonyms)?
            
            Text to analyze:
            ==================================
                """  + GCN_text +  """
            ==================================
            
            Rules:
            - If the circular explicitly states a non-detection (no emission, non-detection, upper limit only, etc.) for """+ optical_counterpart_name+ """ or its synonyms, respond "true".
            - Otherwise, respond "false".
            - Return strictly valid JSON as shown above. Do NOT include any other text or explanation.
            
            """

        response2 = self.generate_json_formatted_response(
            prompt=prompt2,
            client=client,
            expected_keys=["non_emission_statement"],
        )[0]
        print(response2)
        

        if response1["optical_transient"].lower() == "true":
            
            prompt3 = """Return a JSON array, where each element is a dictionary containing ONLY the following keys (omit any key if information is missing for that observation):

            - "frequency" (in GHz, float or string, e.g. 8.4 GHz)
            - "flux_density" (in mJy, float or string, e.g. 0.24 mJy)
            - "uncertainty" (in mJy, if given, float or string)
            - "type" ("upper_limit" or "detection")
            - "name" (target name as given, only if present)
            - "right_ascension" (if present)
            - "declination" (if present)
            - "year", "month", "day", "hour", "minute", "second" (as integers, if present)
            
            Example output:
            [
            {
                "frequency": 8.4 GHz,
                "flux_density": 0.24 mJy,
                "uncertainty": 0.03 mJy,
                "type": "detection",
                "name": "SSS17a",
                "right_ascension": "13:09:48.08",
                "declination": "-23:22:53.3",
                "year": 2017,
                "month": 8,
                "day": 18,
                "hour": 4,
                "minute": 44,
                "second": 11
            }
            ]
            
            Analyze the following GCN circular and extract data ONLY for radio observations that target """+optical_counterpart_name+""" (the optical transient, the """+gw+""" counterpart, etc.), not """ + host_galaxy + """ or any other source.
            
            Text to analyze:
            ==================================
                    """ +GCN_text+ """
            ==================================
            
            Rules:
            - If there are multiple observations, include a separate dictionary for each one in the JSON array.
            - Do not include any keys if information is not present for that observation.
            - Units: frequency in GHz, flux density and uncertainty in mJy.
            - Only return the JSON array; no extra text or formatting.
            """
            response3 = self.generate_json_formatted_response(
                prompt=prompt3,
                client=client,
                expected_keys=["frequency", "flux_density", "type"],
            )
            if response3 is None:
                print("No valid JSON response from OpenAI.")
                return []
            for json_data in response3:

                if json_data["type"] == "detection":
                    json_data["detected"] = True
                else:
                    json_data["detected"] = False
            
                non_emission_statement = response2["non_emission_statement"].lower() == "true"
                json_data["non_emission_statement"] = non_emission_statement
                # json_data["non_emission_statement"] = response1["non_emission_statement"].lower() == "true"
                
                if json_data["non_emission_statement"] and json_data["detected"]:
                    print("Warning: Non-emission statement found in GCN.")
                    json_data["detected"] = False
                    
                json_data["time"] = str(gps_seconds)
                json_data["GCN_number"] = GCN_number
                json_data["time_string"] = time_string
                
                all_data_points_from_thisGCN.append(json_data)
                
        pprint.pprint(all_data_points_from_thisGCN)
        print(f"====Finished processing {path_GCN}====")
        self.logger.info(f"====Finished processing {path_GCN}====")

        print(" ")
        print(" ")
        print(" ")
        print(" ")
        print(GCN_text)
        return all_data_points_from_thisGCN

    def clean_data(self, data):
        """
        Parameters:
            data (dict)
        output: 
            new_data (dict)

        Applies all flags.
        """
        new_data = deepcopy(data)


        #put frequency in GHz. Frequency is a necesarry field
        freq_str = str(new_data['frequency'])
        new_freq, freq_flag = self.fix_freq(freq_str)
        new_data['frequency'] = new_freq
        new_data['freq_flag'] = freq_flag


        #put flux density in mJy
        if 'uncertainty' in new_data.keys():
            uncertainty_str = str(new_data['uncertainty'])
            new_uncertainty, uncertainty_flag = self.fix_freq(freq_str)
            new_data['uncertainty'] = new_uncertainty
            new_data['uncertainty_flag'] = uncertainty_flag

        #get the datetime and the time flag:
        date_time, time_flag, date_string = self.fix_time(data)
        new_data['datetime'] = date_time
        new_data['time_flag'] = time_flag
        new_data['date_string'] = date_string

        #Put the RA and Dec in a consistent format:
        if 'right_ascension' in new_data.keys():
            RA = self.fix_RA(new_data['right_ascension'])
            new_data['right_ascension'] = RA
            RA = True
        else:
            RA = False
            
        if 'declination' in new_data.keys():
            Dec = self.fix_Dec(new_data['declination'])
            new_data['declination'] = Dec
            Dec = True
        else:
            Dec = False

        #Set the RA Dec flag if either is missing:
        if RA == True and Dec == True:
            new_data['RA_Dec_flag'] = 0
        else:
            new_data['RA_Dec_flag'] = 1

        if 'name' in new_data.keys():
            new_data['name_flag'] = 0
        else:
            new_data['name_flag'] = 1

        #put flux density in mJy
        flux_dens_str = str(new_data['flux_density'])
        new_data['flux_density'], new_data['flux_density_flag'] = self.fix_flux_density(flux_dens_str)

        total_flags = new_data['freq_flag'] + new_data['time_flag'] + new_data['RA_Dec_flag'] + new_data['flux_density_flag'] + new_data['name_flag']

        new_data['total_flags'] = total_flags
        return new_data


    def fix_flux_density(self, flux_dens_str):

        """
        Parameters:
            flux_dens_str (str): the flux density returned by the LLM.
        returns:
            flux density in mJy
            flag (Bool)
        """
        
        FD_flag = 0
        flux_dens_str = str(flux_dens_str).lower().strip()
        #print(flux_dens_str)

        match = re.search(r"([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)", flux_dens_str)  # Find the first number
        if not match:
            FD_flag = 1
            return None, FD_flag  # No valid number found

        num = float(match.group(1))

        if any(unit in flux_dens_str for unit in ["mJy", "milliJy", "milli Jy", "mjy", "millijy", "milli jy"]):
            return num, FD_flag  # Already in mJy
        elif any(unit in flux_dens_str for unit in ["uJy", "microJy", "micro Jy", "muJy", "micro J", "ujy", "microjy", "micro jy", "mujy", "micro j"]):
            return num / 1000, FD_flag  # Convert µJy to mJy
        elif any(word in flux_dens_str for word in ["erg", "mag", "rms", "sigma", "ab"]):
            return None, FD_flag# Invalid or ambiguous unit
        else:
            FD_flag = 1
        

        return num, FD_flag  # Default case

    def fix_freq(self, freq_string):

        """
        Parameters:
            freq_string (str): freq returned by the LLM.
        returns:
            freq. in GHz
            flag (Bool)
        """
        freq_flag = 0
        if isinstance(freq_string, (int, float)):  # If it's already a number, no units
            freq_flag = 1
            if freq_string < 100:  # Assume GHz
                return float(freq_string), freq_flag
            elif freq_string > 1e9:  # Assume it's in Hz
                return freq_string / 1e9, freq_flag
            return float(freq_string), freq_flag  # Otherwise, return as is

        freq_string = str(freq_string).lower().strip()

        parts = freq_string.split()

        if len(parts) == 2 and parts[1] not in {"MHz", "GHz", "mhz", "ghz"}:
            freq_flag = 1
            return None, freq_flag  # If there are two words and the second is not MHz or GHz, return None

        match = re.search(r"([0-9]*\.?[0-9]+)", freq_string)  # Find the first number
        if not match:
            freq_flag = 1
            return None, freq_flag  # No valid number found

        num = float(match.group(1))

        if "GHz" in freq_string or "ghz" in freq_string:
            return num, freq_flag
        elif "MHz" in freq_string or "mhz" in freq_string:
            return num / 1000, freq_flag  # Convert MHz to GHz
        elif num < 100:  # Assume GHz
            freq_flag = 1
            return num, freq_flag
        elif num > 1e9:  # Assume it's in Hz
            freq_flag = 1
            return num / 1e9, freq_flag
        elif num > 1e4 and num <1e6:  # Assume it's in MHz
            freq_flag = 1
            return num / 1e3, freq_flag

        return num, freq_flag  # Default case

    def fix_time(self, dictionary):

        """
        Parameters:
            dictionary (dict): returned by the LLM
        returns:
            datetime
            time_flag (1 if time not reported, else 0)
            date_string (str)
        A function to return the datetime object, and a time_flag = 1 to signal to the user if the date was not included.
        """
        
        time_flag = 0

        try:
            # Try extracting from fields year, month, day (+ optional hour/minute/second)
            year = dictionary['year']
            month = dictionary['month']
            day = dictionary['day']
            hour = dictionary.get('hour', 12)
            minute = dictionary.get('minute', 0)
            second = dictionary.get('second', 0)

            dt = datetime(year, month, day, hour, minute, second)

        except Exception:
            # Fallback to using time_string
            time_flag = 1
            try:
                time_str = dictionary['time_string']
                # Example format: "DATE:    17/08/18 05:07:58 GMT"
                time_str = time_str.replace("DATE:", "").strip().replace("GMT", "").strip()
                dt = datetime.strptime(time_str, "%y/%m/%d %H:%M:%S")
            except Exception as e:
                raise ValueError(f"Could not parse time from time_string: {dictionary.get('time_string', 'N/A')}") from e

        date_string = dt.strftime("%y/%m/%d %H:%M:%S GMT")

        return dt, time_flag, date_string


    def fix_RA(self, RA):
        """
        Converts RA from decimal hours to 'HH:MM:SS.SSS' format,
        or returns the input if already sexagesimal.
        """
        try:
            decimal_RA_hours = float(RA)
            angle = Angle(decimal_RA_hours, unit=u.hourangle)
            return angle.to_string(unit=u.hourangle, sep=':', precision=3, pad=True)
        except ValueError:
            return str(RA).strip()

    def fix_Dec(self, Dec):
        """
        Converts Dec from decimal degrees to '±DD:MM:SS.SSS' format,
        or returns the input if already sexagesimal.
        """
        try:
            decimal_Dec = float(Dec)
            angle = Angle(decimal_Dec, unit=u.deg)
            return angle.to_string(unit=u.deg, sep=':', precision=3, pad=True, alwayssign=True)
        except ValueError:
            return str(Dec).strip()

    def dicts_to_CSV_for_fitting(self, communicator, dict_list, output_csv, GW_dt):
        rows = []
        for d in dict_list:
            # Parse observation time
            try:
                year = int(d.get('year', 0))
                month = int(d.get('month', 1))
                day = int(d.get('day', 1))
                hour = int(d.get('hour', 0))
                minute = int(d.get('minute', 0))
                second = int(d.get('second', 0))
                obs_dt = datetime(year, month, day, hour, minute, second)
            except Exception:
                obs_dt = None
            
            # Compute days since merger
            if obs_dt:
                days = (obs_dt - GW_dt).total_seconds() / 86400.0
            else:
                gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
                seconds_since_gps_epoch = d.get('time')

                obs_dt = gps_epoch + timedelta(seconds=float(seconds_since_gps_epoch))
                days = (obs_dt - GW_dt).total_seconds() / 86400.0

            # Frequency in GHz (float)
            freq = d.get('frequency')
            try:
                freq = float(str(freq).replace('GHz','').strip())
            except:
                freq = np.nan

            # mJy and format for upper limits/detections
            flux = d.get('flux_density', np.nan)
            try:
                flux = float(flux)
            except:
                flux = np.nan
            unc = d.get('uncertainty', None)
            try:
                unc = float(unc)
            except:
                unc = None

            # Figure out what to print in "mJy"
            typ = str(d.get('type', '')).lower()
            non_em = d.get('non_emission_statement', False)
            if typ == "upper_limit" or (isinstance(non_em, bool) and non_em):

                if np.isnan(flux):
                    mJy_entry = ""
                else:
                    mJy_entry = f"<{flux:.2g}"
            elif typ == "detection":
                if unc is not None and not np.isnan(unc) and unc != 0:
                    mJy_entry = f"{flux:.2E}±{unc:.2E}"
                    d['unc_flag'] = 0
                else:
                    mJy_entry = f"{flux:.2E}"
                    d['unc_flag'] = 1
            else:
                mJy_entry = ""


            # Flags
            time_flag = d.get('time_flag', np.nan)
            RA_Dec_flag = d.get('RA_Dec_flag', np.nan)
            name_flag = d.get('name_flag', np.nan)
            freq_flag = d.get('freq_flag', np.nan)
            FD_flag = d.get('flux_density_flag', np.nan)
            GCN_number = d.get('GCN_number', np.nan)
            target = d.get('name', np.nan)
            unc_flag = d.get('unc_flag', np.nan)

            # RA and Dec
            RA = d.get('right_ascension', '')
            Dec = d.get('declination', '')

            row = {
                'days': days,
                'GCN_number': GCN_number,
                'type': typ,
                'target': target,
                'mJy': mJy_entry,
                'uncertainty_flag': unc_flag,
                'GHz': freq,
                'time_flag': time_flag,
                'RA_Dec_flag': RA_Dec_flag,
                'name_flag': name_flag,
                'freq_flag': freq_flag,
                'FD_flag': FD_flag,
                'RA': RA,
                'Dec': Dec,
            }
            rows.append(row)
            gw =  self.ai_parser_config.ai_parser_configs.llm_parameters.gw

            communicator.send_NewFluxTimeDataAdded_event(gw, row, output_csv)
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"Saved table to {output_csv}")
        self.logger.info(f"Saved table to {output_csv}")


        return df


