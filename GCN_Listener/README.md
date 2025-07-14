# Multimessenger App - Gravitational Wave Data

## Overview
This repository is part of a **Multimessenger App** designed to analyze data from different astronomical sources. In multimessenger astronomy, signals from various messengers—such as gravitational waves, radio waves, and electromagnetic waves—are combined to gain a more comprehensive understanding of astrophysical phenomena.

This repo specifically contains files and code related to **GCN Listener** module, forming one of the core components of the overall multimessenger workflow. The app integrates these data streams with other repositories handling gravitational wave analysis to create a unified event detection and analysis framework.

## Installation and running your first inference

You will need a machine running Linux x86 architecture to build container image. If using ARM64 or other architecture, one need to update Apptainer definition file accordingly. Refer [Usage on Delta AI section](##usage-on-delta-ai)

Please follow these steps:

1.  Clone this repository and `cd` into it.

    ```bash
    git clone https://github.com/parth7stark/MMA_RadioWave/tree/main
    cd ./MMA_RadioWave/GCN_Listener
    ```

2. Build the Apptainer image:

    ```bash
    apptainer build MMA_GCN_Listener_miniapp.sif apptainer/MMA_GCN_Listener_miniapp.def
    ```

3. Setup Octopus Connection

    To allow the server and detectors to connect to the Octopus event fabric, you need to export the username (user's OpenID) and password (AWS secret key) in the .bashrc file for all sites where the components will run.

    * **Note:** You must configure the .bashrc file for all sites where the Kafka server, clients, or detectors are running.

    You can append these lines to the end of the .bashrc file using the following command:

    ```bash
    cat <<EOL >> ~/.bashrc
    # Kafka Configuration
    export OCTOPUS_AWS_ACCESS_KEY_ID="AKIA4J4XQFUIIB7Y4FG3"
    export OCTOPUS_AWS_SECRET_ACCESS_KEY="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"
    export OCTOPUS_BOOTSTRAP_SERVERS='b-1-public.diaspora.fy49oq.c9.kafka.us-east-1.amazonaws.com:9198,b-2-public.diaspora.fy49oq.c9.kafka.us-east-1.amazonaws.com:9198'
    EOL   
    ```

    After updating the .bashrc, reload it to apply the changes to your current shell session:
    
    ```bash
    source ~/.bashrc
    ```

4. Start GCN Listener

    The GCN Listener module consists of two continuously running Python processes:

    ### 1. `examples/ocotpus/run_potential_merger_listener.py`
    This script listens for **"PotentialMerger"** events published by the GW module via **Octopus**.

    - When a new gravitational wave merger event is detected, the event metadata is saved to a local JSON file (`GWPotentialMergers.json`).
    - This file acts as a local database of candidate GW events, used later to match with incoming GCN alerts.

    ### 2. `examples/ocotpus/run_gcn_listener.py`

    This script listens for alerts from the GCN Kafka stream, including:

    - LVK Notices: Initial, Update, Retraction, Counterpart
    - GCN Circulars from partner observatories

    When a GCN Initial Notice with BNS probability > 0.5 arrives:

    - The script checks the `GWPotentialMergers.json` file for a matching GW event based on timestamp.

    - If a match is found, the alert is flagged as a super-event of interest.

    The script also:

    - Publishes a "New GCN Circular Added" message to the Octopus event fabric.
    - Triggers downstream pipelines (e.g., GCN circular classification or radio data AI parsing).

   * Update the config file:

        Open the `examples/configs/gcn_listener_config.yaml` configuration file and update the credentials and  following paths with the appropriate ones for your system:

        - **GCN credential:** To stream GCN notices and circulars, you'll need your own Kafka credentials.  
        Follow the official [Start Streaming GCN Notices](https://gcn.nasa.gov/quickstart) quick start guide to create a client ID and secret.

        - **Simulation Data Path:** Specify the location of the dummy events for simulation

        - **Result Path:** Define where the GCN Listener wil save notices and circulars for events of interest.

        - **Logging Output Path:** Define where the logs should be saved.


   * Update the Job script:
      
       The sample job script can be located in the repository under the name `job_scripts/GCN_listener.sh` and `job_scripts/Potential_merger_listener.sh`

        ```bash
        job_scripts/GCN_listener.sh
        job_scripts/Potential_merger_listener.sh

    
        - Modify the SLURM parameters in the script to suit your computing environment (e.g., partition, time, and resources).
        ```

    * Submit the Job Script
    
        Use the following command to submit the job script:
    
        ```bash
        sbatch job_scripts/Potential_merger_listener.sh
        sbatch job_scripts/GCN_listener.sh
        ```

    Submitting the job script will automatically start the Potential Merger Listener and GCN notices and circular listener.


5.  Once the run begins, two output files will be generated for each potential merger listener and gcn listener in your working directory: 
`<job-name>-err-<job_id>.log` (error logs) and `<job-name>-out-<job_id>.log` (output logs). Additionally, the job's output files will be saved in the log output directory specified in your configuration file.

## Usage on Delta AI

Since Delta AI is an **ARM64 architecture machine**, so the Apptainer definition file must be updated to use a base image and install libraries compatible with the ARM64 architecture.

Just use the `MMA_GCN_Listener_miniapp_arm64.def` for building the image.
The remaining steps are the same as those for Delta or Linux x86 architecture.


## Usage on Polaris

Polaris compute nodes do not have network access to the event fabric's port. Therefore, an HTTPS proxy is required. While this will be set up in the future, as a workaround, you can run the server or detector on the **login node**.

Modification to above usage steps:

**Step 1:** Clone repo (no change)

**Step 2:** Download dataset and checkpoints (no change)

**Step 3:** Build the Apptainer image:

To build the Apptainer images, you need to be on the Polaris compute nodes. Follow these steps:

1. **Start an Interactive Session:**
   ```bash
   qsub -I -A <Project> -q debug -l select=1 -l walltime=01:00:00 -l filesystems=home:eagle -l singularity_fakeroot=true
   ```

2. **Set Proxy Environment Variables:**

   ```bash
   export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
   export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
   export http_proxy=http://proxy.alcf.anl.gov:3128
   export https_proxy=http://proxy.alcf.anl.gov:3128

   export BASE_SCRATCH_DIR=/local/scratch/ # For Polaris
   export APPTAINER_TMPDIR=$BASE_SCRATCH_DIR/apptainer-tmpdir
   mkdir -p $APPTAINER_TMPDIR

   export APPTAINER_CACHEDIR=$BASE_SCRATCH_DIR/ apptainer-cachedir
   mkdir -p $APPTAINER_CACHEDIR
   ```

3. **Load Apptainer Module:**

   ```bash
   ml use /soft/modulefiles
   ml load spack-pe-base/0.8.1
   ml load apptainer
   ml load e2fsprogs
   ```

4. **Build Apptainer Images:**
   
    ```bash
    apptainer build --fakeroot MMA_GCN_Listener_miniapp.sif apptainer/MMA_GCN_Listener_miniapp.def
    ```


**Step 4:** Update configurations

Update the configurations for the potential merger and gcn listener as described in the previous steps. Ensure the paths in the config files point to the appropriate locations on your system.

**Step 5:** Start container on login node

To start the server or detectors, execute the following commands from the login node.

    * To start potential merger listener, run:

    ```bash
    apptainer exec --nv \
    MMA_GCN_Listener_miniapp.sif \
    python /app/examples/octopus/run_potential_merger_listener.py --config <absolute path to gcn listener config file>/gcn_listener_config.yaml
    ```

    * To start GCN Notices and Circular Listener, run:

    ```bash
    apptainer exec --nv \
    MMA_GCN_Listener_miniapp.sif \
    python /app/examples/octopus/run_gcn_listener.py --config <absolute path to FL gcn listener config file>/gcn_listener_config.yaml
    ```
    

## Related Projects
This repo focuses on radio wave data. For gravitational wave and joint analysis, please visit [Gravitational Wave Analysis Repo](https://github.com/parth7stark/MMA_GravitationalWave/tree/main) and [GW-RW Joint Analysis Repo](https://github.com/parth7stark/MMA_MultimessengerAnalysis/tree/main). Together, these repositories work within the multimessenger framework to capture and analyze various cosmic events.