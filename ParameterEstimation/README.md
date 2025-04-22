# Multimessenger App - Rarameter Estimation

## Overview
This repository is part of a **Multimessenger App** designed to analyze data from different astronomical sources. In multimessenger astronomy, signals from various messengers—such as gravitational waves, radio waves, and electromagnetic waves—are combined to gain a more comprehensive understanding of astrophysical phenomena.

This repository contains demos for running the machine learning framework Dingo-BNS, which performs fast and accurate inference of gravitational waves from binary neutron stars.

This repo specifically contains files and code related to **Parameter Estimation** analysis using Dingo-BNS, a machine learning framework Dingo-BNS which performs fast and accurate inference of gravitational waves from binary neutron stars. It forms one of the core components of the overall multimessenger workflow. The app integrates these data streams with other repositories handling gravitational wave analysis to create a unified event detection and analysis framework.

## Todo List and Project Plan
Please refer to our Box folder for the latest project tasks and roadmap: [Link](https://anl.app.box.com/s/q11vyc14fmjz3gfefyk4o7plsgsr4wnp)

## Related Projects
This repo focuses on radio wave data. For gravitational wave analysis, please visit [Gravitational Wave Analysis Repo](https://github.com/parth7stark/GW_VerticalFL/tree/main). Together, these repositories work within the multimessenger framework to capture and analyze various cosmic events.

## Future Plans
- Integration of additional messenger types (e.g., neutrinos, gamma rays)
- Real-time data streaming and event detection
- Cross-correlation between different datasets for enhanced analysis

## Installation

```
conda create -n mma_radiowave python=3.10 --y
conda activate mma_radiowave
pip install -r requirements.txt
```