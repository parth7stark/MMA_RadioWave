# Multimessenger App - Radio Wave Data

## Overview
This repository is part of a **Multimessenger App** designed to analyze data from different astronomical sources. In multimessenger astronomy, signals from various messengers—such as gravitational waves, radio waves, and electromagnetic waves—are combined to gain a more comprehensive understanding of astrophysical phenomena.

In multi-messenger astronomy, signals from different astrophysical messengers (e.g., gravitational waves and radio waves) provide complementary insights into the same cosmic events. 

This module focuses on automating the end-to-end processing of **radio follow-up observations** for **gravitational wave (GW)** merger candidates, particularly binary neutron star (BNS) events.

The module performs:

- Real-time ingestion and classification of GCN notices and circulars

- AI-driven parsing of flux-time information from GCN circulars.

- Privacy-enhancing, distributed fitting of radio light curves

These capabilities enable rapid and scalable integration of electromagnetic follow-ups with GW detections, facilitating timely and precise parameter estimation.


## Repository Structure
```
MMA_RadioWave/
├── GCNListener/       # Listens to GCN alerts and classifies circulars
├── AI_Parser/         # Extracts flux/time from radio circulars
├── FederatedFitting/  # Performs distributed light curve modeling
└── README.md
```

---

## Module Components

### 1. GCN Listener: GCN Alert Processing

- Subscribes to **GCN alerts** from observatories and MMA_GravitationalWave module **GW detection triggers**.
- Detects potential matches between GCN alerts and GW merger events based on timestamp and metadata.
- For alerts with **BNS probability > 50%**, checks if it corresponds to an event of interest.
- Publishes “New GCN circular added” messages via the **Octopus event fabric**.
- A **GCN classifier** module categorizes the circular type (radio, optical, gamma-ray, etc.).

---

### 2. AI Parser: Radio Astronomy Data Handling

- For circulars classified as **radio**, applies a domain-specific **AI parser**.
- Extracts key observational parameters such as **flux density**, **frequency**, and **observation time**.
- Outputs structured metadata and saves it to a **distributed datastore**.
- Each parsed circular is linked to its associated **GW event ID** for cross-messenger correlation.

---

### 3. Federated Fitting: Distributed Light Curve Modeling

- Implements **federated MCMC** to fit radio afterglow light curves across distributed observation sites.
- **Data never leaves the site** — only posterior samples and log-likelihoods are shared.
- Supports **progressive fitting**, where model updates as new observations arrive.
- Produces final model parameters and **credible intervals**, which are published back to the event fabric.

**Federated MCMC Architecture:**

- A central server proposes parameters (`θ`) at each iteration.
- These are broadcast to data sites, which compute **partial log-likelihoods** using local data.
- The server aggregates the total posterior:  
  `posterior ∝ prior + ∑ log-likelihoods`
- Accept/reject decisions guide parameter updates.

---

## Purpose and Impact

The Radio module enables **low-latency**, **privacy-enhancing**, and **scalable** radio follow-up for gravitational wave events. It:

- Reduces the time to generate radio constraints on jet structure and energetics.
- Maintains **data locality** while still allowing joint inference across institutions.
- Forms a critical link in the **joint multimessenger analysis** pipeline, feeding posterior distributions into the overlap module for combined cosmological inference.

---

## Getting Started

Each subdirectory (`GCNListener/`, `AI_Parser/`, `FederatedFitting/`) contains modular scripts and configuration templates.

To run a typical radio follow-up analysis:

1. Start the **GCN listener** to monitor alerts and classify circulars.
2. When a **radio circular** is detected, use the AI parser to extract observations.
3. Run **federated fitting** across participating sites to model the light curve.
4. Feed the posterior samples into the **OverlapAnalysis/** module to combine with GW posteriors.


## Related Projects
This repo focuses on radio wave data. For gravitational wave and joint analysis, please visit [Gravitational Wave Analysis Repo](https://github.com/parth7stark/MMA_GravitationalWave/tree/main) and [GW-RW Joint Analysis Repo](https://github.com/parth7stark/MMA_MultimessengerAnalysis/tree/main). Together, these repositories work within the multimessenger framework to capture and analyze various cosmic events.