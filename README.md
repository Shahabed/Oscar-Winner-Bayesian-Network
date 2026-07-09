# Oscar Winner Prediction using Bayesian Networks

This repository contains an expert-level implementation of a **Bayesian Network** to predict the number of Oscar wins for films based on their performance across different award shows and other relevant features.

## Project Background

The goal of this project is to model the complex relationships between various film attributes and Oscar success using probabilistic graphical models.

In the 2024 Oscars, the model correctly predicted that *Oppenheimer* would be the big winner, forecasting 9 wins while the film ultimately secured 7 awards.

## Repository Structure
oscar-winner-bayesian-network/
├── data/
│   └── oscar_data.csv
├── notebooks/
│   └── oscar_bayesian_network.ipynb
├── src/
│   └── bayesian_network.py
├── README.md
├── requirements.txt
└── LICENSE

## Key Features

- Expert-defined Bayesian Network structure
- Uses `pgmpy` library
- Bayesian Parameter Estimation
- Probabilistic inference
- Reproducible and well-documented code

## Installation

```bash
pip install -r requirements.txt
