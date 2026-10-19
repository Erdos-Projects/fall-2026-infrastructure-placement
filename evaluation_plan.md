# Evaluation Plan - Areas of High-Risk Nitrate Pollution in Salinas Valley 

## Unit of Analysis:
Data is arranged geographically and temporally (2022 - 2025). 

## Data split strategy:
Models without forecasting will take the average measurement per specific time period. 
Geographic leakage: 
Split train and test sets via spatial declustering equal area grid cell approach [3] 

Models with planned forecasting:
Split train and test sets via spatial declustering equal area grid cell approach [3] 

Temporal leakage: Split train and test sets between early and late measurements (custom). 

Proposed data splits (see `notebooks/evaluation/leakage_tests.ipynb`):


## Stress tests:
