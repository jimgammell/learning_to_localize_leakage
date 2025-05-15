# Learning to Localize Leakage of Cryptographic Sensitive Variables
By REDACTED

README is in progress. Need to clean things up so that it is easy to use. Here are a few of the salient files:
- `./training_modules/adversarial_leakage_localization`: the implementation of our algorithm. Lightning module is in the `module.py` file.
- `./training_modules/supervised_deep_sca`: our supervised learning trainer for the deep learning baselines.
- `./utils/baseline_assessments/first_order_statistics.py`: our implementation of the SNR, SOSD, CPA baselines.
- `./utils/baseline_assessments/neural_net_attribution.py`: implementations of all the deep learning baselines apart from OccPOI and second-order occlusion.
- `./utils/baseline_assessments/occpoi.py`: implementation of OccPOI.
- `./utils/baseline_assessments/second_order_occlusion.py`: implementation of second-order occlusion.
- `./datasets`: contains code for the datasets we use.
- `./models/mpl_1d.py`: the neural net architecture used in our experiments.

Our paper can be found REDACTED.

## Installation

We used Python 3.9.19. It seems like things break if the Python version is too recent.
1) Create a conda environment with `<conda/mamba/micromamba> create --name=leakage-localization python=3.9`.
2) Activate it with `<conda/mamba/micromamba> activate leakage-localization`.
3) Navigate to the project directory and install required packages with `pip3 install -r requirements.txt`.

## Usage

All experiments are run via argparse commands to `src/run_trial.py`. Additionally, we have a self-contained minimal working example in `TODO`.

### Available datasets

If not already present, datasets will be auto-downloaded to the directory specified in `config/global_variables.yaml>resource_dirname` (by default, the `resources` subdirectory of the project directory) when you 
- ASCADv1 (fixed) [(link)](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key): selected with `--dataset=ascadv1_fixed`.
- ASCADv1 (variable) [(link)](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key): selected with `--dataset=ascadv1_variable`.
- DPAv4 (Zaid version) [(link)](https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA): selected with `--dataset=dpav4`.
- AES-HD [(link)](https://github.com/AISyLab/AES_HD): selected with `--dataset=aes_hd`.
- One Truth Prevails (1024-bit) [(link)](https://github.com/ECSIS-lab/one_truth_prevails): selected with `--dataset=otp`.
- One Trace is All it Takes 

## Citation

Please cite the following paper if you use our work:
REDACTED