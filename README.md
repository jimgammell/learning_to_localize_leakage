# Learning to Localize Leakage of Cryptographic Sensitive Variables
By Jimmy Gammell, Anand Raghunathan, Abolfazl Hashemi, and Kaushik Roy

IN PROGRESS: writing README and updating code so that it is usable by other people.

Our paper can be found [here](https://arxiv.org/abs/2503.07464).

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
```
@misc{gammell2025learninglocalizeleakagecryptographic,
      title={Learning to Localize Leakage of Cryptographic Sensitive Variables}, 
      author={Jimmy Gammell and Anand Raghunathan and Abolfazl Hashemi and Kaushik Roy},
      year={2025},
      eprint={2503.07464},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.07464}, 
}
```