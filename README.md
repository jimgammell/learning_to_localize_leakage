# Learning to Localize Leakage of Cryptographic Sensitive Variables

This is the official implementation accompanying the paper "Learning to Localize Leakage of Cryptographic Sensitive Variables" (TMLR 2026) by Jimmy Gammell, Anand Raghunathan, Abolfazl Hashemi, and Kaushik Roy.

## Installation

This code was tested using Python 3.9.19, and certain dependencies (e.g. Captum) seem to break with more-recent versions. Follow the instructions below to install the project and its dependencies:
1) Create and activate an environment for the project. For example, to do this with [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html):
```bash
micromamba create --name leakage-localization python=3.9
micromamba activate leakage-localization
```
2) Clone the project, then install it alongside its dependencies:
```bash
git clone git@github.com:jimgammell/learning_to_localize_leakage
cd learning_to_localize_leakage
pip install -e .
```
### Downloading datasets

You may download some or all of the datasets used in our paper, and extract them to the project directory as follows:
- ASCADv1 (fixed key) ([link](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key)):
```bash
mkdir -p resources/ascadv1-fixed
cd resources/ascadv1-fixed
wget https://www.data.gouv.fr/api/1/datasets/r/e7ab6f9e-79bf-431f-a5ed-faf0ebe9b08e -O ASCAD_data.zip
unzip ASCAD_data.zip
````
- ASCADv1 (variable key) ([link](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key)):
```bash
mkdir -p resources/ascadv1-variable
cd resources/ascadv1-variable
wget https://www.data.gouv.fr/api/1/datasets/r/b4ace767-c2a4-4db4-8e01-4527b5b91f00 -O ascad-variable.h5
````
- DPAv4 (version distributed by Zaid et al.) ([link](https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/tree/master/DPA-contest%20v4)):
```bash
mkdir -p resources/dpav4
cd resources/dpav4
wget https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/raw/refs/heads/master/DPA-contest%20v4/DPAv4_dataset.zip
unzip DPAv4_dataset.zip
````
- AES-HD: Our paper used the version distributed [here](https://github.com/AISyLab/AES_HD), which appears to no longer be available. I assume [this version](https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/raw/refs/heads/master/AES_HD/AES_HD_dataset.zip) distributed by Zaid et al. is the same:
```bash
mkdir -p resources/aes-hd
cd resources/aes-hd
wget https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/raw/refs/heads/master/AES_HD/AES_HD_dataset.zip
unzip AES_HD_dataset.zip
````
- One Trace is All it Takes ([link](https://github.com/leoweissbart/MachineLearningBasedSideChannelAttackonEdDSA)):
```bash
mkdir -p resources/otiait
cd resources/otiait
wget https://github.com/leoweissbart/MachineLearningBasedSideChannelAttackonEdDSA/raw/refs/heads/master/databaseEdDSA.h5
````
- One Truth Prevails (1024-bit version) ([link](https://github.com/ECSIS-lab/one_truth_prevails/tree/main)): This dataset is distributed through Google Drive, and I'm not aware of an easy way to download it from the command line. Instead, just create the directory with the following commands, then navigate [here](https://drive.google.com/drive/folders/19ulxDZmvY5LMbwtzwp1jyGbrP3JHr4vr), download the files `a.npy`, `p_labels.txt`, `p.npy`, and move them to this directory.
```bash
mkdir -p resources/otp
cd resources/otp
````

## Citation

If you use this code or our work, please consider citing it as follows:
```bibtex
@article{Gammell_Learning_to_Localize_2026,
    author = {Gammell, Jimmy and Raghunathan, Anand and Hashemi, Abolfazl and Roy, Kaushik},
    journal = {Transactions on Machine Learning Research},
    title = {{Learning to Localize Leakage of Cryptographic Sensitive Variables}},
    url = {https://openreview.net/forum?id=9qxCSU8nDO&},
    year = {2026}
}
```

### Contact

For questions or to point out bugs, please open an issue or contact me at `jgammell@purdue.edu`.