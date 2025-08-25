# AIRHAR (RadMamba)
**AIRHAR** is a learning framework built in PyTorch for radar-based human activity recognition. Developed by the [Lab of Efficient Machine Intelligence](https://www.tudemi.com) @ Delft University of Technology, AIRHAR aims to bridge the gap between machine learning and signal processing of radar system.

The framework provides a comprehensive solution for training neural network models to classify non-continuous and continuous human acivities. By leveraging state-of-the-art deep learning techniques, AIRHAR enables researchers and engineers to develop more energy-efficient wireless communication systems.

We invite you to contribute to this project by providing your own backbone neural networks, pre-trained models, or measured radar-based human activities/gesture datasets.

Our latest work: [RadMamba: Efficient Human Activity Recognition through Radar-based Micro-Doppler-Oriented Mamba State-Space Model](https://arxiv.org/abs/2504.12039)

# Project Structure
```
.
└── backbone        # Configuration files for classifiers
└── datasets        # datasets preparation files for open source datasets in RadMamba work
└── log             # Experimental log data (automatically generated)
└── modules         # Major functional modules
└── save            # Saved models
└── steps           # Implementation steps (classification .. to be continued)
└── utils           # Utility libraries
└── argument.py     # Command-line arguments and configuration
└── main.py         # Main entry point
└── model.py        # Top-level neural network models
└── project.py      # Core class for hyperparameter management and utilities

```

# Environment Setup

This project has been tested with PyTorch 2.5.1 and Ubuntu 24.04 LTS.

### Setting Up Your Environment

We recommend using Miniconda for environment management:

```bash
# Install Miniconda (Linux)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

# For MacOS, use:
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Create a Python environment with required packages
conda create -n AIRHAR python=3.13 numpy scipy pandas matplotlib tqdm adabound einops h5py scikit-learn cv2 
conda activate AIRHAR
```

### Installing PyTorch

For **Linux or Windows** systems:
- With CPU only:
  ```bash
  pip3 install torch torchvision torchaudio
  ```
- With NVIDIA GPU (CUDA 12.6):
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```
  Note: Ensure you have the latest NVIDIA GPU drivers installed to support CUDA 12.6

For **macOS** systems:
```bash
pip3 install torch torchvision torchaudio
```


## Reproducing Published Results
### Data Downloads
Please follow the instructions fro
#### Non-continuous CW Dataset (DIAT)
https://ieee-dataport.org/documents/diat-μradhar-radar-micro-doppler-signature-dataset-human-suspicious-activity-recognition
#### Non-continuous FMCW Dataset (CI4R)
https://github.com/ci4r/CI4R-Activity-Recognition-datasets
#### Continuous FMCW Dataset (UoG20)
UoG2020 dataset from Glasgow is being prepared for integration into the existing collection at  https://researchdata.gla.ac.uk/848/. 


### Data Preparation
To reproduce the datasets preprocessing shown in **RadMamba**, please change the path in data_prepare.sh to your data path:

```bash
bash data_prepare.sh
```
This script prepares three datasets: non-continuous CW dataset (DIAT), non-continuous FMCW dataset (CI4R), and continuous FMCW dataset (UoG20)

### Classification
To reproduce the classification results in Figure 4(a1) (a2) (a3), and Table IV:
```bash
bash run_experiments_DIAT.sh
bash run_experiments_CI4R.sh
bash run_experiments_UoG20.sh
```
These scripts train various classifier models across model sizes.


# Authors & Citation
If you find this repository helpful, please cite our work:
- [RadMamba: Efficient Human Activity Recognition through Radar-based Micro-Doppler-Oriented Mamba State-Space Model](https://arxiv.org/abs/2504.12039)
```
@misc{wu2025radmambaefficienthumanactivity,
      title={RadMamba: Efficient Human Activity Recognition through Radar-based Micro-Doppler-Oriented Mamba State-Space Model}, 
      author={Yizhuo Wu and Francesco Fioranelli and Chang Gao},
      year={2025},
      eprint={2504.12039},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.12039}, 
}
```

# Acknowledgment
This work was partially supported by the European Research Executive Agency (REA) under the Marie Skłodowska-Curie Actions (MSCA) Postdoctoral Fellowship program, Grant No. 101107534 (AIRHAR).

# Contributors

- **Chang Gao** - Project Lead
- **Yizhuo Wu** - Core Developer

