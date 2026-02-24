# CWFormer for Traffic State Prediction

This repository contains the source code for the model **CWFormer**, implemented based on the open-source spatial-temporal prediction framework [LibCity](https://github.com/LibCity/Bigscity-LibCity). 

This code is provided for the purpose of double-blind peer review. All identifiable information has been removed.

## 📂 Core Structure

The implementation of CWFormer is seamlessly integrated into the LibCity framework. The core modifications and additions are located in the following directories:

* `libcity/model/traffic_flow_prediction/CWFormer.py`: The core network architecture of CWFormer.
* `libcity/config/model/traffic_state_pred/CWFormer.json`: The default hyperparameters for the model.
* `libcity/config/task_config.json`: The global registration of the CWFormer model.
* `run_model.py`: The main entry script for training and evaluation.

## ⚙️ Requirements

Please ensure you have the correct environment set up. The primary dependencies include:
* Python >= 3.7
* PyTorch >= 1.8.0
* Other requirements are consistent with the standard LibCity framework.

## 🚀 How to Run

You can train and evaluate the CWFormer model from scratch using the provided `run_model.py` script. 

For example, to run the model on the PeMS08 dataset, execute the following command in your terminal:

```bash
python run_model.py --task traffic_state_pred --model CWFormer --dataset PeMS08
