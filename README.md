# Project Structure

This repository contains the full implementation of the M2H-SWIR model, including LUT-based simulation, model training, fine-tuning, and evaluation.

<img width="836" height="607" alt="LUT-M2H-SWIR" src="https://github.com/user-attachments/assets/637488c5-5f28-4b8f-8c24-287b686f9af8" />

LUT-M2H-SWIR

    data/                    # Data directory (Zenodo)
       real/                  # Measured UAV and ASD data
       lut/                   # LUT-generated training data
       ancillary/             # Soil spectral data
       sensor/                # UAV sensor spectral response functions

    src/
       prosail-sim/           # LUT generation
         generate_lut.py

       train/                 # Training scripts
         train_lut_standard.py
         finetune_on_real.py
         train_data_driven.py

      eval/                  # Evaluation scripts
        evaluate_reconstruction.py
        eval_on_real.py

    models/                  # Model weights
      README.md
      requirements.txt


All raw data (excluding code) are publicly available on Zenodo (DOI: 10.5281/zenodo.18173915).

# Model Overview：M2H-SWIR Model

M2H–SWIR is a hybrid physical–deep learning framework that reconstructs full-spectrum hyperspectral reflectance (400–2500 nm) from multispectral UAV data (400–900 nm) by combining PROSAIL-PRO simulations with CNN-based learning. Model outputs are validated using ground-based ASD canopy reflectance measurements.

## 1. LUT-Based Pretraining：Generate training data and pre-trained model
### 1) Generate LUT
python src/prosail-sim/generate_lut.py
### 2) Train M2H-SWIR on LUT
python src/train/train_lut_standard.py --alpha 0.2 --beta 2.0

## 2. Fine-Tuning on Real Data
### Using a pretrained LUT model as initialization
python src/train/finetune_on_real.py --pretrained models/m2h_swir_lut_new_head.pt

## 3. Direct Data-Driven Training
### The M2H-SWIR model framework, but using only our own experimental data as input-driven data.
python src/train/train_data_driven.py --model_type m2h
### or Simple Model
python src/train/train_data_driven.py --model_type simple_cnn

## 4. Evaluation
### On LUT (and optional real ASD)
python src/eval/evaluate_reconstruction.py --model models/m2h_swir_lut_new_head.pt

### On real UAV–ASD data
python src/eval/eval_on_real.py --model models/m2h_swir_finetuned_new_head.pt


