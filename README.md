The validation data for our experiment is our own measured data, located in the data folder. The training data needs to be generated using (1) enerate LUT. The specific range of the LUT is based on the relevant parameters of wheat in our experiment and other literature. You can directly download all the raw data (excluding code) from 10.5281/zenodo.18173915.

# 1. LUT-Based Pretraining
## 1) Generate LUT
python src/prosail-sim/generate_lut.py
## 2) Train M2H-SWIR on LUT
python src/train/train_lut_standard.py --alpha 0.2 --beta 2.0

# 2. Fine-Tuning on Real Data
## Using a pretrained LUT model as initialization
python src/train/finetune_on_real.py --pretrained models/m2h_swir_lut_new_head.pt

# 3. Direct Data-Driven Training
python src/train/train_data_driven.py --model_type m2h
## or
python src/train/train_data_driven.py --model_type simple_cnn

# 4. Evaluation
## On LUT (and optional real ASD)
python src/eval/evaluate_reconstruction.py --model models/m2h_swir_lut_new_head.pt

## On real UAV–ASD data
python src/eval/eval_on_real.py --model models/m2h_swir_finetuned_new_head.pt


