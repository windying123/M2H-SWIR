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


