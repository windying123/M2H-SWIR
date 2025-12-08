"""Global configuration for the M2H-SWIR project.

Edit these paths and hyperparameters according to your environment.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SENSOR_DIR = DATA_DIR / "sensor"
LUT_DIR = DATA_DIR / "lut"
REAL_DIR = DATA_DIR / "real"
MODEL_DIR = PROJECT_ROOT / "models"

SENSOR_SRF_PATH = SENSOR_DIR / "SRF.xlsx"

WL_MIN = 400
WL_MAX = 2500
WL_STEP = 1
NUM_WL = int((WL_MAX - WL_MIN) / WL_STEP) + 1

# Example nitrogen-sensitive wavelengths in nm (customize as needed)
N_SENSITIVE_WAVELENGTHS = [490,550,705, 717, 740, 1510, 2100, 2170, 2200, 2250,840, 870, 890, 1200, 1645, 1680,1715,1680,1730,1940,2060,2180,2240]


DEFAULT_ALPHA = 0.22
DEFAULT_BETA = 1.3

BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 1e-4
PATIENCE = 30
RANDOM_SEED = 42
