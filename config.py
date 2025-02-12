import torch

# Random seeds for reproducibility
SEED = 46
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters
NUM_SAMPLES = 100
INPUT_DIM = 9  # 3x3 SPD matrices flattened
LATENT_DIM = 2

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.7
VAL_RATIO = 0.20
TEST_RATIO = 0.50

# SPD matrix parameters
SPD_EPS = 1e-4  # Small value to ensure positive definiteness