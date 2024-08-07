# %%
from src.models.DNNClassifier import DNNClassifier, ModelTrainer
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn.functional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn
from datetime import date
import logging

# %% Initial Setup
# Parameters
DATA_SET = "10K_01"
RECORD = False
LEARNING_RATE = 0.01
EPOCHS = 500
TEST_SIZE = 0.2
INPUT_SIZE = 3
OUTPUT_SIZE = 4
NUM_LAYERS = 9
NUM_NEURONS = 20
ACTIVATION = nn.SELU

# Display plots in separate window
# mpl.use('macosx')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %% Data loading and scaling
DATA_PATH = Path("data/lambert/processed")
DATA_NAME = "transfer_data_" + DATA_SET + ".csv"

# create saved_models directory
MODEL_PATH = Path("src/models/saved_models")
today = date.today()
MODEL_NAME = str(today) + "_" + DATA_SET + ".pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

df = pd.read_csv(DATA_PATH / DATA_NAME)

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the scaler to each column separately
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

df_Features = df.iloc[:, :INPUT_SIZE]
df_Labels = df.iloc[:, -OUTPUT_SIZE:]

data_Features = df_Features.values
data_Labels = df_Labels.values

# Fit and transform the features
x = torch.tensor(data_Features, dtype=torch.float32)
y = torch.tensor(data_Labels, dtype=torch.float32)

# Check if CUDA (NVIDIA GPU) is available
cuda_available = torch.cuda.is_available()

logging.info("CUDA (NVIDIA GPU) available: " + str(cuda_available))

# Move your model and processed tensors to the GPU (if available)
device = torch.device("cuda" if cuda_available else "mps")

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)

# %% Construct a model class that subclasses nn.Module
torch.manual_seed(42)

# create an instance of the model
model_01 = DNNClassifier(INPUT_SIZE, OUTPUT_SIZE, NUM_LAYERS, NUM_NEURONS, ACTIVATION)
model_01.state_dict()

# Your model and processed setup
model_01.to(device)
x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)

# %% Train model
# Setting up a loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params=model_01.parameters(), lr=LEARNING_RATE, momentum=0.75)  # lr = learning rate

model_01_trainer = ModelTrainer(model_01, loss_fn, optimizer, DATA_NAME)
model_01_trainer.train(EPOCHS, x_train, x_test, y_train, y_test, RECORD)
model_01_trainer.plot_training_curves()

# %% Saving model
torch.save(obj=model_01.state_dict(), f=MODEL_SAVE_PATH)
logging.info("Model has been written to " + f"{MODEL_SAVE_PATH}")
plt.show()
