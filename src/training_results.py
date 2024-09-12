# %%
from src.models.DNNClassifier import DNNClassifier, ModelTrainer,DNNResNet,RNNClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn.functional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn
from datetime import date
import matplotlib as mpl
import numpy as np
from scipy import stats
import seaborn as sns
import logging
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer
)

# %% Initial Setup
# Parameters
DATA_SET = "10K_05"
RECORD = False

ACTIVATION = nn.Softsign
EPOCHS = 300
LEARNING_RATE = 0.001
NUM_LAYERS = 50
NUM_NEURONS = 32
scaler = MaxAbsScaler()
TEST_SIZE = 0.05

INPUT_SIZE = 10
OUTPUT_SIZE = 1

output_columns = ['m0_maximum [kg]',]

# Display plots in separate window
# mpl.use('macosx')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %% Data loading and scaling
#DATA_PATH = Path("data/processed")
#DATA_NAME = "transfer_data_" + DATA_SET + ".csv"

DATA_PATH = Path("data/low_thrust/")
DATA_NAME = "low_thrust_segment_statistics_m0.csv"



# create saved_models directory
MODEL_PATH = Path("src/models/saved_models")
today = date.today()
#MODEL_NAME = str(today) + "_" + DATA_SET + ".pth"
MODEL_NAME = str(today) + ".pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

df = pd.read_csv(DATA_PATH / DATA_NAME)



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
model_01 = RNNClassifier(INPUT_SIZE, OUTPUT_SIZE, NUM_LAYERS, NUM_NEURONS, ACTIVATION)
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

def unscale(scaled_value):
    unscaled_value = scaler.inverse_transform(scaled_value)
    return pd.DataFrame(unscaled_value)



# %% Make estimates
model_01.eval()
with torch.inference_mode():
    pred_train = model_01(x_train) #Prediction on the train data
    pred_test = model_01(x_test)#Prediction on the test data

df_result_pred_train_scaled = pd.concat([pd.DataFrame(x_test.cpu().numpy()), pd.DataFrame(pred_train.cpu().numpy())],
                                    ignore_index=True,
                                    axis='columns')
df_result_y_train_scaled = pd.concat([pd.DataFrame(x_test.cpu().numpy()), pd.DataFrame(y_train.cpu().numpy())],
                                    ignore_index=True,
                                    axis='columns')
df_result_y_test_scaled = pd.concat([pd.DataFrame(x_test.cpu().numpy()), pd.DataFrame(y_test.cpu().numpy())],
                                    ignore_index=True,
                                    axis='columns')
df_result_pred_test_scaled = pd.concat([pd.DataFrame(x_test.cpu().numpy()), pd.DataFrame(pred_test.cpu().numpy())],
                                    ignore_index=True,
                                    axis='columns')


# Apply to unscale function to each column of inputs arrays
df_result_pred_train = unscale(df_result_pred_train_scaled)
df_result_y_test = unscale(df_result_y_test_scaled)
df_result_y_train = unscale(df_result_y_train_scaled)
df_result_pred_test = unscale(df_result_pred_test_scaled)


y_test = df_result_y_test.iloc[:, -len(output_columns):].values  # Dynamically extract columns
pred_test = df_result_pred_test.iloc[:, -len(output_columns):].values  # Dynamically extract columns
pred_train = df_result_pred_train.iloc[:, -len(output_columns):].values
y_train = df_result_y_train.iloc[:, -len(output_columns):].values  # Training actual values




errors_train = (y_train - pred_train)
errors_test = (y_test - pred_test)

print("Print Statistics")
n_points = 50

for i, column in enumerate(output_columns):
    print(f"{column:<{max(len(col) for col in output_columns)}} | "
          f"train MAE: {np.mean(np.abs(errors_train[:n_points, i])):.4f} [Kg] | "
          f"train ME: {np.mean(errors_train[:n_points, i]):.4f} [Kg] | "
          f"Test MAE: {np.mean(np.abs(errors_test[:n_points, i])):.4f} [Kg] | "
          f"Test ME: {np.mean(errors_test[:n_points, i]):.4f} [Kg]")


# %% Saving model
torch.save(obj=model_01.state_dict(), f=MODEL_SAVE_PATH)
logging.info("Model has been written to " + f"{MODEL_SAVE_PATH}")
plt.show()
