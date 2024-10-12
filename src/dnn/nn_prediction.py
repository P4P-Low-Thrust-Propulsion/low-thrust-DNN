# %%
from src.models.DNN import DNNRegressor
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import matplotlib as mpl
import logging
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer
)

# %% Initial Setup
lambert = False

# Parameters
EPOCHS = 400
LEARNING_RATE = 0.001
NUM_LAYERS = 12
NUM_NEURONS_1 = 256
NUM_NEURONS_2 = 256
NUM_NEURONS_3 = 128
x_scaler = joblib.load("src/models/saved_models/x_scaler_v2.pkl")
y_scaler = joblib.load("src/models/saved_models/y_scaler_v2.pkl")
TEST_SIZE = 0.2
ACTIVATION = nn.Softsign

INPUT_SIZE = 10
OUTPUT_SIZE = 2

RECORD = False

if lambert:
    output_columns = ['v0_x [km/s]', 'v0_y [km/s]', 'vf_x [km/s]', 'vf_y [km/s]']
    label = 'km/s'
else:
    output_columns = ['m0_maximum [kg]', 'm1_maximum [kg]']
    label = 'kg'

# Display plots in separate window
# mpl.use('macosx')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if lambert:
    input_columns = [
        'rel_dist_x',
        'rel_dist_y',
        'tof'
    ]
else:
    input_columns = [
        'r0 [AU]',
        'vr0 [km/s]',
        'vt0 [km/s]',
        'r1 [AU]',
        't1 [AU]',
        'n1 [AU]',
        'vr1 [km/s]',
        'vt1 [km/s]',
        'vn1 [km/s]',
        'tof [days]'
    ]

# %% Data loading and scaling
if lambert:
    DATA_PATH = Path("data/lambert/datasets/processed")
    DATA_NAME = "transfer_data_10K_01.csv"
else:
    DATA_PATH = Path("data/low_thrust/datasets/processed")
    DATA_NAME = "new_transfer_statistics_500K_v2.csv"

if lambert:
    MODEL_PATH = Path("src/models/saved_models/2024-09-21_lambert_10K.pth")
else:
    MODEL_PATH = Path("src/models/saved_models/2024-10-03_low_thrust_500K.pth")

MODEL_SAVE_PATH = MODEL_PATH

# Check if CUDA (NVIDIA GPU) is available
cuda_available = torch.cuda.is_available()
logging.info("CUDA (NVIDIA GPU) available: " + str(cuda_available))

# Move your model and processed tensors to the GPU (if available)
device = torch.device("cuda" if cuda_available else "mps")

torch.manual_seed(42)
model_01 = DNNRegressor(INPUT_SIZE, OUTPUT_SIZE, NUM_NEURONS_1, NUM_NEURONS_2, NUM_NEURONS_3)
model_01.to(device)
model_01.load_state_dict(torch.load(MODEL_SAVE_PATH))

df = pd.read_csv(DATA_PATH / DATA_NAME)
df_original = pd.read_csv(DATA_PATH / DATA_NAME)

df_Features = df.iloc[:, :INPUT_SIZE]
df_Labels = df.iloc[:, -OUTPUT_SIZE:]

df_Features = x_scaler.transform(df_Features)
df_Labels = y_scaler.transform(df_Labels)

data_Features = df_Features
data_Labels = df_Labels

# Fit and transform the features
x = torch.tensor(data_Features, dtype=torch.float32)
y = torch.tensor(data_Labels, dtype=torch.float32)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)

# %% Make estimates
model_01.eval()
with torch.no_grad():
    pred_test = model_01(x_test)
    pred_train = model_01(x_train)
    

# %% Unscale the values
def unscale(scaled_value):
    unscaled_value = y_scaler.inverse_transform(scaled_value)
    return pd.DataFrame(unscaled_value)


df_result_pred_train_scaled = pd.DataFrame(pred_train.cpu().numpy())
df_result_y_train_scaled = pd.DataFrame(y_train.cpu().numpy())
df_result_y_test_scaled = pd.DataFrame(y_test.cpu().numpy())
df_result_pred_test_scaled = pd.DataFrame(pred_test.cpu().numpy())

# Apply to unscale function to each column of inputs arrays
df_result_pred_training = unscale(df_result_pred_train_scaled)
df_result_y_test = unscale(df_result_y_test_scaled)
df_result_y_train = unscale(df_result_y_train_scaled)
df_result_pred_test = unscale(df_result_pred_test_scaled)

# %% Plot 1 (Distribution Plot)
plt.ion()
# Create subplots (4x3 grid for 12 subplots, one will remain empty)
if lambert:
    fig, axes = plt.subplots(1, 3, figsize=(9, 6))
else:
    fig, axes = plt.subplots(4, 3, figsize=(9, 6))
axes = axes.flatten()

# Loop over each input column and create a histogram
for i, column in enumerate(input_columns):
    axes[i].hist(df_original[column], bins=20, color='black')
    axes[i].set_title(column)
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')

# If there are any remaining subplots that don't have data, hide them
for j in range(len(input_columns), len(axes)):
    fig.delaxes(axes[j])

# Set overall title
fig.suptitle('Training Data Distribution: Inputs')

# Adjust layout for better appearance
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(9, 6))
axes = axes.flatten()

# Loop over each input column and create a histogram
for i, column in enumerate(output_columns):
    axes[i].hist(df_original[column], bins=20, color='black')
    axes[i].set_title(column)
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')

# If there are any remaining subplots that don't have data, hide them
for j in range(len(output_columns), len(axes)):
    fig.delaxes(axes[j])

# Set overall title
fig.suptitle('Training Data Distribution: Outputs')

# Adjust layout for better appearance
plt.tight_layout()
plt.show()


# Extract test data
y_test = df_result_y_test.iloc[:, -len(output_columns):].values  # Dynamically extract columns
pred_test = df_result_pred_test.iloc[:, -len(output_columns):].values  # Dynamically extract columns
pred_train = df_result_pred_training.iloc[:, -len(output_columns):].values
y_train = df_result_y_train.iloc[:, -len(output_columns):].values  # Training actual values

# Analysis
error_train = (y_train - pred_train)
error_train_percentage = (error_train / y_train) * 100
filtered_error_train_percentage = np.where(error_train_percentage > 10000, 10000, error_train_percentage)
filtered_error_train_percentage = np.where(filtered_error_train_percentage < -10000, -10000, filtered_error_train_percentage)

error_test = (y_test - pred_test)
error_test_percentage = (error_test / y_test) * 100
filtered_error_test_percentage = np.where(error_test_percentage > 10000, 10000, error_test_percentage)
filtered_error_test_percentage = np.where(filtered_error_test_percentage < -10000, -10000, filtered_error_test_percentage)

# %% Plot 2 (Prediction vs Actual plot)
n_columns = len(output_columns)
i = 0

# Split into two halves
mid_point = n_columns // 2
column_sets = [output_columns[:mid_point], output_columns[mid_point:]]

for column_set in column_sets:
    fig, axes = plt.subplots(2, len(column_set), figsize=(9, 6))  # 2 rows: Train and Test

    # Handle case where there's only 1 output column
    if len(column_set) == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # Make sure axs is 2D even for 1 output column

    # Plot each component dynamically for both training and testing data
    for j, col_name in enumerate(column_set):
        # Training data (top row)
        ax_train = axes[0, j]
        ax_train.scatter(y_train[:, i], pred_train[:, i], color='blue', alpha=0.5, s=4)
        ax_train.plot([min(y_train[:, i]), max(y_train[:, i])], [min(y_train[:, i]), max(y_train[:, i])],
                      color='green', linestyle='--', label='Target value')
        ax_train.set_title(f"Predicted vs Actual {col_name} (Training)")
        ax_train.set_xlabel(f"Actual {col_name}")
        ax_train.set_ylabel(f"Predicted {col_name}")
        ax_train.legend(loc="best", fontsize='small')

        # Test data (bottom row)
        ax_test = axes[1, j]
        ax_test.scatter(y_test[:, i], pred_test[:, i], color='blue', alpha=0.5, s=4)
        ax_test.plot([min(y_test[:, i]), max(y_test[:, i])], [min(y_test[:, i]), max(y_test[:, i])],
                     color='green', linestyle='--', label='Target value')
        ax_test.set_title(f"Predicted vs Actual {col_name} (Test)")
        ax_test.set_xlabel(f"Actual {col_name}")
        ax_test.set_ylabel(f"Predicted {col_name}")
        ax_test.legend(loc="best", fontsize='small')

        i += 1

    # Adjust layout for better appearance
    plt.tight_layout()
    plt.show()

# %% Plot 3 (Residue)
n_columns = len(output_columns)
i = 0

# Split into two halves
mid_point = n_columns // 2
column_sets = [output_columns[:mid_point], output_columns[mid_point:]]

for column_set in column_sets:
    fig, axes = plt.subplots(2, len(column_set), figsize=(9, 6))  # 2 rows: Train and Test

    # Handle case where there's only 1 output column
    if len(column_set) == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # Make sure axs is 2D even for 1 output column

    # Loop through each output column and plot residuals for both training and test data
    for j, col_name in enumerate(column_set):
        # Calculate Mean Error (ME) and Mean Absolute Error (MAE) for training
        me_train = np.mean(error_train[:, i])
        mae_train = np.mean(np.abs(error_train[:, i]))

        # Calculate Mean Error (ME) and Mean Absolute Error (MAE) for test
        me_test = np.mean(error_test[:, i])
        mae_test = np.mean(np.abs(error_test[:, i]))

        # Training residual plot (top row)
        ax_train = axes[0, j]
        ax_train.scatter(y_train[:, i], error_train[:, i], color='blue', alpha=0.5, s=4)
        ax_train.axhline(y=0, color='green', linestyle='--', label='Target residual')
        ax_train.axhline(y=me_train, color='blue', linestyle='--', label=f'Mean residual (ME): {me_train:.2f} ' + label)
        ax_train.axhline(y=mae_train, color='orange', linestyle='--', label=f'Mean abs residual (MAE): {mae_train:.2f} ' + label)
        ax_train.set_title(f"Residual vs Actual {col_name} (Training)")
        ax_train.set_xlabel(f"Actual {col_name}")
        ax_train.set_ylabel("Residuals [" + label + "]")
        ax_train.legend(loc="best", fontsize='small')

        # Test residual plot (bottom row)
        ax_test = axes[1, j]
        ax_test.scatter(y_test[:, i], error_test[:, i], color='blue', alpha=0.5, s=4)
        ax_test.axhline(y=0, color='green', linestyle='--', label='Target residual')
        ax_test.axhline(y=me_test, color='blue', linestyle='--', label=f'Mean residual (ME): {me_test:.2f} '+label)
        ax_test.axhline(y=mae_test, color='orange', linestyle='--', label=f'Mean abs residual (MAE): {mae_test:.2f} ' + label)
        ax_test.set_title(f"Residual vs Actual {col_name} (Test)")
        ax_test.set_xlabel(f"Actual {col_name}")
        ax_test.set_ylabel("Residuals [" + label + "]")
        ax_test.legend(loc="best", fontsize='small')

        i += 1

    # Adjust layout for better appearance
    plt.tight_layout()
    plt.show()

# %% Plot 4 (Residue percentage)
n_columns = len(output_columns)
i = 0

# Split into two halves
mid_point = n_columns // 2
column_sets = [output_columns[:mid_point], output_columns[mid_point:]]

for column_set in column_sets:
    fig, axes = plt.subplots(2, len(column_set), figsize=(9, 6))  # 2 rows, num_columns columns of subplots

    # Handle case where there's only 1 output column
    if len(column_set) == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # Make sure axs is 2D even for 1 output column

    # Loop through each output column and plot residual percentages for both training and test
    for j, col_name in enumerate(column_set):
        # Calculate mean and mean absolute residual percentages for training
        me_train = np.mean(filtered_error_train_percentage[:, i])
        mae_train = np.mean(np.abs(filtered_error_train_percentage[:, i]))

        # Calculate mean and mean absolute residual percentages for test
        me_test = np.mean(filtered_error_test_percentage[:, i])
        mae_test = np.mean(np.abs(filtered_error_test_percentage[:, i]))

        # Training residual plot (top row)
        ax_train = axes[0, j]  # Get the current axis for training data
        ax_train.scatter(y_train[:, i], filtered_error_train_percentage[:, i], color='blue', alpha=0.5, s=4)
        ax_train.axhline(y=0, color='green', linestyle='--', label='Target residual')
        ax_train.axhline(y=me_train, color='blue', linestyle='--', label=f'Mean residual: {me_train:.2f}%')
        ax_train.axhline(y=mae_train, color='orange', linestyle='--', label=f'Mean abs residual: {mae_train:.2f}%')
        ax_train.set_title(f"Residual % vs Actual {col_name} (Training)")
        ax_train.set_xlabel(f"Actual {col_name}")
        ax_train.set_ylabel("Residues [%]")
        ax_train.legend(loc="best", fontsize='small')

        # Test residual plot (bottom row)
        ax_test = axes[1, j]  # Get the current axis for test data
        ax_test.scatter(y_test[:, i], filtered_error_test_percentage[:, i], color='blue', alpha=0.5, s=4)
        ax_test.axhline(y=0, color='green', linestyle='--', label='Target residual')
        ax_test.axhline(y=me_test, color='blue', linestyle='--', label=f'Mean residual: {me_test:.2f}%')
        ax_test.axhline(y=mae_test, color='orange', linestyle='--', label=f'Mean abs residual: {mae_test:.2f}%')
        ax_test.set_title(f"Residual % vs Actual {col_name} (Test)")
        ax_test.set_xlabel(f"Actual {col_name}")
        ax_test.set_ylabel("Residues [%]")
        ax_test.legend(loc="best", fontsize='small')

        i += 1

    # Adjust layout for better appearance
    plt.tight_layout()
    plt.show()

# Define percentile levels to plot
percentiles = np.linspace(0, 100, 101)  # Percentiles from 0 to 100

# Compute percentiles for training and test errors
test_percentiles = np.percentile(filtered_error_test_percentage, percentiles, axis=0)

# Plot percentiles for each output column
n_columns = len(output_columns)
mid_point = n_columns // 2
column_sets = [output_columns[:mid_point], output_columns[mid_point:]]

fig, ax = plt.subplots(figsize=(9, 6))  # Single subplot

# Loop through each output column and plot percentiles on the same subplot
for j, col_name in enumerate(output_columns):
    # Plot test percentiles
    ax.plot(percentiles, test_percentiles[:, j], label=f'{col_name} Test Error Percentiles')

# Add the mean prediction absolute error (MPAE) line for reference
ax.axhline(y=2.1, color='green', linestyle='--', label='MPAE: 2.1%')

# Add labels and title
ax.set_title("Percentile Error Distribution Across Output Columns")
ax.set_xlabel('Percentile')
ax.set_ylabel('Error [%]')

# Add legend
ax.legend(loc="best", fontsize='small')

# Adjust layout for better appearance
plt.tight_layout()
plt.ioff()
plt.show()

print("Plotting complete")
