# %%
from src.models.DNNClassifier import DNNClassifier
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer
)
import numpy as np
import torch.nn as nn
import torch.nn.functional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from scipy import stats
import logging
import gc

# %% Initial Setup
# Parameters

RECORD = False
ACTIVATION = nn.ReLU
EPOCHS = 900
LEARNING_RATE = 0.5
NUM_LAYERS = 6
NUM_NEURONS = 32
scaler = MaxAbsScaler()
TEST_SIZE = 0.2

INPUT_SIZE = 10
OUTPUT_SIZE = 1

output_columns = ['m0_maximum [kg]',]


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

# input_columns = [
#     'a0 [AU]',
#     'e0',
#     'omega0 [deg]',
#     'theta0 [deg]',
#     'a1 [Au]',
#     'e1',
#     'i1 [deg]',
#     'Omega1 [deg]',
#     'omega1 [deg]',
#     'theta1 [deg]',
#     'tof [days]'
# ]



# Display plots in separate window
# mpl.use('macosx')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %% Data loading and scaling
#DATA_PATH = Path("data/lambert/processed")
#DATA_NAME = "transfer_data_" + DATA_SET + ".csv"

DATA_PATH = Path("data/low_thrust/")
DATA_NAME = "low_thrust_segment_statistics_.csv"

MODEL_PATH = Path("src/models/saved_models/2024-09-11.pth")
MODEL_SAVE_PATH = MODEL_PATH

# Check if CUDA (NVIDIA GPU) is available
cuda_available = torch.cuda.is_available()
logging.info("CUDA (NVIDIA GPU) available: " + str(cuda_available))

# Move your model and processed tensors to the GPU (if available)
device = torch.device("cuda" if cuda_available else "mps")
torch.manual_seed(42)
model_01 = DNNClassifier(INPUT_SIZE, OUTPUT_SIZE, NUM_LAYERS, NUM_NEURONS, ACTIVATION)
model_01.to(device)
model_01.load_state_dict(torch.load(MODEL_SAVE_PATH))


df = pd.read_csv(DATA_PATH / DATA_NAME)
df_original = pd.read_csv(DATA_PATH / DATA_NAME)


# Fit and transform the scaler to each column separately
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

df_Features = df.iloc[:, :INPUT_SIZE]
df_Labels = df.iloc[:, -OUTPUT_SIZE:]

data_Features = df_Features.values
data_Labels = df_Labels.values

# Fit and transform the features
x = torch.tensor(data_Features, dtype=torch.float32)
y = torch.tensor(data_Labels, dtype=torch.float32)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)

# %% Make estimates
model_01.eval()
with torch.inference_mode():
    pred_test = model_01(x_test)
    pred_train = model_01(x_train)
    


# %% Unscale the values
def unscale(scaled_value):
    unscaled_value = scaler.inverse_transform(scaled_value)
    return pd.DataFrame(unscaled_value)
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
df_result_pred_training = unscale(df_result_pred_train_scaled)
df_result_y_test = unscale(df_result_y_test_scaled)
df_result_y_train = unscale(df_result_y_train_scaled)
df_result_pred_test = unscale(df_result_pred_test_scaled)



# %% Plotting

# Create subplots (4x3 grid for 12 subplots, one will remain empty)
fig, axes = plt.subplots(4, 3, figsize=(18, 14))
axes = axes.flatten()

# Loop over each input column and create a histogram
for i, column in enumerate(input_columns):
    axes[i].hist(df_original[column], bins=11, color='black')
    axes[i].set_title(column)
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Number of Data')

# If there are any remaining subplots that don't have data, hide them
for j in range(len(input_columns), len(axes)):
    fig.delaxes(axes[j])

# Set overall title
fig.suptitle('COE Training Data Distribution', fontsize=16)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure
plt.savefig('training_data_distribution.png')

# Show the figure
#plt.show()


#Plot 2

n_points = 50 # Number of points to plot (adjust as needed)
results = range(n_points)

# Define the output columns dynamically
output_columns = ['m0_maximum [kg]', 'm1_maximum [kg]']  # Add more columns if needed

# Extracting data for plotting
x_test_n = x_test[:n_points, :]
x_test_n = x_test_n.cpu().numpy()

# Extract test data
y_test = df_result_y_test.iloc[:, -len(output_columns):].values[:n_points]  # Dynamically extract columns
pred_test = df_result_pred_test.iloc[:, -len(output_columns):].values[:n_points]  # Dynamically extract columns
pred_train = df_result_pred_training.iloc[:, -len(output_columns):].values[:n_points]
y_train = df_result_y_train.iloc[:, -len(output_columns):].values[:n_points]  # Training actual values


# Create a figure with four subplots (2 rows, 2 columns)
fig, axs = plt.subplots(2, len(output_columns), figsize=(12, 10))  # 2 rows, 2 columns of subplots
if len(output_columns) == 1:
    axes = [axes]

# Plot each component dynamically for training and testing in subplots
for i, col_name in enumerate(output_columns):  # Loop over output columns dynamically
    # Training plots
    axs[0, i].scatter(results[:n_points], y_train[:, i], c='green', label=f'y_train {col_name}')
    axs[0, i].scatter(results[:n_points], pred_train[:, i], c='blue', label=f'pred_train {col_name}')
    axs[0, i].set_xlabel('transfer')
    axs[0, i].set_ylabel(f'{col_name}')
    axs[0, i].set_title(f'Scatter Plot of {col_name} Training Predictions')
    axs[0, i].legend()

    # Test plots
    axs[1, i].scatter(results[:n_points], y_test[:, i], c='green', label=f'y_test {col_name}')
    axs[1, i].scatter(results[:n_points], pred_test[:, i], c='blue', label=f'pred_test {col_name}')
    axs[1, i].set_xlabel('transfer')
    axs[1, i].set_ylabel(f'{col_name}')
    axs[1, i].set_title(f'Scatter Plot of {col_name} Test Predictions')
    axs[1, i].legend()

# Adjust layout for better appearance
plt.tight_layout()
plt.show()


# Analysis
error_train = (y_train - pred_train)
error_train_percentage = (error_train / y_train) * 100
filtered_error_train_percentage = np.where(error_train_percentage > 100, 100, error_train_percentage)
filtered_error_train_percentage = np.where(filtered_error_train_percentage < -100, -100, filtered_error_train_percentage)


error_test = (y_test - pred_test)
error_test_percentage = (error_test / y_test) * 100
filtered_error_test_percentage = np.where(error_test_percentage > 100, 100, error_test_percentage)
filtered_error_test_percentage = np.where(filtered_error_test_percentage < -100, -100, filtered_error_test_percentage)

# Plot 2 (Prediction vs Actual plot)
n_bins = 1500# Create a figure with 2 rows and len(output_columns) columns of subplots
fig, axes = plt.subplots(2, len(output_columns), figsize=(15, 7.5))  # 2 rows: Train and Test; len(output_columns) columns
if len(output_columns) == 1:
    axes = [axes]

# Plot each component dynamically for both training and testing data
for i, col_name in enumerate(output_columns):
    # Training data (top row)
    ax_train = axes[0, i]
    ax_train.scatter(y_train[:, i], pred_train[:, i], color='blue', alpha=0.5, s=4)
    ax_train.plot([min(y_train[:, i]), max(y_train[:, i])], [min(y_train[:, i]), max(y_train[:, i])],
                  color='green', linestyle='--')
    ax_train.set_title(f"Predicted vs Actual {col_name} (Training)")
    ax_train.set_xlabel(f"Actual {col_name}")
    ax_train.set_ylabel(f"Predicted {col_name}")

    # Test data (bottom row)
    ax_test = axes[1, i]
    ax_test.scatter(y_test[:, i], pred_test[:, i], color='blue', alpha=0.5, s=4)
    ax_test.plot([min(y_test[:, i]), max(y_test[:, i])], [min(y_test[:, i]), max(y_test[:, i])],
                 color='green', linestyle='--')
    ax_test.set_title(f"Predicted vs Actual {col_name} (Test)")
    ax_test.set_xlabel(f"Actual {col_name}")
    ax_test.set_ylabel(f"Predicted {col_name}")

# Adjust layout for better appearance
plt.tight_layout()
plt.show()




#Plot 2
# Create a figure with 2 rows and len(output_columns) columns of subplots
fig, axes = plt.subplots(2, len(output_columns), figsize=(15, 10))  # 2 rows: Train and Test; len(output_columns) columns
# If there's only one column, `axes` will not be a list of lists, so we handle that case
if len(output_columns) == 1:
    axes = [axes]

# Loop through each output column and plot residuals for both training and test data
for i, col_name in enumerate(output_columns):
    # Calculate Mean Error (ME) and Mean Absolute Error (MAE) for training
    me_train = np.mean(error_train[:n_points, i])
    mae_train = np.mean(np.abs(error_train[:n_points, i]))

    # Calculate Mean Error (ME) and Mean Absolute Error (MAE) for test
    me_test = np.mean(error_test[:n_points, i])
    mae_test = np.mean(np.abs(error_test[:n_points, i]))

    # Training residual plot (top row)
    ax_train = axes[0, i]
    ax_train.scatter(y_train[:n_points, i], error_train[:n_points, i], color='blue', alpha=0.5, s=4)
    ax_train.axhline(y=0, color='green', linestyle='--', label='Target residual')
    ax_train.axhline(y=me_train, color='blue', linestyle='--', label=f'Mean residual (ME): {me_train:.2f} kg')
    ax_train.axhline(y=mae_train, color='orange', linestyle='--', label=f'Mean abs residual (MAE): {mae_train:.2f} kg')
    ax_train.set_title(f"Residual vs Actual {col_name} (Training)")
    ax_train.set_xlabel(f"Actual {col_name} [kg]")
    ax_train.set_ylabel("Residuals [kg]")
    ax_train.legend()

    # Test residual plot (bottom row)
    ax_test = axes[1, i]
    ax_test.scatter(y_test[:n_points, i], error_test[:n_points, i], color='blue', alpha=0.5, s=4)
    ax_test.axhline(y=0, color='green', linestyle='--', label='Target residual')
    ax_test.axhline(y=me_test, color='blue', linestyle='--', label=f'Mean residual (ME): {me_test:.2f} kg')
    ax_test.axhline(y=mae_test, color='orange', linestyle='--', label=f'Mean abs residual (MAE): {mae_test:.2f} kg')
    ax_test.set_title(f"Residual vs Actual {col_name} (Test)")
    ax_test.set_xlabel(f"Actual {col_name} [kg]")
    ax_test.set_ylabel("Residuals [kg]")
    ax_test.legend()

# Adjust layout for better appearance
plt.tight_layout()
plt.show()







# Plot 3 (Residue percentage)
# Create a figure with 2 rows (one for training, one for test) and len(output_columns) columns
fig, axes = plt.subplots(2, len(output_columns), figsize=(15, 10))  # 2 rows, num_columns columns of subplots

# If there's only one column, `axes` will not be a list of lists, so we handle that case
if len(output_columns) == 1:
    axes = [axes]

# Loop through each output column and plot residual percentages for both training and test
# Loop through each output column and plot residual percentages for both training and test
for i, col_name in enumerate(output_columns):
    # Calculate mean and mean absolute residual percentages for training
    me_train = np.mean(filtered_error_train_percentage[:n_points, i])
    mae_train = np.mean(np.abs(filtered_error_train_percentage[:n_points, i]))

    # Calculate mean and mean absolute residual percentages for test
    me_test = np.mean(filtered_error_test_percentage[:n_points, i])
    mae_test = np.mean(np.abs(filtered_error_test_percentage[:n_points, i]))

    # Training residual plot (top row)
    ax_train = axes[0, i]  # Get the current axis for training data
    ax_train.scatter(y_train[:n_points, i], filtered_error_train_percentage[:n_points, i], color='blue', alpha=0.5, s=4)
    ax_train.axhline(y=0, color='green', linestyle='--', label='Target residual')
    ax_train.axhline(y=me_train, color='blue', linestyle='--', label=f'Mean residual: {me_train:.2f}%')
    ax_train.axhline(y=mae_train, color='orange', linestyle='--', label=f'Mean abs residual: {mae_train:.2f}%')
    ax_train.set_title(f"Residual % vs Actual {col_name} (Training)")
    ax_train.set_xlabel(f"Actual {col_name} [kg]")
    ax_train.set_ylabel("Residues (%)")
    ax_train.legend()

    # Test residual plot (bottom row)
    ax_test = axes[1, i]  # Get the current axis for test data
    ax_test.scatter(y_test[:n_points, i], filtered_error_test_percentage[:n_points, i], color='blue', alpha=0.5, s=4)
    ax_test.axhline(y=0, color='green', linestyle='--', label='Target residual')
    ax_test.axhline(y=me_test, color='blue', linestyle='--', label=f'Mean residual: {me_test:.2f}%')
    ax_test.axhline(y=mae_test, color='orange', linestyle='--', label=f'Mean abs residual: {mae_test:.2f}%')
    ax_test.set_title(f"Residual % vs Actual {col_name} (Test)")
    ax_test.set_xlabel(f"Actual {col_name} [kg]")
    ax_test.set_ylabel("Residues (%)")
    ax_test.legend()

# Adjust layout for better appearance
plt.tight_layout()
plt.show()


# Assuming pred_test_velocities has two columns, e.g., 'pred_test_1' and 'pred_test_2'

index = 0
# Create subplots (2 subplots per input column)
for column in input_columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # First subplot for pred_test_1
    axes[0].scatter(x_test_n[:,index], error_test[:n_points,0], alpha=0.5)


    axes[0].set_title(f'{column} vs m0_error[kg]')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('m0_error[kg]')

    # Second subplot for pred_test_2
    axes[1].scatter(x_test_n[:,index], error_test[:n_points, 1], alpha=0.5)
    axes[1].set_title(f'{column} vs m1_error[kg]')
    axes[1].set_xlabel(column)
    axes[1].set_ylabel('m1_error[kg]')
    index = index + 1
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure for this input column
    #plt.savefig(f'scatter_{column}.png')

    # Show the figure
    #plt.show()

print("Plotting complete")

#
# n = 10  # Number of points to plot (adjust as needed)
# results = range(n)
#
# # Extracting data for plotting (assuming df_result_y_test and df_result_pred_test contain your data)
# y_test_velocities = df_result_y_test.iloc[:, -4:].values[:n]  # Extracting last 4 columns for y_test
# pred_test_velocities = df_result_pred_test.iloc[:, -4:].values[:n]  # Extracting last 4 columns for pred_test
#
# # Plotting each component separately
# for i in range(2):  # Loop over x, y components
#     plt.figure()
#
#     # Plot y_test and pred_test for the i-th component (initial and final velocities)
#     plt.scatter(results[:n], y_test_velocities[:, i], c='green', label=f'y_test v0 {["x", "y"][i]}')
#     plt.scatter(results[:n], pred_test_velocities[:, i], c='blue', label=f'pred_test v0 {["x", "y"][i]}')
#
#     # Set labels and title
#     plt.xlabel('Epochs')
#     plt.ylabel(f'Velocity Component {["X", "Y"][i]} [km/s]')
#     plt.title(f'Scatter Plot of Initial Velocity Component {["X", "Y"][i]} Predictions')
#     plt.legend()
#     plt.show()
#
# for i in range(2):  # Loop over x, y components
#     plt.figure()
#
#     # Plot y_test and pred_test for the i-th component (initial and final velocities)
#     plt.scatter(results[:n], y_test_velocities[:, 2 + i], c='green', label=f'y_test vf {["x", "y"][i]}')
#     plt.scatter(results[:n], pred_test_velocities[:, 2 + i], c='blue', label=f'pred_test  vf {["x", "y"][i]}')
#
#     # Set labels and title
#     plt.xlabel('Epochs')
#     plt.ylabel(f'Velocity Component {["X", "Y"][i]} [km/s]')
#     plt.title(f'Scatter Plot of Final Velocity Component {["X", "Y"][i]} Predictions')
#     plt.legend()
#     plt.show()
#
# # %% Analysis
# # y_test = y_test.cpu().numpy()
# # pred_test = pred_test.cpu().numpy()
# y_test = df_result_y_test.iloc[:, -4:].values  # Extracting last 4 columns for y_test
# pred_test = df_result_pred_test.iloc[:, -4:].values  # Extracting last 4 columns for pred_test
# error_test = (y_test - pred_test)
# error_test_percentage = (error_test / y_test) * 100
# filtered_error_test_percentage = np.where(error_test_percentage > 100, 100, error_test_percentage)
# filtered_error_test_percentage = np.where(filtered_error_test_percentage < -100, -100, filtered_error_test_percentage)
#
# # %%  Plot 1 (Prediction vs Actual plot)
# # Plotting actual vs predicted values
# n_bins = 1500
# fig, axes = plt.subplots(2, 2, figsize=(15, 15))
# (ax1, ax2), (ax3, ax4) = axes
# ax1.scatter(y_test[:n_bins, 0], pred_test[:n_bins, 0], color='blue', alpha=0.5, s=4)
# ax1.plot([min(y_test[:n_bins, 0]), max(y_test[:n_bins, 0])], [min(y_test[:n_bins, 0]), max(y_test[:n_bins, 0])],
#          color='green', linestyle='--')
# ax1.set_title("Predicted Vs Actual ")
# ax1.set_xlabel("Actual Initial Velocity Values X Direction [km/s] ")
# ax1.set_ylabel("Predicted Initial Velocity Values X Direction [km/s]")
#
# ax2.scatter(y_test[:n_bins, 1], pred_test[:n_bins, 1], color='blue', alpha=0.5, s=4)
# ax2.plot([min(y_test[:n_bins, 1]), max(y_test[:n_bins, 1])], [min(y_test[:n_bins, 1]), max(y_test[:n_bins, 1])],
#          color='green', linestyle='--')
# ax2.set_title("Predicted Vs Actual  ")
# ax2.set_xlabel("Actual Initial Values Y Direction [km/s]")
# ax2.set_ylabel("Predicted Initial Values Y Direction [km/s] ")
#
# ax3.scatter(y_test[:n_bins, 2], pred_test[:n_bins, 2], color='blue', alpha=0.5, s=4)
# ax3.plot([min(y_test[:n_bins, 2]), max(y_test[:n_bins, 2])], [min(y_test[:n_bins, 2]), max(y_test[:n_bins, 2])],
#          color='green', linestyle='--')
# ax3.set_title("Predicted Vs Actual ")
# ax3.set_xlabel("Actual Final Values X Direction [km/s]")
# ax3.set_ylabel("Predicted Final Values X Direction [km/s]")
#
# ax4.scatter(y_test[:n_bins, 3], pred_test[:n_bins, 3], color='blue', alpha=0.5, s=4)
# ax4.plot([min(y_test[:n_bins, 3]), max(y_test[:n_bins, 3])], [min(y_test[:n_bins, 3]), max(y_test[:n_bins, 3])],
#          color='green', linestyle='--')
# ax4.set_title("Predicted Vs Actual ")
# ax4.set_xlabel("Actual Final Values Y Direction [km/s]")
# ax4.set_ylabel("Predicted Final Values Y Direction [km/s]")
#
# # Add column titles
# fig.text(0.25, 0.95, 'X Direction', ha='center', fontsize=16)
# fig.text(0.75, 0.95, 'Y Direction', ha='center', fontsize=16)
#
# # Add row titles
# fig.text(0.06, 0.75, 'Initial Velocity', va='center', rotation='vertical', fontsize=16)
# fig.text(0.06, 0.25, 'Final Velocity', va='center', rotation='vertical', fontsize=16)
# # plt.savefig('Actual vs predicted.png', dpi=1200)
# plt.show()
#
# # %% Plot 2 (Residual plots)
# # Plotting actual vs predicted values
# fig, axes = plt.subplots(2, 2, figsize=(15, 15))
# (ax1, ax2), (ax3, ax4) = axes
# ax1.scatter(y_test[:n_bins, 0], error_test[:n_bins, 0], color='blue', alpha=0.5, s=4)
# ax1.axhline(y=0, color='green', linestyle='--')
# ax1.set_title("Residual Vs Actual")
# ax1.set_xlabel("Actual Initial Velocity X direction [km/s]")
# ax1.set_ylabel("Residues [km/s]")
#
# ax2.scatter(y_test[:n_bins, 1], error_test[:n_bins, 1], color='blue', alpha=0.5, s=4)
# ax2.axhline(y=0, color='green', linestyle='--')
# ax2.set_title("Residual Vs Actual")
# ax2.set_xlabel("Actual Initial Velocity Y direction [km/s]")
# ax2.set_ylabel("Residues [km/s]")
#
# ax3.scatter(y_test[:n_bins, 2], error_test[:n_bins, 2], color='blue', alpha=0.5, s=4)
# ax3.axhline(y=0, color='green', linestyle='--')
# ax3.set_title("Residual Vs Actual")
# ax3.set_xlabel("Actual Final Velocity X direction [km/s]")
# ax3.set_ylabel("Residues [km/s]")
#
# ax4.scatter(y_test[:n_bins, 3], error_test[:n_bins, 3], color='blue', alpha=0.5, s=4)
# ax4.axhline(y=0, color='green', linestyle='--')
# ax4.set_title("Residual Vs Final")
# ax4.set_xlabel("Actual Final Velocity Y direction [km/s]")
# ax4.set_ylabel("Residues [km/s]")
#
# # Add column titles
# fig.text(0.25, 0.95, 'X Direction', ha='center', fontsize=16)
# fig.text(0.75, 0.95, 'Y Direction', ha='center', fontsize=16)
#
# # Add row titles
# fig.text(0.06, 0.75, 'Initial Velocity', va='center', rotation='vertical', fontsize=16)
# fig.text(0.06, 0.25, 'Final Velocity', va='center', rotation='vertical', fontsize=16)
# # plt.savefig('Residues vs actual.png', dpi=1200)
# plt.show()
#
# # %% Plot 3 (Residue percentage)
# fig, axes = plt.subplots(2, 2, figsize=(15, 15))
# (ax1, ax2), (ax3, ax4) = axes
# ax1.scatter(y_test[:n_bins, 0], filtered_error_test_percentage[:n_bins, 0], color='blue', alpha=0.5, s=4)
# ax1.axhline(y=0, color='green', linestyle='--')
# ax1.set_title("Residual % Vs Actual Initial Velocity X")
# ax1.set_xlabel("Actual Initial Velocity X Direction [km/s]")
# ax1.set_ylabel("Residues (%)")
#
# ax2.scatter(y_test[:n_bins, 1], filtered_error_test_percentage[:n_bins, 1], color='blue', alpha=0.5, s=4)
# ax2.axhline(y=0, color='green', linestyle='--')
# ax2.set_title("Residual % Vs Actual Initial Velocity Y")
# ax2.set_xlabel("Actual Initial Velocity Y Direction [km/s]")
# ax2.set_ylabel("Residues (%)")
#
# ax3.scatter(y_test[:n_bins, 2], filtered_error_test_percentage[:n_bins, 2], color='blue', alpha=0.5, s=4)
# ax3.axhline(y=0, color='green', linestyle='--')
# ax3.set_title("Residual % Vs Actual Final Velocity X")
# ax3.set_xlabel("Actual Final Velocity X Direction [km/s]")
# ax3.set_ylabel("Residues (%)")
#
# ax4.scatter(y_test[:n_bins, 3], filtered_error_test_percentage[:n_bins, 3], color='blue', alpha=0.5, s=4)
# ax4.axhline(y=0, color='green', linestyle='--')
# ax4.set_title("Residual % Vs Actual Final Velocity Y")
# ax4.set_xlabel("Actual Final Velocity Y Direction [km/s]")
# ax4.set_ylabel("Residues (%)")
#
# # Add column titles
# fig.text(0.25, 0.95, 'X Direction', ha='center', fontsize=16)
# fig.text(0.75, 0.95, 'Y Direction', ha='center', fontsize=16)
#
# # Add row titles
# fig.text(0.06, 0.75, 'Initial Velocity', va='center', rotation='vertical', fontsize=16)
# fig.text(0.06, 0.25, 'Final Velocity', va='center', rotation='vertical', fontsize=16)
# # plt.savefig('Residues percent vs actual.png', dpi=1200)
# plt.show()

# %% Plot 3 (Bar Chart)
# width = 50
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#
# ax1.bar(range(len(error_test[:n_bins, 0])), error_test[:n_bins, 0], width=width, color='skyblue')
# ax1.set_title("Absolute error_test X ")
# ax1.set_xlabel("Data point")
# ax1.set_ylabel("Absolute error_test ")
#
# ax2.bar(range(len(error_test[:n_bins, 1])), error_test[:n_bins, 1], width=width, color='skyblue')
# ax2.set_title("Absolute error_test Y ")
# ax2.set_xlabel("Data point")
# ax2.set_ylabel("Absolute error_test ")
#
# plt.tight_layout()
# plt.show()

# %% Plot 4 (QQ plot )
# plt.figure(figsize=(10, 6))
# stats.probplot(error_test[:n_bins, 0], dist="norm", plot=plt)
# plt.title('Q-Q Plot')
# plt.grid(True)
# plt.show()
#
# # %% Plot 5 Prediction with confidence intervals
# sigma = 1.96
# n = 200
# mean_residuals_x = np.mean(error_test[:n_bins, 0])
# std_residuals_x = np.std(error_test[:n_bins, 0])
# mean_residuals_y = np.mean(error_test[:n_bins, 1])
# std_residuals_y = np.std(error_test[:n_bins, 1])
#
# lower_bound_x = pred_test[:n_bins, 0] - sigma * std_residuals_x
# upper_bound_x = pred_test[:n_bins, 0] + sigma * std_residuals_x
# lower_bound_y = pred_test[:n_bins, 1] - sigma * std_residuals_y
# upper_bound_y = pred_test[:n_bins, 1] + sigma * std_residuals_y
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# ax1.scatter(y_test[:n_bins, 0], pred_test[:n_bins, 0], alpha=0.5, label='Predictions')
# ax1.fill_between(y_test[:n_bins, 0], lower_bound_x, upper_bound_x, color='red', alpha=0.2,
#                  label=f'{sigma * 100:.0f}% Prediction Interval')
# ax1.set_xlabel('Actual')
# ax1.set_ylabel('Predicted')
# ax1.set_title('Prediction Intervals x')
# ax1.legend()
#
# ax2.scatter(y_test[:n_bins, 1], pred_test[:n_bins, 1], alpha=0.5, label='Predictions')
# ax2.fill_between(y_test[:n_bins, 1], lower_bound_y, upper_bound_y, color='red', alpha=0.2,
#                  label=f'{sigma * 100:.0f}% Prediction Interval')
# ax2.set_xlabel('Actual')
# ax2.set_ylabel('Predicted')
# ax2.set_title('Prediction Intervals y')
# ax2.legend()
#
# plt.tight_layout()
# plt.show()

# %% Plot 6 Feature importance (ABS of the weight)
# state_dict = model_01.state_dict()
# # Extracting weights for the first layer (input to first hidden layer)
# # Extract weights for the first four nodes in the first layer
# weights_first_layer_x = abs(state_dict['layer1.weight'][:, 0].numpy())
# weights_first_layer_y = abs(state_dict['layer1.weight'][:, 1].numpy())
# weights_first_layer_z = abs(state_dict['layer1.weight'][:, 2].numpy())
# weights_first_layer_t = abs(state_dict['layer1.weight'][:, 3].numpy())
#
# # Number of subplots (rows, columns)
# fig, axs = plt.subplots(2, 2, figsize=(20, 8))
#
# # Plotting each weight series in its own subplot
# axs[0, 0].bar(range(len(weights_first_layer_x)), weights_first_layer_x)
# axs[0, 0].set_xlabel('Feature Index')
# axs[0, 0].set_ylabel('Feature Importance')
# axs[0, 0].set_title('Feature Importance Plot rel X')
#
# axs[0, 1].bar(range(len(weights_first_layer_y)), weights_first_layer_y)
# axs[0, 1].set_xlabel('Feature Index')
# axs[0, 1].set_ylabel('Feature Importance')
# axs[0, 1].set_title('Feature Importance Plot rel Y')
#
# axs[1, 0].bar(range(len(weights_first_layer_z)), weights_first_layer_z)
# axs[1, 0].set_xlabel('Feature Index')
# axs[1, 0].set_ylabel('Feature Importance')
# axs[1, 0].set_title('Feature Importance Plot rel Z')
#
# axs[1, 1].bar(range(len(weights_first_layer_t)), weights_first_layer_t)
# axs[1, 1].set_xlabel('Feature Index')
# axs[1, 1].set_ylabel('Feature Importance')
# axs[1, 1].set_title('Feature Importance Plot tof')
#
# # Adjust layout to prevent overlap
# plt.tight_layout()
#
# # Show the plot
# plt.show(block=True)
