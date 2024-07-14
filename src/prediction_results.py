# %%
from src.models.DNNClassifier import DNNClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn as nn
import torch.nn.functional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from scipy import stats
import logging

# %% Initial Setup
# Parameters
DATA_SET = "10K_01"
MODEL_NAME = "2024-07-10_" + DATA_SET + ".pth"
TEST_SIZE = 0.2
INPUT_SIZE = 3
OUTPUT_SIZE = 4
NUM_LAYERS = 3
NUM_NEURONS = 64
ACTIVATION = nn.SELU

# Display plots in separate window
# mpl.use('macosx')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %% Data loading and scaling
DATA_PATH = Path("data/processed")
DATA_NAME = "transfer_data_" + DATA_SET + ".csv"

MODEL_PATH = Path("src/models/saved_models")
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Check if CUDA (NVIDIA GPU) is available
cuda_available = torch.cuda.is_available()
logging.info("CUDA (NVIDIA GPU) available: " + str(cuda_available))

# Move your model and processed tensors to the GPU (if available)
device = torch.device("cuda" if cuda_available else "mps")

model_01 = DNNClassifier(INPUT_SIZE, OUTPUT_SIZE, NUM_LAYERS, NUM_NEURONS, ACTIVATION)
model_01.to(device)
model_01.load_state_dict(torch.load(MODEL_SAVE_PATH))

df = pd.read_csv(DATA_PATH / DATA_NAME)
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)

# %% Make estimates
model_01.eval()
with torch.inference_mode():
    y_pred = model_01(x_test)


# %% Unscale the values
def unscale(scaled_value):
    unscaled_value = scaler.inverse_transform(scaled_value)
    return pd.DataFrame(unscaled_value)


df_result_y_test_scaled = pd.concat([pd.DataFrame(x_test.cpu().numpy()), pd.DataFrame(y_test.cpu().numpy())],
                                    ignore_index=True,
                                    axis='columns')
df_result_y_pred_scaled = pd.concat([pd.DataFrame(x_test.cpu().numpy()), pd.DataFrame(y_pred.cpu().numpy())],
                                    ignore_index=True,
                                    axis='columns')

# Apply to unscale function to each column of inputs arrays
df_result_y_test = unscale(df_result_y_test_scaled)
df_result_y_pred = unscale(df_result_y_pred_scaled)

# %% Plotting
n = 10  # Number of points to plot (adjust as needed)
results = range(n)

# Extracting data for plotting (assuming df_result_y_test and df_result_y_pred contain your data)
y_test_velocities = df_result_y_test.iloc[:, -4:].values[:n]  # Extracting last 4 columns for y_test
y_pred_velocities = df_result_y_pred.iloc[:, -4:].values[:n]  # Extracting last 4 columns for y_pred

# Plotting each component separately
for i in range(2):  # Loop over x, y components
    plt.figure()

    # Plot y_test and y_pred for the i-th component (initial and final velocities)
    plt.scatter(results[:n], y_test_velocities[:, i], c='green', label=f'y_test v0 {["x", "y"][i]}')
    plt.scatter(results[:n], y_pred_velocities[:, i], c='blue', label=f'y_pred v0 {["x", "y"][i]}')

    # Set labels and title
    plt.xlabel('Epochs')
    plt.ylabel(f'Velocity Component {["X", "Y"][i]} [km/s]')
    plt.title(f'Scatter Plot of Initial Velocity Component {["X", "Y"][i]} Predictions')
    plt.legend()
    plt.show()

for i in range(2):  # Loop over x, y components
    plt.figure()

    # Plot y_test and y_pred for the i-th component (initial and final velocities)
    plt.scatter(results[:n], y_test_velocities[:, 2 + i], c='green', label=f'y_test vf {["x", "y"][i]}')
    plt.scatter(results[:n], y_pred_velocities[:, 2 + i], c='blue', label=f'y_pred  vf {["x", "y"][i]}')

    # Set labels and title
    plt.xlabel('Epochs')
    plt.ylabel(f'Velocity Component {["X", "Y"][i]} [km/s]')
    plt.title(f'Scatter Plot of Final Velocity Component {["X", "Y"][i]} Predictions')
    plt.legend()
    plt.show()

# %% Analysis
# y_test = y_test.cpu().numpy()
# y_pred = y_pred.cpu().numpy()
y_test = df_result_y_test.iloc[:, -4:].values  # Extracting last 4 columns for y_test
y_pred = df_result_y_pred.iloc[:, -4:].values  # Extracting last 4 columns for y_pred
errors = (y_test - y_pred)
error_percentage = (errors / y_test) * 100
filtered_error_percentage = np.where(error_percentage > 100, 100, error_percentage)
filtered_error_percentage = np.where(filtered_error_percentage < -100, -100, filtered_error_percentage)

# %%  Plot 1 (Prediction vs Actual plot)
# Plotting actual vs predicted values
n_bins = 1500
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
(ax1, ax2), (ax3, ax4) = axes
ax1.scatter(y_test[:n_bins, 0], y_pred[:n_bins, 0], color='blue', alpha=0.5, s=4)
ax1.plot([min(y_test[:n_bins, 0]), max(y_test[:n_bins, 0])], [min(y_test[:n_bins, 0]), max(y_test[:n_bins, 0])],
         color='green', linestyle='--')
ax1.set_title("Predicted Vs Actual ")
ax1.set_xlabel("Actual Initial Velocity Values X Direction [km/s] ")
ax1.set_ylabel("Predicted Initial Velocity Values X Direction [km/s]")

ax2.scatter(y_test[:n_bins, 1], y_pred[:n_bins, 1], color='blue', alpha=0.5, s=4)
ax2.plot([min(y_test[:n_bins, 1]), max(y_test[:n_bins, 1])], [min(y_test[:n_bins, 1]), max(y_test[:n_bins, 1])],
         color='green', linestyle='--')
ax2.set_title("Predicted Vs Actual  ")
ax2.set_xlabel("Actual Initial Values Y Direction [km/s]")
ax2.set_ylabel("Predicted Initial Values Y Direction [km/s] ")

ax3.scatter(y_test[:n_bins, 2], y_pred[:n_bins, 2], color='blue', alpha=0.5, s=4)
ax3.plot([min(y_test[:n_bins, 2]), max(y_test[:n_bins, 2])], [min(y_test[:n_bins, 2]), max(y_test[:n_bins, 2])],
         color='green', linestyle='--')
ax3.set_title("Predicted Vs Actual ")
ax3.set_xlabel("Actual Final Values X Direction [km/s]")
ax3.set_ylabel("Predicted Final Values X Direction [km/s]")

ax4.scatter(y_test[:n_bins, 3], y_pred[:n_bins, 3], color='blue', alpha=0.5, s=4)
ax4.plot([min(y_test[:n_bins, 3]), max(y_test[:n_bins, 3])], [min(y_test[:n_bins, 3]), max(y_test[:n_bins, 3])],
         color='green', linestyle='--')
ax4.set_title("Predicted Vs Actual ")
ax4.set_xlabel("Actual Final Values Y Direction [km/s]")
ax4.set_ylabel("Predicted Final Values Y Direction [km/s]")

# Add column titles
fig.text(0.25, 0.95, 'X Direction', ha='center', fontsize=16)
fig.text(0.75, 0.95, 'Y Direction', ha='center', fontsize=16)

# Add row titles
fig.text(0.06, 0.75, 'Initial Velocity', va='center', rotation='vertical', fontsize=16)
fig.text(0.06, 0.25, 'Final Velocity', va='center', rotation='vertical', fontsize=16)
# plt.savefig('Actual vs predicted.png', dpi=1200)
plt.show()

# %% Plot 2 (Residual plots)
# Plotting actual vs predicted values
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
(ax1, ax2), (ax3, ax4) = axes
ax1.scatter(y_test[:n_bins, 0], errors[:n_bins, 0], color='blue', alpha=0.5, s=4)
ax1.axhline(y=0, color='green', linestyle='--')
ax1.set_title("Residual Vs Actual")
ax1.set_xlabel("Actual Initial Velocity X direction [km/s]")
ax1.set_ylabel("Residues [km/s]")

ax2.scatter(y_test[:n_bins, 1], errors[:n_bins, 1], color='blue', alpha=0.5, s=4)
ax2.axhline(y=0, color='green', linestyle='--')
ax2.set_title("Residual Vs Actual")
ax2.set_xlabel("Actual Initial Velocity Y direction [km/s]")
ax2.set_ylabel("Residues [km/s]")

ax3.scatter(y_test[:n_bins, 2], errors[:n_bins, 2], color='blue', alpha=0.5, s=4)
ax3.axhline(y=0, color='green', linestyle='--')
ax3.set_title("Residual Vs Actual")
ax3.set_xlabel("Actual Final Velocity X direction [km/s]")
ax3.set_ylabel("Residues [km/s]")

ax4.scatter(y_test[:n_bins, 3], errors[:n_bins, 3], color='blue', alpha=0.5, s=4)
ax4.axhline(y=0, color='green', linestyle='--')
ax4.set_title("Residual Vs Final")
ax4.set_xlabel("Actual Final Velocity Y direction [km/s]")
ax4.set_ylabel("Residues [km/s]")

# Add column titles
fig.text(0.25, 0.95, 'X Direction', ha='center', fontsize=16)
fig.text(0.75, 0.95, 'Y Direction', ha='center', fontsize=16)

# Add row titles
fig.text(0.06, 0.75, 'Initial Velocity', va='center', rotation='vertical', fontsize=16)
fig.text(0.06, 0.25, 'Final Velocity', va='center', rotation='vertical', fontsize=16)
# plt.savefig('Residues vs actual.png', dpi=1200)
plt.show()

# %% Plot 3 (Residue percentage)
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
(ax1, ax2), (ax3, ax4) = axes
ax1.scatter(y_test[:n_bins, 0], filtered_error_percentage[:n_bins, 0], color='blue', alpha=0.5, s=4)
ax1.axhline(y=0, color='green', linestyle='--')
ax1.set_title("Residual % Vs Actual Initial Velocity X")
ax1.set_xlabel("Actual Initial Velocity X Direction [km/s]")
ax1.set_ylabel("Residues (%)")

ax2.scatter(y_test[:n_bins, 1], filtered_error_percentage[:n_bins, 1], color='blue', alpha=0.5, s=4)
ax2.axhline(y=0, color='green', linestyle='--')
ax2.set_title("Residual % Vs Actual Initial Velocity Y")
ax2.set_xlabel("Actual Initial Velocity Y Direction [km/s]")
ax2.set_ylabel("Residues (%)")

ax3.scatter(y_test[:n_bins, 2], filtered_error_percentage[:n_bins, 2], color='blue', alpha=0.5, s=4)
ax3.axhline(y=0, color='green', linestyle='--')
ax3.set_title("Residual % Vs Actual Final Velocity X")
ax3.set_xlabel("Actual Final Velocity X Direction [km/s]")
ax3.set_ylabel("Residues (%)")

ax4.scatter(y_test[:n_bins, 3], filtered_error_percentage[:n_bins, 3], color='blue', alpha=0.5, s=4)
ax4.axhline(y=0, color='green', linestyle='--')
ax4.set_title("Residual % Vs Actual Final Velocity Y")
ax4.set_xlabel("Actual Final Velocity Y Direction [km/s]")
ax4.set_ylabel("Residues (%)")

# Add column titles
fig.text(0.25, 0.95, 'X Direction', ha='center', fontsize=16)
fig.text(0.75, 0.95, 'Y Direction', ha='center', fontsize=16)

# Add row titles
fig.text(0.06, 0.75, 'Initial Velocity', va='center', rotation='vertical', fontsize=16)
fig.text(0.06, 0.25, 'Final Velocity', va='center', rotation='vertical', fontsize=16)
# plt.savefig('Residues percent vs actual.png', dpi=1200)
plt.show()

# %% Plot 3 (Bar Chart)
# width = 50
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#
# ax1.bar(range(len(errors[:n_bins, 0])), errors[:n_bins, 0], width=width, color='skyblue')
# ax1.set_title("Absolute errors X ")
# ax1.set_xlabel("Data point")
# ax1.set_ylabel("Absolute errors ")
#
# ax2.bar(range(len(errors[:n_bins, 1])), errors[:n_bins, 1], width=width, color='skyblue')
# ax2.set_title("Absolute errors Y ")
# ax2.set_xlabel("Data point")
# ax2.set_ylabel("Absolute errors ")
#
# plt.tight_layout()
# plt.show()

# %% Plot 4 (QQ plot )
# plt.figure(figsize=(10, 6))
# stats.probplot(errors[:n_bins, 0], dist="norm", plot=plt)
# plt.title('Q-Q Plot')
# plt.grid(True)
# plt.show()
#
# # %% Plot 5 Prediction with confidence intervals
# sigma = 1.96
# n = 200
# mean_residuals_x = np.mean(errors[:n_bins, 0])
# std_residuals_x = np.std(errors[:n_bins, 0])
# mean_residuals_y = np.mean(errors[:n_bins, 1])
# std_residuals_y = np.std(errors[:n_bins, 1])
#
# lower_bound_x = y_pred[:n_bins, 0] - sigma * std_residuals_x
# upper_bound_x = y_pred[:n_bins, 0] + sigma * std_residuals_x
# lower_bound_y = y_pred[:n_bins, 1] - sigma * std_residuals_y
# upper_bound_y = y_pred[:n_bins, 1] + sigma * std_residuals_y
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# ax1.scatter(y_test[:n_bins, 0], y_pred[:n_bins, 0], alpha=0.5, label='Predictions')
# ax1.fill_between(y_test[:n_bins, 0], lower_bound_x, upper_bound_x, color='red', alpha=0.2,
#                  label=f'{sigma * 100:.0f}% Prediction Interval')
# ax1.set_xlabel('Actual')
# ax1.set_ylabel('Predicted')
# ax1.set_title('Prediction Intervals x')
# ax1.legend()
#
# ax2.scatter(y_test[:n_bins, 1], y_pred[:n_bins, 1], alpha=0.5, label='Predictions')
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
