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
import matplotlib as mpl
import numpy as np
from scipy import stats
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Display plots in separate window
# mpl.use('macosx')

DATA_PATH = Path("data/processed")
DATA_SET = "10K_10"
DATA_NAME = "transfer_data_" + DATA_SET + ".csv"

# create saved_models directory
MODEL_PATH = Path("src/models/saved_models")
today = date.today()
MODEL_NAME = str(today) + "_" + DATA_SET + ".pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

df = pd.read_csv(DATA_PATH / DATA_NAME)

# Initialize the scaler
scaler = StandardScaler()  # or MaxAbsScaler()

# Fit and transform the scaler to each column separately
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# pd.set_option('display.max_columns', None)
# logging.info("\n" + str(df.describe(include='all')))

df_Features = df.iloc[:, :3]
df_Labels = df.iloc[:, -4:]

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

# %% Setting up training params
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %% Construct a model class that subclasses nn.Module
torch.manual_seed(42)

# create an instance of the model
model_01 = DNNClassifier()
model_01.state_dict()

# Your model and processed setup
model_01.to(device)
x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)

# %% Train model
# Setting up a loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params=model_01.parameters(), lr=0.001, momentum=0.75)  # lr = learning rate

model_01_trainer = ModelTrainer(model_01, loss_fn, optimizer, DATA_NAME)
epochs_array = model_01_trainer.train(300, x_train, x_test, y_train, y_test, False)
model_01_trainer.plot_training_curves()


# %% Saving model
torch.save(obj=model_01.state_dict(), f=MODEL_SAVE_PATH)
logging.info("Model has been written to " + f"{MODEL_SAVE_PATH}")
plt.show()


# %% Make estimates
model_01_trainer.model.eval()  # turns off difference setting sin th emodel not needed evaluating/testing
with torch.inference_mode():
    y_pred = model_01_trainer.model(x_test)  # Perform inference on GPU
    # y_pred = torch.round(torch.sigmoid(y_pred))


# %% Unscale the values
def unscale(scaled_value):
    unscaled_value = scaler.inverse_transform(scaled_value)
    return pd.DataFrame(unscaled_value)


df_result_y_test_scaled = pd.concat([pd.DataFrame(x_test.cpu().numpy()), pd.DataFrame(y_test.cpu().numpy())], ignore_index=True,
                                    axis='columns')
df_result_y_pred_scaled = pd.concat([pd.DataFrame(x_test.cpu().numpy()), pd.DataFrame(y_pred.cpu().numpy())], ignore_index=True,
                                    axis='columns')

# Apply to unscale function to each column of inputs arrays
df_result_y_test = unscale(df_result_y_test_scaled)
df_result_y_pred = unscale(df_result_y_pred_scaled)

# %% Plotting
n = 10  # Number of points to plot (adjust as needed)

# Extracting data for plotting (assuming df_result_y_test and df_result_y_pred contain your data)
y_test_velocities = df_result_y_test.iloc[:, -4:].values[:n]  # Extracting last 4 columns for y_test
y_pred_velocities = df_result_y_pred.iloc[:, -4:].values[:n]  # Extracting last 4 columns for y_pred

# Plotting each component separately
for i in range(2):  # Loop over x, y components
    plt.figure()

    # Plot y_test and y_pred for the i-th component (initial and final velocities)
    plt.scatter(epochs_array[:n], y_test_velocities[:, i], c='green', label=f'y_test v0 {["x", "y"][i]}')
    plt.scatter(epochs_array[:n], y_pred_velocities[:, i], c='blue', label=f'y_pred v0 {["x", "y"][i]}')

    # Set labels and title
    plt.xlabel('Epochs')
    plt.ylabel(f'Velocity Component {["X", "Y"][i]} [km/s]')
    plt.title(f'Scatter Plot of Initial Velocity Component {["X", "Y"][i]} Predictions')
    plt.legend()
    plt.show()

for i in range(2):  # Loop over x, y components
    plt.figure()

    # Plot y_test and y_pred for the i-th component (initial and final velocities)
    plt.scatter(epochs_array[:n], y_test_velocities[:, 2+i], c='green', label=f'y_test vf {["x", "y"][i]}')
    plt.scatter(epochs_array[:n], y_pred_velocities[:, 2+i], c='blue', label=f'y_pred  vf {["x", "y"][i]}')

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
errors = abs(y_test - y_pred)
error_percentage = (errors / y_test) * 100
filtered_error_percentage = np.where(abs(error_percentage) > 500, 100, error_percentage)

# %%  Plot 1 (Prediction vs Actual plot)
# Plotting actual vs predicted values
n_bins = 1500
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
(ax1, ax2), (ax3, ax4) = axes
ax1.scatter(y_test[:n_bins, 0], y_pred[:n_bins, 0], color='blue', alpha=0.5)
ax1.plot([min(y_test[:n_bins, 0]), max(y_test[:n_bins, 0])], [min(y_test[:n_bins, 0]), max(y_test[:n_bins, 0])],
         color='red', linestyle='--')
ax1.set_title("Actual Vs Predicted Initial x Values  ")
ax1.set_xlabel("Actual Values ")
ax1.set_ylabel("Predicted Values ")

ax2.scatter(y_test[:n_bins, 1], y_pred[:n_bins, 1], color='blue', alpha=0.5)
ax2.plot([min(y_test[:n_bins, 1]), max(y_test[:n_bins, 1])], [min(y_test[:n_bins, 1]), max(y_test[:n_bins, 1])],
         color='red', linestyle='--')
ax2.set_title("Actual Vs Predicted Initial y values ")
ax2.set_xlabel("Actual Values ")
ax2.set_ylabel("Predicted Values ")

ax3.scatter(y_test[:n_bins, 2], y_pred[:n_bins, 2], color='blue', alpha=0.5)
ax3.plot([min(y_test[:n_bins, 2]), max(y_test[:n_bins, 2])], [min(y_test[:n_bins, 2]), max(y_test[:n_bins, 2])],
         color='red', linestyle='--')
ax3.set_title("Actual Vs Predicted Final x Values  ")
ax3.set_xlabel("Actual Values ")
ax3.set_ylabel("Predicted Values ")

ax4.scatter(y_test[:n_bins, 3], y_pred[:n_bins, 3], color='blue', alpha=0.5)
ax4.plot([min(y_test[:n_bins, 3]), max(y_test[:n_bins, 3])], [min(y_test[:n_bins, 3]), max(y_test[:n_bins, 3])],
         color='red', linestyle='--')
ax4.set_title("Actual Vs Predicted Final y values ")
ax4.set_xlabel("Actual Values ")
ax4.set_ylabel("Predicted Values ")

plt.tight_layout()
plt.show()

# %% Plot 2 (Residual plots)
# Plotting actual vs predicted values
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
(ax1, ax2), (ax3, ax4) = axes
ax1.scatter(y_pred[:n_bins, 0], errors[:n_bins, 0], color='blue', alpha=0.5)
ax1.axhline(y=0, color='red', linestyle='--')
ax1.set_title("Residual Plot Initial x ")
ax1.set_xlabel("Predicted Values")
ax1.set_ylabel("Residues ")

ax2.scatter(y_pred[:n_bins, 1], errors[:n_bins, 1], color='blue', alpha=0.5)
ax2.axhline(y=0, color='red', linestyle='--')
ax2.set_title("Residual Plot Initial y ")
ax2.set_xlabel("Predicted Values")
ax2.set_ylabel("Residues ")

ax3.scatter(y_pred[:n_bins, 2], errors[:n_bins, 2], color='blue', alpha=0.5)
ax3.axhline(y=0, color='red', linestyle='--')
ax3.set_title("Residual Plot Final x ")
ax3.set_xlabel("Predicted Values")
ax3.set_ylabel("Residues ")

ax4.scatter(y_pred[:n_bins, 3], errors[:n_bins, 3], color='blue', alpha=0.5)
ax4.axhline(y=0, color='red', linestyle='--')
ax4.set_title("Residual Plot Final y ")
ax4.set_xlabel("Predicted Values")
ax4.set_ylabel("Residues ")

plt.tight_layout()
plt.show()

# %% Plot 3 (Residue percentage)
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
(ax1, ax2), (ax3, ax4) = axes
ax1.scatter(y_test[:n_bins, 0], filtered_error_percentage[:n_bins, 0], color='blue', alpha=0.5)
ax1.axhline(y=0, color='red', linestyle='--')
ax1.set_title("Residual Plot Initial x as percentage")
ax1.set_xlabel("Predicted Values")
ax1.set_ylabel("Residues %")

ax2.scatter(y_test[:n_bins, 1], filtered_error_percentage[:n_bins, 1], color='blue', alpha=0.5)
ax2.axhline(y=0, color='red', linestyle='--')
ax2.set_title("Residual Plot Initial y as percentage")
ax2.set_xlabel("Predicted Values")
ax2.set_ylabel("Residues %")

ax3.scatter(y_test[:n_bins, 2], filtered_error_percentage[:n_bins, 2], color='blue', alpha=0.5)
ax3.axhline(y=0, color='red', linestyle='--')
ax3.set_title("Residual Plot Final x as percentage")
ax3.set_xlabel("Predicted Values")
ax3.set_ylabel("Residues %")

ax4.scatter(y_test[:n_bins, 3], filtered_error_percentage[:n_bins, 3], color='blue', alpha=0.5)
ax4.axhline(y=0, color='red', linestyle='--')
ax4.set_title("Residual Plot Final y as percentage")
ax4.set_xlabel("Predicted Values")
ax4.set_ylabel("Residues %")

plt.tight_layout()
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