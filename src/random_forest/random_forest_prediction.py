import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'data/low_thrust/datasets/processed/new_transfer_statistics_500.csv'
data = pd.read_csv(file_path)

output_columns = ['m0_maximum [kg]', 'm1_maximum [kg]']

# Separate features (first 10 columns) and target (last two columns: m_0 and m_1)
x = data.iloc[:, :10]  # First 10 columns as features
y = data.iloc[:, -2:]  # Last two columns as targets (m_0 and m_1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Load the saved model from the file
best_rf_model = joblib.load('src/models/saved_models/random_forest_model.pkl')

# Make predictions using the best model
pred_train = best_rf_model.predict(x_train)
pred_test = best_rf_model.predict(x_test)

# Convert to NumPy arrays and get only the first n_points
y_train = y_train.to_numpy()[:]  # First n_points of y_train
y_test = y_test.to_numpy()[:]    # First n_points of y_test

pred_train = pred_train[:]
pred_test = pred_test[:]

# Calculate the errors for the first n_points only
error_test = (pred_test[:] - y_test[:])  # Errors for test set
error_train = (pred_train[:] - y_train[:])  # Errors for train set

error_train_percentage = (error_train / y_train) * 100
filtered_error_train_percentage = np.where(error_train_percentage > 100, 100, error_train_percentage)
filtered_error_train_percentage = np.where(filtered_error_train_percentage < -100, -100, filtered_error_train_percentage)
filtered_error_train_percentage = filtered_error_train_percentage[:]

error_test_percentage = (error_test / y_test) * 100
filtered_error_test_percentage = np.where(error_test_percentage > 100, 100, error_test_percentage)
filtered_error_test_percentage = np.where(filtered_error_test_percentage < -100, -100, filtered_error_test_percentage)
filtered_error_test_percentage = filtered_error_test_percentage[:]

# Plot 1 (Prediction vs Actual plot)
fig, axes = plt.subplots(2, len(output_columns), figsize=(15, 7.5))  # 2 rows: Train and Test; len(output_cols) cols
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

# Plot 2 (Residual plot with Mean, Max, and Min Errors)
fig, axes = plt.subplots(2, len(output_columns), figsize=(15, 10))  # 2 rows: Train and Test; len(output_cols) cols
if len(output_columns) == 1:
    axes = [axes]

# Loop through each output column and plot residuals for both training and test data
for i, col_name in enumerate(output_columns):
    # Calculate Mean Error (ME), Mean Absolute Error (MAE), Max and Min Absolute Errors for training
    me_train = np.mean(error_train[:, i])
    mae_train = np.mean(np.abs(error_train[:, i]))
    max_abs_train = np.max(np.abs(error_train[:, i]))
    min_abs_train = np.min(np.abs(error_train[:, i]))

    # Calculate Mean Error (ME), Mean Absolute Error (MAE), Max and Min Absolute Errors for test
    me_test = np.mean(error_test[:, i])
    mae_test = np.mean(np.abs(error_test[:, i]))
    max_abs_test = np.max(np.abs(error_test[:, i]))
    min_abs_test = np.min(np.abs(error_test[:, i]))

    # Training residual plot (top row)
    ax_train = axes[0, i]
    ax_train.scatter(y_train[:, i], error_train[:, i], color='blue', alpha=0.5, s=4)
    ax_train.axhline(y=0, color='green', linestyle='--', label='Target residual')
    ax_train.axhline(y=me_train, color='blue', linestyle='--', label=f'Mean residual (ME): {me_train:.2f} kg')
    ax_train.axhline(y=mae_train, color='orange', linestyle='--', label=f'Mean abs residual (MAE): {mae_train:.2f} kg')
    ax_train.axhline(y=max_abs_train, color='red', linestyle='--', label=f'Max abs residual: {max_abs_train:.2f} kg')
    ax_train.axhline(y=-max_abs_train, color='red', linestyle='--')
    ax_train.axhline(y=min_abs_train, color='purple', linestyle='--', label=f'Min abs residual: {min_abs_train:.2f} kg')
    ax_train.set_title(f"Residual vs Actual {col_name} (Training)")
    ax_train.set_xlabel(f"Actual {col_name} [kg]")
    ax_train.set_ylabel("Residuals [kg]")
    ax_train.legend()

    # Test residual plot (bottom row)
    ax_test = axes[1, i]
    ax_test.scatter(y_test[:, i], error_test[:, i], color='blue', alpha=0.5, s=4)
    ax_test.axhline(y=0, color='green', linestyle='--', label='Target residual')
    ax_test.axhline(y=me_test, color='blue', linestyle='--', label=f'Mean residual (ME): {me_test:.2f} kg')
    ax_test.axhline(y=mae_test, color='orange', linestyle='--', label=f'Mean abs residual (MAE): {mae_test:.2f} kg')
    ax_test.axhline(y=max_abs_test, color='red', linestyle='--', label=f'Max abs residual: {max_abs_test:.2f} kg')
    ax_test.axhline(y=-max_abs_test, color='red', linestyle='--')
    ax_test.axhline(y=min_abs_test, color='purple', linestyle='--', label=f'Min abs residual: {min_abs_test:.2f} kg')
    ax_test.set_title(f"Residual vs Actual {col_name} (Test)")
    ax_test.set_xlabel(f"Actual {col_name} [kg]")
    ax_test.set_ylabel("Residuals [kg]")
    ax_test.legend()

# Adjust layout for better appearance
plt.tight_layout()
plt.show()

# Plot 3 (Residual percentage plot with Mean, Max, and Min Errors)
fig, axes = plt.subplots(2, len(output_columns), figsize=(15, 10))  # 2 rows, num_columns columns of subplots

if len(output_columns) == 1:
    axes = [axes]

# Loop through each output column and plot residual percentages for both training and test
for i, col_name in enumerate(output_columns):
    # Calculate mean, max, min absolute residual percentages for training
    me_train = np.mean(filtered_error_train_percentage[:, i])
    mae_train = np.mean(np.abs(filtered_error_train_percentage[:, i]))
    max_abs_train = np.max(np.abs(filtered_error_train_percentage[:, i]))
    min_abs_train = np.min(np.abs(filtered_error_train_percentage[:, i]))

    # Calculate mean, max, min absolute residual percentages for test
    me_test = np.mean(filtered_error_test_percentage[:, i])
    mae_test = np.mean(np.abs(filtered_error_test_percentage[:, i]))
    max_abs_test = np.max(np.abs(filtered_error_test_percentage[:, i]))
    min_abs_test = np.min(np.abs(filtered_error_test_percentage[:, i]))

    # Training residual percentage plot (top row)
    ax_train = axes[0, i]
    ax_train.scatter(y_train[:, i], filtered_error_train_percentage[:, i], color='blue', alpha=0.5, s=4)
    ax_train.axhline(y=0, color='green', linestyle='--', label='Target residual')
    ax_train.axhline(y=me_train, color='blue', linestyle='--', label=f'Mean residual: {me_train:.2f}%')
    ax_train.axhline(y=mae_train, color='orange', linestyle='--', label=f'Mean abs residual: {mae_train:.2f}%')
    ax_train.axhline(y=max_abs_train, color='red', linestyle='--', label=f'Max abs residual: {max_abs_train:.2f}%')
    ax_train.axhline(y=-max_abs_train, color='red', linestyle='--')
    ax_train.axhline(y=min_abs_train, color='purple', linestyle='--', label=f'Min abs residual: {min_abs_train:.2f}%')
    ax_train.set_title(f"Residual % vs Actual {col_name} (Training)")
    ax_train.set_xlabel(f"Actual {col_name} [kg]")
    ax_train.set_ylabel("Residues (%)")
    ax_train.legend()

    # Test residual percentage plot (bottom row)
    ax_test = axes[1, i]
    ax_test.scatter(y_test[:, i], filtered_error_test_percentage[:, i], color='blue', alpha=0.5, s=4)
    ax_test.axhline(y=0, color='green', linestyle='--', label='Target residual')
    ax_test.axhline(y=me_test, color='blue', linestyle='--', label=f'Mean residual: {me_test:.2f}%')
    ax_test.axhline(y=mae_test, color='orange', linestyle='--', label=f'Mean abs residual: {mae_test:.2f}%')
    ax_test.axhline(y=max_abs_test, color='red', linestyle='--', label=f'Max abs residual: {max_abs_test:.2f}%')
    ax_test.axhline(y=-max_abs_test, color='red', linestyle='--')
    ax_test.axhline(y=min_abs_test, color='purple', linestyle='--', label=f'Min abs residual: {min_abs_test:.2f}%')
    ax_test.set_title(f"Residual % vs Actual {col_name} (Test)")
    ax_test.set_xlabel(f"Actual {col_name} [kg]")
    ax_test.set_ylabel("Residues (%)")
    ax_test.legend()

# Adjust layout for better appearance
plt.tight_layout()
plt.show()

print("Plots with Mean, Max, and Min errors completed!")
