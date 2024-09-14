import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'data/low_thrust/low_thrust_segment_statistics.csv'
data = pd.read_csv(file_path)

# Separate features (first 10 columns) and target (last two columns: m_0 and m_1)
X = data.iloc[:, :-2]  # First 10 columns as features
y = data.iloc[:, -2:]  # Last two columns as targets (m_0 and m_1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [300],
    'max_depth': [30],
    'min_samples_split': [2],
    'min_samples_leaf': [2]
}

# Initialize the Random Forest Regressor for multi-output regression
rf_model = RandomForestRegressor(random_state=42)

# Set up the GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best model from the grid search
best_rf_model = grid_search.best_estimator_

# Make predictions using the best model

pred_train = best_rf_model.predict(X_train)
pred_test = best_rf_model.predict(X_test)

n_points = 20

# Convert to NumPy arrays and get only the first n_points
y_train = y_train.to_numpy()[:n_points]  # First n_points of y_train
y_test = y_test.to_numpy()[:n_points]    # First n_points of y_test

pred_train = pred_train[:n_points]
pred_test = pred_test[:n_points]

# Calculate the errors for the first n_points only
error_test = (pred_test[:n_points] - y_test[:n_points])  # Errors for test set
error_train = (pred_train[:n_points] - y_train[:n_points])  # Errors for train set



error_train_percentage = (error_train / y_train) * 100
filtered_error_train_percentage = np.where(error_train_percentage > 100, 100, error_train_percentage)
filtered_error_train_percentage = np.where(filtered_error_train_percentage < -100, -100, filtered_error_train_percentage)
filtered_error_train_percentage = filtered_error_train_percentage[:n_points]

error_test_percentage = (error_test / y_test) * 100
filtered_error_test_percentage = np.where(error_test_percentage > 100, 100, error_test_percentage)
filtered_error_test_percentage = np.where(filtered_error_test_percentage < -100, -100, filtered_error_test_percentage)
filtered_error_test_percentage = filtered_error_test_percentage[:n_points]


print("Print Statistics")

output_columns = ['m0_maximum [kg]','m1_maximum [kg]',]


for i, column in enumerate(output_columns):
    print(f"{column:<{max(len(col) for col in output_columns)}} | "
          f"train MAE: {np.mean(np.abs(error_train[:n_points, i])):.4f} [Kg] | "
          f"train ME: {np.mean(error_train[:n_points, i]):.4f} [Kg] | "
          f"Test MAE: {np.mean(np.abs(error_test[:n_points, i])):.4f} [Kg] | "
          f"Test ME: {np.mean(error_test[:n_points, i]):.4f} [Kg]")
print("Best Parameters: ", grid_search.best_params_)



results = range(n_points)

# Create a figure with four subplots (2 rows, 2 columns)
fig, axs = plt.subplots(2, len(output_columns), figsize=(12, 10))  # 2 rows, 2 columns of subplots
if len(output_columns) == 1:
    axs = [axs]

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




