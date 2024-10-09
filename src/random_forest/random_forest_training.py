import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'data/low_thrust/datasets/processed/new_transfer_statistics_500K_v2.csv'
data = pd.read_csv(file_path)

output_columns = ['m0_maximum [kg]', 'm1_maximum [kg]']

# Separate features (first 10 columns) and target (last two columns: m_0 and m_1)
x = data.iloc[:, :10]  # First 10 columns as features
y = data.iloc[:, -2:]  # Last two columns as targets (m_0 and m_1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

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
grid_search.fit(x_train, y_train)

# Get the best parameters and best model from the grid search
best_rf_model = grid_search.best_estimator_

# Save the model to a file
joblib.dump(best_rf_model, 'src/models/saved_models/random_forest_model_500K.pkl')

print("Model saved successfully!")


print("Print Statistics")
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

for i, column in enumerate(output_columns):
    print(f"{column:<{max(len(col) for col in output_columns)}} | "
          f"train MAE: {np.mean(np.abs(error_train[:, i])):.4f} [Kg] | "
          f"train ME: {np.mean(error_train[:, i]):.4f} [Kg] | "
          f"Test MAE: {np.mean(np.abs(error_test[:, i])):.4f} [Kg] | "
          f"Test ME: {np.mean(error_test[:, i]):.4f} [Kg]")
print("Best Parameters: ", grid_search.best_params_)






