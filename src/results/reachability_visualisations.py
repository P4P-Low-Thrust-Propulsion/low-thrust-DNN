import numpy as np
import torch.nn.functional
from src.models.DNN import DNNRegressor
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import joblib
from scipy.interpolate import griddata
import matplotlib as mpl
from astropy.time import Time
from astropy import units as u
from poliastro.bodies import Sun, Earth
from poliastro.twobody import Orbit
from poliastro.util import time_range
from poliastro.ephem import Ephem
from pathlib import Path
import matplotlib.tri as tri
import logging
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer
)

# Initial Setup
dnn = True

# Parameters
NUM_NEURONS_1 = 256
NUM_NEURONS_2 = 256
NUM_NEURONS_3 = 128
x_scaler = joblib.load("src/models/saved_models/x_scaler.pkl")
y_scaler = joblib.load("src/models/saved_models/y_scaler.pkl")
TEST_SIZE = 0.2

INPUT_SIZE = 10
OUTPUT_SIZE = 2

RECORD = False

# Display plots in separate window
# mpl.use('macosx')
mu = 1.32712440018e11  # km^3 / s^2
tof = 150  # days

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Unscale the values
def unscale(scaled_value):
    unscaled_value = y_scaler.inverse_transform(scaled_value)
    return unscaled_value


# Function to calculate the circular velocity at a distance r from the Sun
def calc_circular_velocity(r):
    return np.sqrt(mu / r)  # km/s


# Function to calculate velocity vector in the same orbital plane
def calc_velocity_in_plane(r_vector, v_magnitude, angular_momentum):
    # Cross product of the position and angular momentum gives the velocity direction
    velocity_direction = np.cross(r_vector, angular_momentum)
    velocity_direction /= np.linalg.norm(velocity_direction)

    return velocity_direction * v_magnitude


# Function to propagate orbit and store points
def propagate_orbit(initial_orbit, times):
    positions = []
    for time in times:
        orbit_at_time = initial_orbit.propagate(time)
        r = orbit_at_time.r.to(u.AU).value  # Convert to AU
        positions.append(r)
    return np.array(positions)


# Initial point values (x0, y0, z0, vx0, vy0, vz0)
initial_point = {
    'x0': 3.0,  # AU
    'y0': 0,  # AU
    'z0': 0,  # AU
}

# Time range for plotting (launch time to arrival time)
date_launch = Time("2025-01-01 00:00", scale="tdb")
date_arrival = date_launch + tof * u.day
times = time_range(date_launch, end=date_arrival, periods=100)

# Convert initial position to km (AU to km conversion)
AU_to_km = 1.496e8
# Convert initial position to km (AU to km conversion)
initial_r = np.array([initial_point['x0'], initial_point['y0'], initial_point['z0']]) * AU_to_km

# Calculate the circular velocity for the initial point
initial_v_mag = calc_circular_velocity(np.linalg.norm(initial_r))

# Calculate the angular momentum (cross product of position and velocity)
angular_momentum = np.cross(initial_r, [0, -1, 0])  # Assume circular orbit in xy-plane

# Initial velocity in the orbital plane
initial_v = calc_velocity_in_plane(initial_r, initial_v_mag, angular_momentum)

# Propagate the initial orbit over the time range and store positions
initial_orbit = Orbit.from_vectors(Sun, initial_r * u.km, initial_v * u.km / u.s, date_launch)
positions = propagate_orbit(initial_orbit, times)

# Split the propagated positions into x, y, z components
x_positions = positions[:, 0]
y_positions = positions[:, 1]
z_positions = positions[:, 2]

# Create initial orbit using Ephem
initial_orbit = Orbit.from_vectors(Sun, initial_r * u.km, initial_v * u.km / u.s, date_launch)

# Propagate the initial orbit to the date of arrival
final_orbit_at_arrival = initial_orbit.propagate(tof * u.day)

# Get the final position and velocity at the end of TOF
final_r_arrival = final_orbit_at_arrival.r  # Position vector [km]
final_v_arrival = final_orbit_at_arrival.v  # Velocity vector [km/s]

# Convert final position from km to AU (optional, depending on your desired units)
final_r_arrival_au = final_r_arrival.to(u.AU)

print(f"Initial position at date of arrival (in km): {initial_r}")
print(f"Initial velocity at date of arrival (in km/s): {initial_v}")
print(f"Final position at date of arrival (in km): {final_r_arrival}")
print(f"Final position at date of arrival (in AU): {final_r_arrival_au}")
print(f"Final velocity at date of arrival (in km/s): {final_v_arrival}")

# Get Earth's position over the same time range
earth_ephem = Ephem.from_body(Earth, times)
earth_positions = np.array([earth_ephem.rv(time)[0].to(u.AU).value for time in times])
earth_x = earth_positions[:, 0]
earth_y = earth_positions[:, 1]
earth_z = earth_positions[:, 2]

# Load data
file_path = 'data/low_thrust/datasets/processed/reachability_01.csv'
data = pd.read_csv(file_path)

# Load the reachability data
grid_df = pd.read_csv('data/low_thrust/datasets/initial/reachability_01.csv')

if dnn:
    # Load DNN model
    MODEL_PATH = Path("src/models/saved_models/2024-10-02_low_thrust_500K.pth")

    # Check if CUDA (NVIDIA GPU) is available
    cuda_available = torch.cuda.is_available()
    logging.info("CUDA (NVIDIA GPU) available: " + str(cuda_available))

    # Move your model and processed tensors to the GPU (if available)
    device = torch.device("cuda" if cuda_available else "mps")

    torch.manual_seed(42)
    model_01 = DNNRegressor(INPUT_SIZE, OUTPUT_SIZE, NUM_NEURONS_1, NUM_NEURONS_2, NUM_NEURONS_3)
    model_01.to(device)
    model_01.load_state_dict(torch.load(MODEL_PATH))

    df_Features = data.iloc[:, :INPUT_SIZE]
    df_Labels = data.iloc[:, -OUTPUT_SIZE:]

    df_Features = x_scaler.transform(df_Features)

    # Convert to torch tensor and move to device
    x_test = torch.tensor(df_Features, dtype=torch.float32).to(device)

    # Make estimates
    model_01.eval()
    with torch.no_grad():
        pred_test = model_01(x_test)

    # Convert predictions back to CPU
    pred_test_np = pred_test.cpu().numpy()

    # Apply to unscale function to each column of inputs arrays
    pred = unscale(pred_test_np)
    predicted_m0 = pred[:, 0]
    # predicted_m0 = df_Labels['m0_maximum [kg]'].values

    # Set values above 2500 to 2500
    predicted_m0 = np.where(predicted_m0 > 2600, 2600, predicted_m0)

else:
    # Load the saved model from the file
    best_rf_model = joblib.load('src/models/saved_models/random_forest_model_200K.pkl')

    # Make predictions using the model
    pred = best_rf_model.predict(data)
    predicted_m0 = pred[:, 0]

# Extract x, y, and initial mass data
x = grid_df['x1 [AU]'].values
y = grid_df['y1 [AU]'].values

# Create a triangulation object using x, y, z coordinates
triang = tri.Triangulation(x, y)

# Create a grid for contouring
# x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 200), np.linspace(y.min(), y.max(), 200))
# m0_grid = griddata((x, y), predicted_m0, (x_grid, y_grid), method='cubic')

# Plot the contour plot
plt.figure(figsize=(9, 6))
contour = plt.tricontourf(triang, predicted_m0, levels=26, cmap="viridis")  # Contour plot
plt.colorbar(contour, label='Predicted Maximum Initial Mass [kg]')  # Colorbar to show m0 values

# plt.contourf(x_grid, y_grid, m0_grid, cmap='viridis')
# plt.colorbar(label='Predicted Initial Mass (m0) [kg]')

# Mark the initial point
# plt.scatter(grid_df['x0 [AU]'], grid_df['y0 [AU]'], color='red', label='Departure Point @T=0', s=50)
# plt.scatter(final_r_arrival_au[0], final_r_arrival_au[1], color='magenta', label='Departure Point @T=tof', s=100)

# # Plot the trajectory (arc) by plotting the stored positions
# plt.plot(x_positions, y_positions, color='magenta', linestyle='--', label='Orbit Arc')
#
# # Plot Earth's orbit and position
# plt.plot(earth_x, earth_y, color='blue', linestyle='--', label="Earth's Orbit")
# plt.scatter(earth_x[-1], earth_y[-1], color='blue', label="Earth at Arrival", s=200, zorder=10)
#
# plt.scatter(0, 0, c='orange', label='Sun', marker='o', s=100)

# Add labels and legend
plt.xlabel('X [AU]')
plt.ylabel('Y [AU]')
plt.title('Reachability Analysis: Contour (2D)')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

# 2D Visualization
plt.figure(figsize=(9, 6))

# Plot the Sun
plt.scatter(0, 0, c='orange', label='Sun', marker='o', s=100)

# Scatter plot with color map for predicted initial mass
scatter = plt.scatter(grid_df['x1 [AU]'], grid_df['y1 [AU]'], c=predicted_m0, cmap='viridis', s=50, alpha=0.75)
plt.colorbar(scatter, label='Predicted Maximum Initial Mass [kg]')

# Mark the initial point
plt.scatter(grid_df['x0 [AU]'], grid_df['y0 [AU]'], color='red', label='Departure Point @T=0', s=50)
plt.scatter(final_r_arrival_au[0], final_r_arrival_au[1], color='magenta', label='Departure Point @T=tof', s=100)

# Plot the trajectory (arc) by plotting the stored positions
plt.plot(x_positions, y_positions, color='magenta', linestyle='--', label='Orbit Arc')

# Plot Earth's orbit and position
plt.plot(earth_x, earth_y, color='blue', linestyle='--', label="Earth's Orbit")
plt.scatter(earth_x[-1], earth_y[-1], color='blue', linestyle='--', label="Earth at Arrival", s=200, zorder=10)

# Plot settings
plt.xlabel('X [AU]')
plt.ylabel('Y [AU]')
plt.title('Reachability Analysis: Predicted Maximum Initial Mass (2D)')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

# 3D Visualization
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color map for predicted initial mass
scatter = ax.scatter(grid_df['x1 [AU]'], grid_df['y1 [AU]'], grid_df['z1 [AU]'],
                     c=predicted_m0, cmap='viridis', s=50, alpha=0.75)
fig.colorbar(scatter, ax=ax, label='Predicted Maximum Initial Mass [kg]')

# Plot the Sun
ax.scatter(0, 0, 0, c='orange', label='Sun', marker='o', s=100)

# Mark the initial point
ax.scatter(grid_df['x0 [AU]'], grid_df['y0 [AU]'], grid_df['z0 [AU]'], color='red', label='Departure Point @T=0',
           s=50,  zorder=10)
ax.scatter(final_r_arrival_au[0], final_r_arrival_au[1], final_r_arrival_au[2], color='magenta',
           label='Departure Point @T=tof', s=100, zorder=10)

# Plot the trajectory (arc) by plotting the stored positions
ax.plot(x_positions, y_positions, z_positions, color='magenta', linestyle='--', label='Orbit Arc')

# Plot Earth's orbit and position
ax.plot(earth_x, earth_y, earth_z, color='blue', linestyle='--', label="Earth's Orbit")
ax.scatter(earth_x[-1], earth_y[-1], earth_z[-1], color='blue', label="Earth at Arrival", s=200, zorder=10)

# Plot settings
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_zlabel('Z [AU]')
ax.set_title('Reachability Analysis: Predicted Maximum Initial Mass (3D)')
ax.legend()
plt.show(block=True)

