import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
g0 = 9.81  # gravitational acceleration (m/s^2)
T = 0.6  # thrust
Isp = 4000  # specific impulse
TOF = 150*24*60*60  # TOF in seconds
index = 1208


# Function to calculate delta-v based on max initial and final mass
def tsiolkovsky_rocket_equation(m0, m1):
    return Isp*g0*np.log(m0/m1)


# Function to calculate burn time (t_b) based on initial mass, thrust, Isp, and delta-v
def calculate_burn_time(m0, T, Isp, dv):
    t_b = (m0 * Isp * g0 / T) * (1 - np.exp(-dv / (Isp * g0)))
    return t_b


# Function to calculate corrected delta-v
def corrected_delta_v(dv_lambert, dv_calculated, t_burning, TOF):
    lambda_factor = t_burning / TOF
    dv_estimated = dv_lambert * (1 - lambda_factor) + dv_calculated * lambda_factor
    return dv_estimated


# Load the CSV data
df_known = pd.read_csv('data/low_thrust/datasets/initial/reachability_expected.csv')
df_predicted = pd.read_csv('data/low_thrust/datasets/initial/reachability_dv_estimate.csv')
df_lambert = pd.read_csv('data/low_thrust/datasets/initial/reachability_lambert_transfer_results.csv')

# Extract parameters from the row
m0_known = df_known.loc[index, 'm0_maximum [kg]']
m1_known = df_known.loc[index, 'm1_maximum [kg]']
m0_predicted = df_predicted.loc[index, 'm0_maximum [kg]']
m1_predicted = df_predicted.loc[index, 'm1_maximum [kg]']
dv_lambert = df_lambert.loc[index, 'Final Delta-V [km/s]']

# Create lists to store results during the mass sweep
masses_known = np.linspace(m0_known, 0, 100)  # Sweep from max initial mass to 0
masses_predicted = np.linspace(m0_known, 0, 100)  # Sweep from max initial mass to 0

# Create lists to store results
dv_estimated_known = []
dv_estimated_predicted = []

# Create lists to store burn times for both known and predicted
burn_time_known = []
burn_time_predicted = []

dv_known = tsiolkovsky_rocket_equation(m0_known, m1_known)
dv_predicted = tsiolkovsky_rocket_equation(m0_predicted, m1_predicted)

# Perform the mass sweep and calculate delta-v for each value of m0 for both known and predicted masses
for m0_k, m0_p in zip(masses_known, masses_predicted):
    # --- Known Mass Sweep ---
    t_b_known = calculate_burn_time(m0_k, T, Isp, dv_known)
    burn_time_known.append(t_b_known)
    dv_est_known = corrected_delta_v(dv_lambert, dv_known, t_b_known, TOF)

    # Store the delta-v estimate for known masses
    dv_estimated_known.append(dv_est_known)

    # --- Predicted Mass Sweep ---
    t_b_predicted = calculate_burn_time(m0_p, T, Isp, dv_predicted)
    burn_time_predicted.append(t_b_predicted)
    dv_est_predicted = corrected_delta_v(dv_lambert, dv_predicted, t_b_predicted, TOF)

    # Store the delta-v estimate for predicted masses
    dv_estimated_predicted.append(dv_est_predicted)

# Visualization of delta-v estimates as a function of initial mass for both known and predicted
plt.figure(figsize=(9, 6))

plt.plot(masses_known, dv_estimated_known, label='Delta-v Estimate (Known)', color='blue')
plt.plot(masses_predicted, dv_estimated_predicted, label='Delta-v Estimate (Predicted)', color='red', linestyle='--')

plt.xlabel('Initial Mass (kg)')
plt.ylabel('Delta-v (m/s)')
plt.title('Delta-v Estimate as a Function of Initial Mass (Known vs Predicted)')
plt.legend()
plt.grid(True)

# Invert x-axis to make it decrease
plt.gca().invert_xaxis()

# Show the plot
plt.show()

#  Calculate difference between predicted and known delta-v estimates
dv_difference = np.array(dv_estimated_predicted) - np.array(dv_estimated_known)

# Plot the difference
plt.figure(figsize=(9, 6))
plt.plot(masses_known, dv_difference, label='Delta-v Difference (Predicted - Known)', color='purple')

plt.xlabel('Initial Mass (kg)')
plt.ylabel('Delta-v Difference (m/s)')
plt.title('Difference Between Predicted and Known Delta-v Estimates')
plt.legend()
plt.grid(True)

# Invert x-axis to make it decrease
plt.gca().invert_xaxis()

# Show the plot
plt.show()

# Calculate relative error (as a percentage)
dv_relative_error = 100 * (np.abs(np.array(dv_estimated_predicted) - np.array(dv_estimated_known)) / np.array(dv_estimated_known))

# Plot the relative error
plt.figure(figsize=(9, 6))
plt.plot(masses_known, dv_relative_error, label='Relative Error (%)', color='green')

plt.xlabel('Initial Mass (kg)')
plt.ylabel('Relative Error (%)')
plt.title('Relative Error Between Predicted and Known Delta-v')
plt.legend()
plt.grid(True)

# Invert x-axis to make it decrease
plt.gca().invert_xaxis()

# Show the plot
plt.show()

# Plot burn time vs initial mass
plt.figure(figsize=(9, 6))

plt.plot(masses_known, burn_time_known, label='Burn Time (Known)', color='blue')
plt.plot(masses_predicted, burn_time_predicted, label='Burn Time (Predicted)', color='red', linestyle='--')

plt.xlabel('Initial Mass (kg)')
plt.ylabel('Burn Time (s)')
plt.title('Burn Time as a Function of Initial Mass (Known vs Predicted)')
plt.legend()
plt.grid(True)

# Invert x-axis to make it decrease
plt.gca().invert_xaxis()

# Show the plot
plt.show()

