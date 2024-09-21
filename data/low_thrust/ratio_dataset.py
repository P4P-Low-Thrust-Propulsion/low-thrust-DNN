import numpy as np
import pandas as pd

# Load the dataset
filename = 'data/low_thrust/datasets/processed/new_transfer_statistics_3000.csv'  # Path to your CSV file
data = pd.read_csv(filename)

# Calculate relative distance between p0 and p1
data['relative_r [AU]'] = data['r1 [AU]'] - data['r0 [AU]']
data['relative_t [AU]'] = data['t1 [AU]'] - data['t0 [AU]']
data['relative_n [AU]'] = data['n1 [AU]'] - data['n0 [AU]']

# Calculate velocity differences (relative velocity between points)
data['relative_vr [km/s]'] = data['vr1 [km/s]'] - data['vr0 [km/s]']
data['relative_vt [km/s]'] = data['vt1 [km/s]'] - data['vt0 [km/s]']
data['relative_vn [km/s]'] = data['vn1 [km/s]'] - data['vn0 [km/s]']

# Calculate the mass ratio m0/m1
data['mass_ratio'] = data['m0_maximum [kg]'] / data['m1_maximum [kg]']

# Select the relevant columns to create the new dataset
new_columns = [
    'relative_r [AU]', 'relative_t [AU]',  'relative_n [AU]', 'relative_vr [km/s]', 'relative_vt [km/s]',
    'relative_vn [km/s]', 'tof [days]', 'm0_maximum [kg]', 'mass_ratio'
]
new_data = data[new_columns]

# Save the new dataset to a CSV file
new_data.to_csv('data/low_thrust/datasets/processed/relative_transfer_statistics.csv', index=False)

print("New dataset saved to 'relative_transfer_statistics.csv'.")
