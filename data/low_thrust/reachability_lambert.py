import pandas as pd
import numpy as np
from astropy.time import Time
from astropy import units as u
from poliastro.bodies import Sun, Earth
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("data/low_thrust/datasets/initial/reachability_dv_estimate.csv")

# Extract the relevant columns for initial and final conditions, and convert AU to km
AU_to_km = u.AU.to(u.km)

r_initial = df[['x0 [AU]', 'y0 [AU]', 'z0 [AU]']].values * AU_to_km * u.km
v_initial = df[['vx0 [km/s]', 'vy0 [km/s]', 'vz0 [km/s]']].values * u.km / u.s
r_final = df[['x1 [AU]', 'y1 [AU]', 'z1 [AU]']].values * AU_to_km * u.km
v_final = df[['vx1 [km/s]', 'vy1 [km/s]', 'vz1 [km/s]']].values * u.km / u.s
tof = df['tof [days]'].values * u.day  # Time of flight in days

# Create an empty list to store the results
results = []

# Loop through each row and compute the Lambert transfer for each case
for i in range(len(df)):
    # Set launch and arrival dates
    date_launch = Time("2025-01-01 00:00", scale="tdb")
    date_arrival = date_launch + tof[i]

    # Create the initial and final orbits
    start = Orbit.from_vectors(Sun, r_initial[i], v_initial[i], date_launch)
    end = Orbit.from_vectors(Sun, r_final[i], v_final[i], date_arrival)

    # Solve for the transfer maneuver
    man_lambert = Maneuver.lambert(start, end, prograde=True)

    # Get the transfer and final orbits
    ss_trans, ss_target = start.apply_maneuver(man_lambert, intermediate=True)

    # Compute delta-v and maneuver details
    initial_dv_vector = man_lambert.impulses[0]
    final_dv_vector = man_lambert.impulses[1]

    initial_dv = initial_dv_vector[1].value
    final_dv = final_dv_vector[1].value

    initial_magnitude = np.linalg.norm(initial_dv)
    final_magnitude = np.linalg.norm(final_dv)

    # Store results in the list
    results.append({
        "Delta-v Total [km/s]": man_lambert.get_total_cost().value,  # Total delta-v
        "Initial Delta-V [km/s]": initial_magnitude,
        "Final Delta-V [km/s]": final_magnitude,
        "Transfer Time [days]": man_lambert.get_total_time().to(u.day).value
    })

# Convert results list to a DataFrame
results_df = pd.DataFrame(results)

# Save the results DataFrame to a new CSV file
results_df.to_csv("data/low_thrust/datasets/initial/reachability_lambert_transfer_results.csv", index=False)

print("Results saved to 'reachability_lambert_transfer_results.csv'.")

