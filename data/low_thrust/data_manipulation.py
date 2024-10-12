from cmath import sqrt
import numpy as np
import pandas as pd
import argparse


def eci_to_rtn(position, velocity):
    """
    Converts a state vector from the Earth-Centered Inertial (ECI) frame to the Radial-Transverse-Normal (RTN) frame.

    Parameters:
    position (np.array): Position vector in the ECI frame (3 elements)
    velocity (np.array): Velocity vector in the ECI frame (3 elements)

    Returns:
    np.array: State vector in the RTN frame
    """

    # Compute specific angular momentum
    h = np.cross(position, velocity)

    # Compute unit vectors in RTN frame
    r_RTN = position / np.linalg.norm(position)
    n_RTN = h / np.linalg.norm(h)
    t_RTN = np.cross(n_RTN, position)
    t_RTN = t_RTN / np.linalg.norm(t_RTN)

    # Construct rotation matrix from ECI to RTN frame
    R = np.vstack([r_RTN, t_RTN, n_RTN]).T

    # Transform position and velocity vectors from ECI to RTN frame
    position_rtn = np.dot(R.T, position)
    velocity_rtn = np.dot(R.T, velocity)

    # Round the results to avoid small floating-point errors
    position_rtn = np.around(position_rtn, decimals=7)
    velocity_rtn = np.around(velocity_rtn, decimals=7)

    # Combine into state vector in RTN frame
    state_rtn = np.hstack([position_rtn, velocity_rtn])

    return state_rtn, R.T


def rv_to_coe(X, mu):
    """
    Converts position and velocity vectors to classical orbital elements (COE).

    Parameters:
    X (np.array): State vector containing position and velocity vectors [x, y, z, xdot, ydot, zdot].
    mu (float): Gravitational parameter of the central body (km^3/s^2).

    Returns:
    dict: Classical orbital elements (COE) as a dictionary:
        - a: Semi-major axis (km)
        - e: Eccentricity
        - i: Inclination (radians)
        - Omega: Longitude of ascending node (radians)
        - omega: Argument of periapsis (radians)
        - theta: True anomaly (radians)
    """

    # Extract position and velocity vectors
    r = X[:3]  # Position vector
    v = X[3:]  # Velocity vector

    # Compute specific angular momentum
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)

    # Compute eccentricity vector
    e_vec = (np.cross(v, h) / mu) - (r / np.linalg.norm(r))
    e = np.linalg.norm(e_vec)

    # Compute specific energy
    E = 0.5 * np.linalg.norm(v) ** 2 - mu / np.linalg.norm(r)

    # Compute semi-major axis
    if abs(E) > 1e-10:  # Avoid division by very small numbers
        a = -mu / (2 * E)
    else:
        a = np.inf  # Parabolic orbits where energy is near zero

    # Compute inclination
    if h_norm > 0:
        i = np.arccos(h[2] / h_norm)
    else:
        i = 0  # Edge case where angular momentum vector is zero
    i = np.degrees(i)

    # Compute right ascension of ascending node (RAAN)
    N = np.cross([0, 0, 1], h)
    N_norm = np.linalg.norm(N)

    if N_norm > 0:
        Omega = np.arccos(N[0] / N_norm)
        if N[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0  # Equatorial orbit (no ascending node)
    Omega = np.degrees(Omega)

    # Compute true anomaly
    if e > 0:
        if np.dot(r, v) >= 0:
            theta = np.arccos(np.dot(e_vec, r) / (e * np.linalg.norm(r)))
        else:
            theta = 2 * np.pi - np.arccos(np.dot(e_vec, r) / (e * np.linalg.norm(r)))
    else:
        theta = 0  # Circular orbit (no true anomaly)
    theta = np.degrees(theta)

    # Compute argument of periapsis
    if e > 0 and N_norm > 0:
        omega = np.arccos(np.dot(N, e_vec) / (N_norm * e))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
            omega = np.degrees(omega)
    else:
        omega = abs(theta-360)

    # Conversion factor from AU to km
    km_to_au = 1 / 149597870.7

    # Store COE in a dictionary
    coe = {
        'a': a * km_to_au,  # Semi-major axis (km)
        'e': e,  # Eccentricity
        'i': i,  # Inclination (radians)
        'Omega': Omega,  # Longitude of ascending node (radians)
        'omega': omega,  # Argument of periapsis (radians)
        'theta': theta  # True anomaly (radians)
    }

    return coe


def convert_row_to_rtn(row):

    # Conversion factor from AU to km
    au_to_km = 149597870.7

    # Convert positions from AU to km
    pos0 = np.array([row['x0 [AU]'], row['y0 [AU]'], row['z0 [AU]']]) * au_to_km
    pos1 = np.array([row['x1 [AU]'], row['y1 [AU]'], row['z1 [AU]']]) * au_to_km

    # Velocities remain in km/s
    vel0 = np.array([row['vx0 [km/s]'], row['vy0 [km/s]'], row['vz0 [km/s]']])
    vel1 = np.array([row['vx1 [km/s]'], row['vy1 [km/s]'], row['vz1 [km/s]']])

    [departure_rtn_state, rtn_matrix] = eci_to_rtn(pos0, vel0)

    rtn_pos0 = np.dot(rtn_matrix, pos0)
    rtn_vel0 = np.dot(rtn_matrix, vel0)
    rtn_pos1 = np.dot(rtn_matrix, pos1)
    rtn_vel1 = np.dot(rtn_matrix, vel1)

    arrival_rtn_state = np.hstack([rtn_pos1, rtn_vel1])

    return {
        'x0_rtn': rtn_pos0[0],
        'y0_rtn': rtn_pos0[1],
        'z0_rtn': rtn_pos0[2],
        'vx0_rtn': rtn_vel0[0],
        'vy0_rtn': rtn_vel0[1],
        'vz0_rtn': rtn_vel0[2],
        'x1_rtn': rtn_pos1[0],
        'y1_rtn': rtn_pos1[1],
        'z1_rtn': rtn_pos1[2],
        'vx1_rtn': rtn_vel1[0],
        'vy1_rtn': rtn_vel1[1],
        'vz1_rtn': rtn_vel1[2]
    }, rv_to_coe(departure_rtn_state, 1.32712440018e11), rv_to_coe(arrival_rtn_state, 1.32712440018e11)


def add_to_new_dataset(data, row, dep_coe, arr_coe, converted_row):
    km_to_au = 1 / 149597870.7
    data.append({
        # RTN Frame Values for Departure
        'r0 [AU]': converted_row['x0_rtn'] * km_to_au,
        'vr0 [km/s]': converted_row['vx0_rtn'],
        'vt0 [km/s]': converted_row['vy0_rtn'],

        # RTN Frame Values for Arrival
        'r1 [AU]': converted_row['x1_rtn'] * km_to_au,
        't1 [AU]': converted_row['y1_rtn'] * km_to_au,
        'n1 [AU]': converted_row['z1_rtn'] * km_to_au,
        'vr1 [km/s]': converted_row['vx1_rtn'],
        'vt1 [km/s]': converted_row['vy1_rtn'],
        'vn1 [km/s]': converted_row['vz1_rtn'],

        # Time of Flight and Mass Values
        'tof [days]': row['tof [days]'],
        'm0_maximum [kg]': row['m0_maximum [kg]'],
        'm1_maximum [kg]': row['m1_maximum [kg]']
    })


def compare_rows(original_row, converted_row, coe_d, coe_a, row_num, tolerance=1e-5):
    match_count = 0
    total_count = len(original_row)

    # Conversion factor from AU to km
    km_to_au = 1 / 149597870.7

    print('=' * 10)
    print(f"Row: {row_num}")
    print('=' * 10)
    print(f"{'Point':<10} {'a':<10} {'e':<10} {'i':<10} {'Omega':<10} {'omega':<10} {'theta'}")
    print('-' * 70)
    print(f"{'Departure':<10} {coe_d['a']:<10.4} {coe_d['e']:<10.5} {coe_d['i']:<10.5} {coe_d['Omega']:<10.5} "
          f"{coe_d['omega']:<10.5} {coe_d['theta']:<10.5}")
    print(f"{'Arrival':<10} {coe_a['a']:<10.4} {coe_a['e']:<10.5} {coe_a['i']:<10.5} {coe_a['Omega']:<10.5} "
          f"{coe_a['omega']:<10.5} {coe_a['theta']:<10.5}")
    print('-' * 70)
    print()

    print(f"{'Column':<15} {'Original Value':<20} {'Converted Value':<20} {'Match'}")
    print('-' * 70)

    # Comparison columns and their corresponding converted columns
    columns = {
        'r0 [AU]': 'x0_rtn',
        't0 [AU]': 'y0_rtn',
        'n0 [AU]': 'z0_rtn',
        'vr0 [km/s]': 'vx0_rtn',
        'vt0 [km/s]': 'vy0_rtn',
        'vn0 [km/s]': 'vz0_rtn',
        'r1 [AU]': 'x1_rtn',
        't1 [AU]': 'y1_rtn',
        'n1 [AU]': 'z1_rtn',
        'vr1 [km/s]': 'vx1_rtn',
        'vt1 [km/s]': 'vy1_rtn',
        'vn1 [km/s]': 'vz1_rtn'
    }

    for original_key, converted_key in columns.items():
        original_value = original_row[original_key]

        # Convert position values back to AU for comparison
        if original_key.startswith('r') or original_key.startswith('t') or original_key.startswith('n'):
            converted_value = converted_row[converted_key] * km_to_au
        else:
            converted_value = converted_row[converted_key]

        match = 'Match' if np.isclose(original_value, converted_value, atol=tolerance) else 'Mismatch'

        # Update match count
        if match == 'Match':
            match_count += 1

        print(f"{original_key:<15} {original_value:<20.6e} {converted_value:<20.6e} {match}")

    # Calculate percentage of matches
    percentage_match = (match_count / len(columns)) * 100
    print('-' * 70)
    print(f"Percentage of Matches: {percentage_match:.2f}%\n\n")
    return percentage_match


def evaluate_accuracy(df, tolerance=1e-5):
    total_match_percentage = 0
    row_count = len(df)
    data = []

    for row_index in range(row_count):
        single_row = df.iloc[row_index]
        converted_row, departure_coe, arrival_coe = convert_row_to_rtn(single_row)
        add_to_new_dataset(data, single_row, departure_coe, arrival_coe, converted_row)

    # THE BELOW CODE MAY BE UNCOMMENTED WHEN COMPARING AGAINST JACKS DATASET THAT CONTAINS THE CORRECT CONVERTED VALUES

    #     row_percentage = compare_rows(single_row, converted_row, departure_coe, arrival_coe, row_index, tolerance)
    #     total_match_percentage += row_percentage
    #
    # overall_accuracy = total_match_percentage / row_count
    # print(f"Number of Rows: {row_count}")
    # print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Scaling constants
    mu = 1.32712440018e11  # km^3 / s^2
    r_scale = 1.49597870691e8  # km / LU
    v_scale = sqrt(mu / r_scale)  # km/s / LU/TU

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate specific dataset.")

    # Add arguments
    parser.add_argument('--DATA_NAME', type=str, default="transfer_statistics_100.csv", help='File to rotate')

    # Parse the arguments
    args = parser.parse_args()

    # Read the CSV file
    df = pd.read_csv('data/low_thrust/datasets/initial/' + args.DATA_NAME)

    # THE BELOW CODE MAY BE UNCOMMENTED WHEN COMPARING AGAINST JACKS DATASET THAT CONTAINS THE CORRECT CONVERTED VALUES
    # velocity_columns = ['vx0 [km/s]', 'vy0 [km/s]', 'vz0 [km/s]', 'vx1 [km/s]', 'vy1 [km/s]', 'vz1 [km/s]',
    # 'vr0 [km/s]', 'vt0 [km/s]', 'vn0 [km/s]', 'vr1 [km/s]', 'vt1 [km/s]', 'vn1 [km/s]']

    velocity_columns = ['vx0 [km/s]', 'vy0 [km/s]', 'vz0 [km/s]', 'vx1 [km/s]', 'vy1 [km/s]', 'vz1 [km/s]']

    # Multiply the velocities by v_scale
    df[velocity_columns] = df[velocity_columns]  # if using jacks old dataset * v_scale.real

    # Evaluate overall accuracy
    new_df = evaluate_accuracy(df)

    # Save the new dataset to a CSV file
    new_df.to_csv('data/low_thrust/datasets/processed/new_' + args.DATA_NAME, index=False)

    print("New dataset with orbital elements saved successfully to new_" + args.DATA_NAME)
