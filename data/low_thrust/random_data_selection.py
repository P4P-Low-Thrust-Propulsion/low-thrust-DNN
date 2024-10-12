import numpy as np
import pandas as pd
import argparse


def format_number(n):
    if n >= 1e6:
        return f"{int(n / 1e6)}M"
    elif n >= 1e3:
        return f"{int(n / 1e3)}K"
    else:
        return str(n)


def generate_transfers(n, original_array, new_array):
    for _ in range(n):
        num_transfer = np.random.randint(0, 468)  # Select random transfer
        start_index = np.random.randint(0, 50)  # Select random starting segment
        end_index = np.random.randint(start_index + 1, 100)  # Select random ending segment

        # Compute indices for start and end rows
        first_row = original_array[start_index + (num_transfer * 100), :]  # First row
        end_row = original_array[end_index + (num_transfer * 100), :]  # End row

        # Combine the first and third row into one longer row
        combined_row = np.hstack((first_row, end_row))

        # Append the combined row to the new array
        new_array.append(combined_row)

    # Convert the new_array to NumPy array
    new_array = np.array(new_array)

    # Create the final array as per column selections
    final = np.zeros((new_array.shape[0], 15))

    final[:, 0:3] = new_array[:, 2:5]  # Copy initial space (x0, y0, z0)
    final[:, 3:6] = new_array[:, 5:8]  # Copy initial velocity (vx0, vy0, vz0)
    final[:, 6:9] = new_array[:, 14:17]  # Copy final space (x1, y1, z1)
    final[:, 9:12] = new_array[:, 17:20]  # Copy final velocity (vx1, vy1, vz1)
    final[:, 12] = new_array[:, 13] - new_array[:, 1]  # Time of flight (tof)
    final[:, 13] = new_array[:, 11]  # Maximum initial mass (m0_maximum)
    final[:, 14] = new_array[:, 23]  # Maximum final mass (m1_maximum)

    # Convert the final array to a DataFrame
    columns = [
        'x0 [AU]', 'y0 [AU]', 'z0 [AU]', 'vx0 [km/s]', 'vy0 [km/s]', 'vz0 [km/s]',
        'x1 [AU]', 'y1 [AU]', 'z1 [AU]', 'vx1 [km/s]', 'vy1 [km/s]', 'vz1 [km/s]',
        'tof [days]', 'm0_maximum [kg]', 'm1_maximum [kg]'
    ]

    final_df = pd.DataFrame(final, columns=columns)

    return final_df


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate specific dataset.")

    # Add arguments
    parser.add_argument('--NUM_TRANSFERS', type=int, default=1000, help='Number of transfers to generate')

    # Parse the arguments
    args = parser.parse_args()

    # Read the CSV file
    filename = 'data/low_thrust/datasets/initial/transfer_information.csv'  # Replace with your CSV file path
    original = np.genfromtxt(filename, delimiter=',', skip_header=1)

    new = []

    df = generate_transfers(args.NUM_TRANSFERS, original, new)

    DATA_NAME = "transfer_statistics_" + format_number(args.NUM_TRANSFERS) + ".csv"

    # Save the DataFrame to a CSV file
    df.to_csv('data/low_thrust/datasets/initial/' + DATA_NAME, index=False)

    print("New dataset saved successfully.")
