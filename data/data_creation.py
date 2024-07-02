from poliastro.bodies import Earth, Mars, Sun
from astropy.coordinates import solar_system_ephemeris
from poliastro.bodies import Sun
from poliastro.ephem import Ephem
from poliastro.plotting import OrbitPlotter3D, StaticOrbitPlotter
from poliastro.util import time_range
from poliastro.maneuver import Maneuver
from astropy.time import Time
from poliastro.twobody.orbit import Orbit
from poliastro.iod import lambert
from astropy import units as u
from astropy.constants import G, M_earth, M_sun
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def format_number(n):
    if n >= 1e6:
        return f"{int(n / 1e6)}M"
    elif n >= 1e3:
        return f"{int(n / 1e3)}K"
    else:
        return str(n)


def rotate_vel_vecs(v_start, v_end, rotation):
    # Apply the rotation matrix
    new_v_start = np.dot(rotation.T, v_start)
    new_v_end = np.dot(rotation.T, v_end)

    # Round the results to avoid small floating-point errors
    new_v_start = np.around(new_v_start, decimals=7)
    new_v_end = np.around(new_v_end, decimals=7)

    return new_v_start, new_v_end


def rotate_pos_vecs(r_start, r_end, flag):
    transfer_plane = TransferFrameRotation(r_start, r_end)
    rotation = transfer_plane.get_rotation()

    # Apply the rotation matrix
    new_r_start = np.dot(rotation.T, r_start)
    new_r_end = np.dot(rotation.T, r_end)

    # Round the results to avoid small floating-point errors
    new_r_start = np.around(new_r_start, decimals=7)
    new_r_end = np.around(new_r_end, decimals=7)

    if flag:
        print("\n ROTATION MATRIX:\n")
        print("r_start:\n", r_start, "\n")
        print("r_end:\n", r_end, "\n")

        print("Rotation Matrix:\n", rotation, "\n")
        print("New r_start (should align with x-axis):\n", new_r_start, "\n")
        print("New r_end:\n", new_r_end, "\n")

    # Validate that new_r_start aligns with the x-axis
    try:
        assert np.allclose(new_r_start[1:], 0), "new_r_start is not aligned with the x-axis."
    except AssertionError as e:
        print("Error:", e)
        exit(1)

    # Validate that new_r_end has no z component
    try:
        assert np.allclose(new_r_end[2:], 0), "new_r_end has a Z component."
    except AssertionError as e:
        print("Error:", e)
        exit(1)

    return new_r_start, new_r_end, rotation


class TransferFrameRotation:
    def __init__(self, r1, r2):
        self.r1 = r1
        self.r2 = r2

    def get_rotation(self):
        r_new = self.r1 / np.linalg.norm(self.r1)  # Normalize r1
        cross_product = np.cross(self.r1, self.r2)

        # Check if r1 and r2 are collinear
        if np.linalg.norm(cross_product) == 0:
            raise ValueError("r1 and r2 are collinear, cannot define a unique rotation plane.")

        n_new = cross_product / np.linalg.norm(cross_product)  # Normalize the cross product
        t_new = np.cross(n_new, r_new)  # Cross product of n_new and r_new for t_new
        t_new = t_new / np.linalg.norm(t_new)

        # Ensure orthogonality
        assert np.allclose(np.dot(r_new, t_new), 0), "r_new and t_new are not orthogonal"
        assert np.allclose(np.dot(r_new, n_new), 0), "r_new and n_new are not orthogonal"
        assert np.allclose(np.dot(t_new, n_new), 0), "t_new and n_new are not orthogonal"

        # Combine into a rotation matrix
        rotation_matrix = np.column_stack((r_new, t_new, n_new))

        return rotation_matrix


class TransferGenerator:
    def __init__(self, n, flag):
        self.flag = flag
        if self.flag:
            self.fig, self.ax = plt.subplots()
            self.plotter = OrbitPlotter3D()
            self.plotter2 = StaticOrbitPlotter(ax=self.ax)
            self.plotter.set_attractor(Sun)
            self.plotter2.set_attractor(Sun)

        self.num_samples = n
        self.df = pd.DataFrame()

    def generate_transfers(self):
        data = []
        solar_system_ephemeris.set("jpl")

        # Initial processed
        date_launch = Time("2011-11-26 15:02", scale="utc").tdb
        earth_ephem = Ephem.from_body(Earth, date_launch)
        ss_earth = Orbit.from_ephem(Sun, earth_ephem, date_launch)
        earth_vecs = earth_ephem.rv(date_launch)

        r_earth = earth_vecs[0]  # Earth's position vector
        v_earth = earth_vecs[1].to(u.km / u.s)  # Earth's velocity vector

        # Define a unit vector pointing from the Sun to the Earth
        sun_to_earth_unit = (r_earth / np.linalg.norm(r_earth)).to(u.dimensionless_unscaled)

        if self.flag:
            self.plotter2.plot_body_orbit(Earth, date_launch, label="Earth at launch position")
            self.plotter.plot_ephem(earth_ephem, date_launch, label="Earth at launch position")

        # Loop to generate and plot multiple transfers
        for i in tqdm(range(self.num_samples), desc="Generating transfers"):
            while True:
                date_arrival = date_launch + np.random.uniform(40, 80) * u.day
                angle = np.random.uniform(np.deg2rad(45), np.deg2rad(90))
                dist_from_earth = np.random.uniform(50e6, 100e6) * u.km

                # Rotate the position vector by the random angle counterclockwise
                rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                            [np.sin(angle), np.cos(angle), 0],
                                            [0, 0, 1]])
                sun_to_earth_unit_rotated = np.dot(rotation_matrix, sun_to_earth_unit)

                # Calculate the final position vector
                r_final = (r_earth + dist_from_earth) * sun_to_earth_unit_rotated

                # Calculate the magnitude of the position vector
                r = np.linalg.norm(r_final)
                G_km = G.to(u.km ** 3 / (u.kg * u.s ** 2))

                # Calculate the circular orbital velocity at this distance for an Earth orbit
                v_final_mag = np.sqrt((G_km * M_sun) / r)

                z_axis = np.cross(r_earth, r_final)
                # Rotate z_axis by a random angle within +-45 degrees
                angle_range = np.deg2rad(np.random.uniform(-45, 45))
                rotation_matrix = np.array([[1, 0, 0],
                                            [0, np.cos(angle_range), np.sin(angle_range)],
                                            [0, -np.sin(angle_range), np.cos(angle_range)]])
                z_axis = np.dot(rotation_matrix, z_axis)

                velocity_direction = np.cross(r_final, z_axis)
                velocity_direction = velocity_direction / np.linalg.norm(velocity_direction)
                v_final = -velocity_direction * v_final_mag

                # Create orbits from initial and final positions
                end = Orbit.from_vectors(Sun, r_final, v_final, date_arrival)
                dest_ephem = Ephem.from_orbit(end, time_range(date_launch, end=date_arrival))
                ss_end = Orbit.from_ephem(Sun, dest_ephem, date_arrival)

                # Solve for the transfer maneuver
                man_lambert = Maneuver.lambert(ss_earth, ss_end, prograde=True)

                # Get the transfer and final orbits
                ss_trans, ss_target = ss_earth.apply_maneuver(man_lambert, intermediate=True)

                tof = man_lambert.get_total_time().to(u.s)
                (v0, vf) = lambert(Sun.k, r_earth, r_final, tof)

                if ss_trans.ecc < 1:
                    break

            initial_dv_vector = man_lambert.impulses[0]
            final_dv_vector = man_lambert.impulses[1]
            initial_dv = initial_dv_vector[1].to(u.km / u.s)
            final_dv = final_dv_vector[1].to(u.km / u.s)
            initial_magnitude = np.linalg.norm(initial_dv)
            final_magnitude = np.linalg.norm(final_dv)

            if self.flag:
                print("Delta-v for the transfer:", man_lambert.get_total_cost())
                print("Initial Delta-V: ", initial_magnitude)
                print("Final Delta-V: ", final_magnitude)
                print("Transfer orbit:")
                print(man_lambert.get_total_time())

                self.plot_vectors_3d(r_earth, r_final, v_earth, v_final, v0, vf, i)
                self.plot_orbit(dest_ephem, date_arrival, ss_earth, man_lambert)

            # Rotate vectors
            r_earth_rot, r_final_rot, rotation_matrix = rotate_pos_vecs(r_earth, r_final, self.flag)
            v0_rot, vf_rot = rotate_vel_vecs(v0, vf, rotation_matrix)
            v_earth_rot, v_final_rot = rotate_vel_vecs(v_earth, v_final, rotation_matrix)

            if self.flag:
                self.plot_vectors_3d(r_earth_rot, r_final_rot, v_earth_rot, v_final_rot, v0_rot, vf_rot, i)

            rel_dist = r_final_rot - r_earth_rot
            # Append the processed along with delta V to the list
            data.append({"rel_dist_x": rel_dist[0].to(u.au) / u.au,
                         "rel_dist_y": rel_dist[1].to(u.au) / u.au,
                         "tof": tof.to(u.day) / u.day,
                         "v0_x": v0_rot[0] / (u.km / u.s),
                         "v0_y": v0_rot[1] / (u.km / u.s),
                         "vf_x": vf_rot[0] / (u.km / u.s),
                         "vf_y": vf_rot[1] / (u.km / u.s)})

        self.df = pd.DataFrame(data)
        logging.info("Data Creation complete")
        if self.flag:
            # Set view parameters
            self.fig.show()
            fig = self.plotter.set_view(30 * u.deg, 260 * u.deg, distance=3 * u.km)
            fig.write_html("data/plots/multiple_transfers.html")
            plt.show(block=True)

    def save_data_to_csv(self, filename):
        self.df.to_csv(filename, index=False)
        logging.info("Data has been written to " + str(filename))

    @staticmethod
    def plot_vectors_3d(r_earth, r_final, v_earth, v_final, v0, vf, i):
        rotated = False
        velocity_label = "Velocity"
        if r_earth[1] == 0:
            rotated = True
        #     velocity_label = "Delta-V"

        # Plot position and velocity vectors in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot Earth position vector
        ax.quiver(0, 0, 0, r_earth[0].value, r_earth[1].value, r_earth[2].value, color='b', label='Earth Position')

        # Plot target position vector
        ax.quiver(0, 0, 0, r_final[0].value, r_final[1].value, r_final[2].value, color='r', label='Target Position')

        # Plot perpendicular vector
        perpendicular_vector = np.cross(r_earth, r_final)
        perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector) * 1e8  # Normalize and scale
        ax.quiver(0, 0, 0, perpendicular_vector[0], perpendicular_vector[1], perpendicular_vector[2], color='g',
                  label='Perpendicular Vector')

        # Plot Earth velocity vector at the end of the Earth position vector
        ax.quiver(r_earth[0], r_earth[1], r_earth[2], v_earth[0].value * 1e7, v_earth[1].value * 1e7,
                  v_earth[2].value * 1e7, color='orange', label='Earth ' + velocity_label)
        # Plot initial velocity vector of transfer arc
        ax.quiver(r_earth[0], r_earth[1], r_earth[2], v0[0].value * 1e7, v0[1].value * 1e7,
                  v0[2].value * 1e7, color='black', label='Initial transfer' + velocity_label)

        # Plot target velocity vector at the end of the target position vector
        ax.quiver(r_final[0], r_final[1], r_final[2], v_final[0] * 1e7, v_final[1] * 1e7, v_final[2] * 1e7,
                  color='purple', label='Target ' + velocity_label)
        # Plot final velocity vector of transfer arc
        ax.quiver(r_final[0], r_final[1], r_final[2], vf[0] * 1e7, vf[1] * 1e7, vf[2] * 1e7,
                  color='black', label='Final transfer ' + velocity_label)

        # Set limits for the axes
        ax.set_xlim([-2e8, 2e8])
        ax.set_ylim([-2e8, 2e8])
        if not rotated:
            ax.set_zlim([-2e8, 2e8])
        else:
            ax.set_aspect('equal', 'box')
        # Set labels and title
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        plt.title('Earth and Target Position and Velocity Vectors (Transfer ' + str(i) + ')')
        if rotated:
            plt.title('Earth and Target Rotated Position and Delta-V Vectors (Transfer ' + str(i) + ')')

        # Display grid and legend
        plt.grid()
        plt.legend()
        plt.show()

    def plot_orbit(self, dest_ephem, date_arrival, ss_earth, man_lambert):
        # Plot orbit transfer and final position
        self.plotter.plot_ephem(dest_ephem, date_arrival, label="Final position", trail=True)
        self.plotter2.plot_ephem(dest_ephem, date_arrival, label="Final position", color="Red", trail=True)
        self.plotter.plot_maneuver(
            ss_earth,
            man_lambert,
            color="black",
            label="Transfer orbit",
        )
        self.plotter2.plot_maneuver(
            ss_earth,
            man_lambert,
            color="black",
            label="Transfer orbit",
        )


# %% Generate processed
# Display plots in separate window
mpl.use('macosx')

DATA_PATH = Path("data/processed")
NUM_TRANSFERS = 5
DATA_NAME = "transfer_data_" + format_number(NUM_TRANSFERS) + ".csv"
DATA_SAVE_PATH = DATA_PATH / DATA_NAME

# Create an instance of TransferGenerator
transfer_gen = TransferGenerator(NUM_TRANSFERS, True)

# Generate transfer processed
transfer_gen.generate_transfers()

# Save processed to CSV
transfer_gen.save_data_to_csv(DATA_SAVE_PATH)

# Less than 10