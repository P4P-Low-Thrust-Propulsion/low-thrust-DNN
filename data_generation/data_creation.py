from poliastro.bodies import Earth, Mars, Sun
from astropy.coordinates import solar_system_ephemeris
from poliastro.bodies import Sun
from poliastro.ephem import Ephem
from poliastro.plotting import OrbitPlotter3D
from poliastro.util import time_range
from poliastro.maneuver import Maneuver
from astropy.time import Time
from poliastro.twobody.orbit import Orbit
from astropy import units as u
import numpy as np
import pandas as pd
from tqdm import tqdm


class TransferGenerator:
    def __init__(self, n, flag):
        self.flag = flag
        if self.flag:
            self.plotter = OrbitPlotter3D()
            self.plotter.set_attractor(Sun)

        self.num_samples = n
        self.df = pd.DataFrame()

    def generate_transfers(self):
        data = []
        solar_system_ephemeris.set("jpl")

        # Initial data
        date_launch = Time("2024-01-01 15:02", scale="utc").tdb
        earth_ephem = Ephem.from_body(Earth, date_launch)
        ss_earth = Orbit.from_ephem(Sun, earth_ephem, date_launch)
        earth_vecs = earth_ephem.rv(date_launch)

        r_earth = earth_vecs[0]  # Earth's position vector
        v_earth = earth_vecs[1]  # Earth's velocity vector

        # Define a unit vector pointing from the Sun to the Earth
        sun_to_earth_unit = (r_earth / np.linalg.norm(r_earth)).to(u.dimensionless_unscaled)

        if self.flag:
            self.plotter.plot_ephem(earth_ephem, date_launch, label="Earth at launch position")

        # Loop to generate and plot multiple transfers
        for i in tqdm(range(self.num_samples), desc="Generating transfers"):
            date_arrival = date_launch + np.random.uniform(50, 150) * u.day
            angle = np.random.uniform(np.deg2rad(20), np.deg2rad(60))
            dist_from_earth = np.random.uniform(100e6, 150e6) * u.km

            # Rotate the position vector by the random angle counterclockwise
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                        [np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])
            sun_to_earth_unit_rotated = np.dot(rotation_matrix, sun_to_earth_unit)

            # Calculate the final position vector
            r_final = (r_earth + dist_from_earth) * sun_to_earth_unit_rotated

            # Create orbits from initial and final positions
            end = Orbit.from_vectors(Sun, r_final, v_earth, date_arrival)
            dest_ephem = Ephem.from_orbit(end, time_range(date_launch, end=date_arrival))
            ss_end = Orbit.from_ephem(Sun, dest_ephem, date_arrival)

            # Solve for the transfer maneuver
            man_lambert = Maneuver.lambert(ss_earth, ss_end)

            # Get the transfer and final orbits
            ss_trans, ss_target = ss_earth.apply_maneuver(man_lambert, intermediate=True)

            initial_velocity_vector = man_lambert.impulses[0]
            final_velocity_vector = man_lambert.impulses[1]
            initial_velocity = initial_velocity_vector[1]
            final_velocity = final_velocity_vector[1]
            initial_magnitude = np.linalg.norm(initial_velocity)
            final_magnitude = np.linalg.norm(final_velocity)

            if self.flag:
                print("Delta-v for the transfer:", man_lambert.get_total_cost())
                print("Initial Delta-V: ", initial_magnitude.to(u.km / u.s))
                print("Final Delta-V: ", final_magnitude.to(u.km / u.s))
                print("Transfer orbit:")
                print(man_lambert.get_total_time())

                self.plotter.plot_ephem(dest_ephem, date_arrival, label="Final position", trail=True)
                self.plotter.plot_maneuver(
                    ss_earth,
                    man_lambert,
                    color="black",
                    label="Transfer orbit",
                )

            rel_dist = r_final - r_earth
            # Append the data along with delta V to the list
            data.append({"start_pos_x": r_earth[0] / u.km,
                         "start_pos_y": r_earth[1] / u.km,
                         "start_pos_z": r_earth[2] / u.km,
                         "rel_dist_x": rel_dist[0] / u.km,
                         "rel_dist_y": rel_dist[1] / u.km,
                         "rel_dist_z": rel_dist[2] / u.km,
                         "tof": man_lambert.get_total_time().to(u.day) / u.day,
                         "delta_v_x": initial_velocity[0].to(u.km / u.s) / (u.km/u.s),
                         "delta_v_y": initial_velocity[1].to(u.km / u.s) / (u.km/u.s),
                         "delta_v_z": initial_velocity[2].to(u.km / u.s) / (u.km/u.s)})

        self.df = pd.DataFrame(data)

        if self.flag:
            # Set view parameters
            fig = self.plotter.set_view(30 * u.deg, 260 * u.deg, distance=3 * u.km)
            fig.write_html("plots/multiple_transfers.html")

    def save_data_to_csv(self, filename):
        self.df.to_csv(filename, index=False)
        print("Data has been written to", filename)


# Create an instance of TransferGenerator
transfer_gen = TransferGenerator(10000, False)

# Generate transfer data
transfer_gen.generate_transfers()

# Save data to CSV
transfer_gen.save_data_to_csv("data/transfer_data.csv")


