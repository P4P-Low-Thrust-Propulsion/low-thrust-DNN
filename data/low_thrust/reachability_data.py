import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from poliastro.bodies import Sun, Earth
from poliastro.plotting import OrbitPlotter3D, StaticOrbitPlotter
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from poliastro.util import time_range

mpl.use('macosx')

mu = 1.32712440018e11  # km^3 / s^2
tof = 250  # days
buffer_radius = 0


def plot_vectors_3D(r_initial, r_final, v_initial, v_final, am):
    # Plot position and velocity vectors in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth position vector
    ax.quiver(0, 0, 0, r_initial[0], r_initial[1], r_initial[2], color='b', label='Initial Position')

    # Plot target position vector
    ax.quiver(0, 0, 0, r_final[0], r_final[1], r_final[2], color='r', label='Target Position')

    # Plot perpendicular vector
    perpendicular_vector = np.cross(r_initial, r_final)
    perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector) * 1e8  # Normalize and scale
    ax.quiver(0, 0, 0, am[0], am[1], am[2], color='g',
              label='Perpendicular Vector')

    # Plot Earth velocity vector at the end of the Earth position vector
    ax.quiver(r_initial[0], r_initial[1], r_initial[2], v_initial[0] * 1e7, v_initial[1] * 1e7,
              v_initial[2] * 1e7, color='orange', label='Initial Velocity')

    # Plot target velocity vector at the end of the target position vector
    ax.quiver(r_final[0], r_final[1], r_final[2], v_final[0] * 1e7, v_final[1] * 1e7, v_final[2] * 1e7,
              color='purple', label='Target Velocity')

    # Set limits for the axes
    ax.set_xlim([-3e8, 3e8])
    ax.set_ylim([-3e8, 3e8])
    ax.set_zlim([-3e8, 3e8])

    # Set labels and title
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    plt.title('Orbital Transfer Visualisation: Initial to Target (Transfer)')

    # Display grid and legend
    plt.grid()
    plt.legend()
    plt.show()


# Function to calculate the distance from the Sun given x, y, z
def calc_distance_from_sun(x, y, z):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


# Function to calculate the circular velocity at a distance r from the Sun
def calc_circular_velocity(r):
    return np.sqrt(mu / r)  # km/s


# Function to calculate velocity vector in the same orbital plane
def calc_velocity_in_plane(r_vector, v_magnitude, angular_momentum):
    # Cross product of the position and angular momentum gives the velocity direction
    velocity_direction = np.cross(r_vector, angular_momentum)
    velocity_direction /= np.linalg.norm(velocity_direction)

    return velocity_direction * v_magnitude


# Function to project a point onto a plane defined by a normal vector
def project_point_onto_plane(point, normal_vector):
    # Formula: P' = P - (P ⋅ N) * N / ||N||^2
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    projection = point - np.dot(point, normal_vector) * normal_vector
    return projection


# Initial point values (x0, y0, z0, vx0, vy0, vz0)
initial_point = {
    'x0': 3,  # AU
    'y0': 0,  # AU
    'z0': 0,  # AU
}

# Set up solar system ephemeris (optional but useful for more accuracy)
solar_system_ephemeris.set("jpl")

# Time range for plotting (launch time to arrival time)
date_launch = Time("2025-01-01 00:00", scale="tdb")
date_arrival = date_launch + tof * u.day
times = time_range(date_launch, end=date_arrival)

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

# Create initial orbit using Ephem
initial_orbit = Orbit.from_vectors(Sun, initial_r * u.km, initial_v * u.km / u.s, date_launch)

# Propagate the initial orbit to the date of arrival
final_orbit_at_arrival = initial_orbit.propagate(tof * u.day)

# Get the final position and velocity at the end of TOF
final_r_arrival = final_orbit_at_arrival.r  # Position vector [km]

# Convert final position from km to AU (optional, depending on your desired units)
final_r_arrival_au = final_r_arrival.to(u.AU)

# Define the grid search range around the initial point for final positions
grid_size = 50  # Defines the number of points along each axis
range_x = np.linspace(-0.2 + final_r_arrival_au[0].value, 0.2 + final_r_arrival_au[0].value, grid_size)  # AU
range_y = np.linspace(-0.2 + final_r_arrival_au[1].value, 0.2 + final_r_arrival_au[1].value, grid_size)  # AU
range_z = np.linspace(0, 0, grid_size)  # AU

# Initialize an empty list to store the grid points
grid_points = []

# Loop through all combinations of x1, y1, z1 values to generate final points
for x1 in range_x:
    for y1 in range_y:
        z1 = range_z[0]
        # Final position vector before projection (in AU)
        r1_vector_AU = np.array([x1, y1, z1])

        # # Project the final position vector onto the orbital plane
        # r1_projected = project_point_onto_plane(r1_vector_AU, angular_momentum) * AU_to_km

        # # Calculate the distance from the initial point
        # distance_from_initial = np.sqrt(
        #     ((r1_vector_AU[0] / AU_to_km) - final_r_arrival_au[0].value) ** 2 + ((r1_vector_AU[1] / AU_to_km) - final_r_arrival_au[1].value) ** 2)
        #
        # # Add point to the list if it is outside the buffer radius
        # if distance_from_initial <= buffer_radius:
        #     continue

        # Calculate the distance from the Sun after projection
        r1 = np.linalg.norm(r1_vector_AU * AU_to_km)

        # Calculate the circular velocity at this distance
        v_circ = calc_circular_velocity(r1)

        # Calculate the velocity vector that keeps the orbit in the same plane
        v1_vector = calc_velocity_in_plane(r1_vector_AU * AU_to_km, v_circ, angular_momentum)

        # # Optionally plot the vectors in 3D (you can adjust this based on your plot logic)
        # plot_vectors_3D(initial_r, r1_projected, initial_v, v1_vector, angular_momentum)

        # Append the initial and final points along with the TOF to the list
        grid_points.append([
            initial_point['x0'], initial_point['y0'], initial_point['z0'],
            initial_v[0], initial_v[1], initial_v[2],
            r1_vector_AU[0] , r1_vector_AU[1] , r1_vector_AU[2] ,
            v1_vector[0], v1_vector[1], v1_vector[2], tof
        ])

# Convert the list to a DataFrame
columns = [
    'x0 [AU]', 'y0 [AU]', 'z0 [AU]', 'vx0 [km/s]', 'vy0 [km/s]', 'vz0 [km/s]',
    'x1 [AU]', 'y1 [AU]', 'z1 [AU]', 'vx1 [km/s]', 'vy1 [km/s]', 'vz1 [km/s]',
    'tof [days]'
]
grid_df = pd.DataFrame(grid_points, columns=columns)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))
plotter = StaticOrbitPlotter(ax)
plotter3D = OrbitPlotter3D()
plotter.set_attractor(Sun)
plotter3D.set_attractor(Sun)

earth_ephem = Ephem.from_body(Earth, date_launch)
plotter3D.plot_ephem(earth_ephem, date_launch, label="Earth at launch position", trail=True)

plotter.plot_body_orbit(Earth, date_launch, label="Sun at launch position", trail=True)

# Create initial orbit using Ephem
initial_orbit = Orbit.from_vectors(Sun, initial_r * u.km, initial_v * u.km / u.s, date_launch)
initial_ephem = Ephem.from_orbit(initial_orbit, times)

# Plot initial orbit
plotter.plot_ephem(initial_ephem, date_arrival, label="Initial Orbit", color="Red", trail=True)
plotter3D.plot_ephem(initial_ephem, date_arrival, label="Initial Orbit", trail=True)

# Plot transfer target orbits
for index, row in grid_df.iterrows():
    final_r = np.array([row['x1 [AU]'], row['y1 [AU]'], row['z1 [AU]']]) * AU_to_km * u.km
    final_v = np.array([row['vx1 [km/s]'], row['vy1 [km/s]'], row['vz1 [km/s]']]) * u.km / u.s
    final_orbit = Orbit.from_vectors(Sun, final_r, final_v, date_arrival)
    final_ephem = Ephem.from_orbit(final_orbit, times)
    if index == 10:
        break
    # Only label the first orbit to avoid clutter
    plotter.plot_ephem(final_ephem, date_arrival, label="Transfer Target " + str(index), color="Blue", trail=True)
    plotter3D.plot_ephem(final_ephem, date_arrival, label="Transfer Target " + str(index), trail=True)

# Show plot
fig.show()
fig = plotter3D.set_view(30 * u.deg, 260 * u.deg, distance=3 * u.km)
fig.write_html("data/low_thrust/plots/reachability_transfers.html")
plt.show(block=True)

# Save the dataframe
grid_df.to_csv('data/low_thrust/datasets/initial/reachability_01.csv', index=False)

