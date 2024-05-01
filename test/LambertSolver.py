# Imports
from datetime import datetime

import pykep as pk
from matplotlib.dates import date2num
from pykep.orbit_plots import plot_planet, plot_lambert
from pykep import AU, DAY2SEC
import pygmo as pg
import numpy as np

from astropy.time import Time
from astropy import units as u
from poliastro.bodies import Earth, Mars, Sun
from poliastro.twobody.orbit import Orbit
from poliastro.maneuver import Maneuver
from poliastro import ephem
from poliastro.ephem import Ephem
from poliastro.util import time_range
from astropy.coordinates import solar_system_ephemeris

from poliastro.plotting.porkchop import PorkchopPlotter

from poliastro.examples import churi, iss, molniya
from poliastro.plotting import OrbitPlotter3D
import plotly.io as pio
pio.renderers.default = "plotly_mimetype+notebook_connected"

# Plotting imports
import matplotlib as mpl
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# ============================================ Pykep ============================================


def pykep():
    # We define the Lambert problem
    t1 = pk.epoch(0)
    t2 = pk.epoch(640)
    dt = (t2.mjd2000 - t1.mjd2000) * DAY2SEC

    earth = pk.planet.jpl_lp('earth')
    rE, vE = earth.eph(t1)

    mars = pk.planet.jpl_lp('mars')
    rM, vM = mars.eph(t2)

    # We solve the Lambert problem
    l = pk.lambert_problem(r1=rE, r2=rM, tof=dt, mu=pk.MU_SUN, max_revs=2)

    # We plot
    mpl.rcParams['legend.fontsize'] = 10

    # Create the figure and axis
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter([0], [0], [0], color=['y'])

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter([0], [0], [0], color=['y'])
    ax2.view_init(90, 0)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter([0], [0], [0], color=['y'])
    ax3.view_init(0, 0)

    for ax in [ax1, ax2, ax3]:
        # Plot the planet orbits
        plot_planet(earth, t0=t1, color=(0.8, 0.8, 1), legend=True, units=AU, axes=ax)
        plot_planet(mars, t0=t2, color=(0.8, 0.8, 1), legend=True, units=AU, axes=ax)
        # Plot the Lambert solutions
        plot_lambert(l, color='b', legend=True, units=AU, axes=ax)
        plot_lambert(l, sol=1, color='g', legend=True, units=AU, axes=ax)
        plot_lambert(l, sol=2, color='g', legend=True, units=AU, axes=ax)

    fig.show()
    # pk.examples.run_example6()
    # run_example6()


# ============================================ Poliastro ============================================


def pol():
    solar_system_ephemeris.set("jpl")

    # Initial data
    date_launch = Time("2011-11-26 15:02", scale="utc").tdb
    date_arrival = Time("2012-08-06 05:17", scale="utc").tdb

    earth = Ephem.from_body(Earth, time_range(date_launch, end=date_arrival))
    mars = Ephem.from_body(Mars, time_range(date_launch, end=date_arrival))

    # Solve for departure and target orbits
    ss_earth = Orbit.from_ephem(Sun, earth, date_launch)
    ss_mars = Orbit.from_ephem(Sun, mars, date_arrival)

    # Solve for the transfer maneuver
    man_lambert = Maneuver.lambert(ss_earth, ss_mars)

    # Get the transfer and final orbits
    ss_trans, ss_target = ss_earth.apply_maneuver(man_lambert, intermediate=True)

    # Display the delta-v and transfer orbit
    print("Delta-v for the transfer:", man_lambert.get_total_cost())
    print("Transfer orbit:")
    print(man_lambert.get_total_time())

    # You can also plot the transfer orbit
    plotter = OrbitPlotter3D()
    plotter.set_attractor(Sun)

    plotter.plot_ephem(earth, date_launch, label="Earth at launch position")
    plotter.plot_ephem(mars, date_arrival, label="Mars at arrival position")
    fig = plotter.plot_maneuver(
        ss_earth,
        man_lambert,
        color="black",
        label="Transfer orbit",
    )
    plotter.set_view(30 * u.deg, 260 * u.deg, distance=3 * u.km)

    fig.write_html("plots/earth_to_mars.html")


def pork_chop_plt():
    # launch_span = time_range("2005-04-30", end="2005-10-07")
    # arrival_span = time_range("2005-11-16", end="2006-12-21")
    #
    # porkchop_plot = PorkchopPlotter(Earth, Mars, launch_span, arrival_span)
    # dv_dpt, dv_arr, c3dpt, c3arr = porkchop_plot.porkchop()

    # Define departure and arrival time ranges
    departure_start = pk.epoch(Time("2005-05-01").jd, "jd")
    departure_end = pk.epoch(Time("2005-11-01").jd, "jd")
    arrival_start = pk.epoch(Time("2006-01-01").jd, "jd")
    arrival_end = pk.epoch(Time("2006-12-01").jd, "jd")

    # Define departure and arrival time ranges
    departure_dates = np.linspace(departure_start.mjd2000, departure_end.mjd2000, num=1000)
    arrival_dates = np.linspace(arrival_start.mjd2000, arrival_end.mjd2000, num=1000)

    departure_dates_plot = Time(departure_dates+51544, format="mjd").iso
    arrival_dates_plot = Time(arrival_dates+51544, format="mjd").iso

    # Compute delta-v for each combination of departure and arrival dates
    v1 = np.zeros((len(departure_dates), len(arrival_dates)))
    v2 = np.zeros((len(departure_dates), len(arrival_dates)))
    delta_v = np.zeros((len(departure_dates), len(arrival_dates)))
    for i, dep_date in enumerate(departure_dates):
        for j, arr_date in enumerate(arrival_dates):

            dt = (arr_date - dep_date) * DAY2SEC

            earth = pk.planet.jpl_lp('earth')
            rE, vE = earth.eph(dep_date)

            mars = pk.planet.jpl_lp('mars')
            rM, vM = mars.eph(arr_date)

            # Compute Lambert problem for Earth to Mars transfer
            l = pk.lambert_problem(r1=rE, r2=rM, tof=dt, mu=pk.MU_SUN)
            # Get the delta-v components
            delta_v_components = l.get_v1()[0] + l.get_v2()[0]  # Total delta-v is sum of departure and arrival delta-v components
            v1[i, j] = np.linalg.norm(np.array(l.get_v1()[0]) - np.array(vE))
            v2[i, j] = np.linalg.norm(np.array(l.get_v2()[0]) - np.array(vM))
            # Compute magnitude of delta-v
            delta_v[i, j] = (v1[i, j] + v2[i, j]) / 1000 # Convert from m/s to km/s

    # Create Porkchop plot
    plt.figure(figsize=(10, 6))
    contour_levels = np.linspace(np.min(delta_v), np.max(delta_v), 50)
    plt.contourf(departure_dates, arrival_dates, delta_v.T, levels=contour_levels, cmap='viridis')

    # contour_levels1 = np.linspace(np.min(v1), np.max(v1), 50)
    # contour_levels2 = np.linspace(np.min(v2), np.max(v2), 50)
    # plt.contourf(departure_dates, arrival_dates, v1, levels=contour_levels1, cmap='jet')
    # plt.contourf(departure_dates, arrival_dates, v2, levels=contour_levels2, cmap='jet')

    plt.colorbar(label='Delta-V (km/s)')
    plt.xlabel('Departure Date (MJD2000)')
    plt.ylabel('Arrival Date (MJD2000)')
    plt.title('Porkchop Plot: Earth to Mars Transfer')
    plt.grid(True)
    plt.show()

# ============================================ Main ============================================


if __name__ == '__main__':
    pykep()

