from numpy import *
from math import *
import pykep as pk
from pykep import DAY2SEC, AU, MU_SUN, RAD2DEG
from pykep.planet import keplerian, jpl_lp, spice


class Flyby:
    jpl_planet = {'mercury': jpl_lp('mercury'),
                  'venus': jpl_lp('venus'),
                  'earth': jpl_lp('earth'),
                  'mars': jpl_lp('mars'),
                  'jupiter': jpl_lp('jupiter'),
                  'saturn': jpl_lp('saturn'),
                  'uranus': jpl_lp('uranus'),
                  'neptune': jpl_lp('neptune'),
                  'pluto': jpl_lp('pluto')
                  }

    mu_sun_spice = 132712440017.99e9
    spice_planet = {'mercury': spice('MERCURY', 'SUN', 'ECLIPJ2000', 'NONE', mu_sun_spice, 22032.080486418e9, 2439.7e3,
                                     2439.7e3 * 1.1),
                    'venus': spice('VENUS', 'SUN', 'ECLIPJ2000', 'NONE', mu_sun_spice, 324858.59882646e9, 6051.9e3,
                                   6051.9e3 * 1.1),
                    'earth': spice('EARTH', 'SUN', 'ECLIPJ2000', 'NONE', mu_sun_spice, 398600.4415e9, 6378.1363e3,
                                   6378.1363e3 * 1.1),
                    'mars': spice('MARS', 'SUN', 'ECLIPJ2000', 'NONE', mu_sun_spice, 42828.314258067e9, 3397.0e3,
                                  3397.0e3 * 1.1),
                    'jupiter': spice('JUPITER', 'SUN', 'ECLIPJ2000', 'NONE', mu_sun_spice, 126712767.8578e9, 71492.0e3,
                                     71492.0e3 * 1.1),
                    'saturn': spice('SATURN', 'SUN', 'ECLIPJ2000', 'NONE', mu_sun_spice, 37940626.061137e9, 60268.0e3,
                                    60268.0e3 * 1.1),
                    'uranus': spice('URANUS', 'SUN', 'ECLIPJ2000', 'NONE', mu_sun_spice, 5794549.0070719e9, 25559.0e3,
                                    25559.0e3 * 1.1),
                    'neptune': spice('NEPTUNE', 'SUN', 'ECLIPJ2000', 'NONE', mu_sun_spice, 6836534.0638793e9, 25269.0e3,
                                     25269.0e3 * 1.1),
                    'pluto': spice('PLUTO', 'SUN', 'ECLIPJ2000', 'NONE', mu_sun_spice, 981.600887707e9, 1162.0e3,
                                   1162.0e3 * 1.1)
                    }

    def __init__(self, flight_plan, travel_days, windows, ignore_last=False, orbit_alt=300000, days=1, spice=False,
                 multi_revs=5):

        self.travel_days = travel_days.copy()
        self.windows = windows.copy()
        self.orbit_alt = orbit_alt
        self.ignore_last = ignore_last
        self.dim = len(flight_plan)
        self.flight_plan = flight_plan.copy()
        self.days = days
        self.spice = spice
        self.multi_revs = multi_revs

        self.planets = []
        if self.spice:
            self.mu_sun = self.mu_sun_spice
            for i in flight_plan:
                self.planets.append(self.spice_planet[i])
        else:
            self.mu_sun = MU_SUN
            for i in flight_plan:
                self.planets.append(self.jpl_planet[i])

        self.x = [0] * self.dim
        self.t = [0] * self.dim
        self.r = [0] * self.dim
        self.v = [0] * self.dim
        self.vo = [0] * self.dim
        self.vi = [0] * self.dim
        self.f = [0] * self.dim
        self.f_all = 0
        self.li_sol = []
        self.l = []

    def get_bounds(self):
        return [-1 * x for x in self.windows], self.windows

    def get_name(self):
        return "Flyby"

    def fitness(self, x):
        self.x = x
        # calculate the times
        self.t[0] = self.travel_days[0] + self.x[0]
        for i in range(1, self.dim):
            self.t[i] = self.days * self.t[i - 1] + self.travel_days[i] + self.x[i]

        # calculate the state vectors of planets
        for i in range(self.dim):
            self.r[i], self.v[i] = self.planets[i].eph(pk.epoch(self.t[i], "mjd"))

        # calculate the solutions of the two Lambert transfers
        self.l = []
        n_sols = []
        for i in range(self.dim - 1):
            self.l.append(
                pk.lambert_problem(self.r[i], self.r[i + 1], (self.t[i + 1] - self.t[i]) * DAY2SEC, self.mu_sun, False,
                                self.multi_revs))
            n_sols.append(self.l[i].get_Nmax() * 2 + 1)

        # perform the dV calculations
        mu0 = self.planets[0].mu_self
        rad0 = self.planets[0].radius + self.orbit_alt
        mu1 = self.planets[-1].mu_self
        rad1 = self.planets[-1].radius + self.orbit_alt

        k = 1
        for i in range(self.dim - 1):
            k = k * n_sols[i]

        vot = [0] * self.dim
        vit = [0] * self.dim
        ft = [0] * self.dim
        self.f_all = 1.0e10

        for kk in range(k):
            d = kk
            li = []
            for j in range(self.dim - 1):
                d, b = divmod(d, n_sols[j])
                li.append(b)

            vot[0] = array(self.l[0].get_v1()[li[0]]) - self.v[0]
            ft[0] = sqrt(dot(vot[0], vot[0]) + 2 * mu0 / rad0) - sqrt(1 * mu0 / rad0)

            if ft[0] > self.f_all:
                continue

            for i in range(1, self.dim - 1):
                vit[i] = array(self.l[i - 1].get_v2()[li[i - 1]]) - self.v[i]
                vot[i] = array(self.l[i].get_v1()[li[i]]) - self.v[i]
                ft[i] = pk.fb_vel(vit[i], vot[i], self.planets[i])

            vit[-1] = array(self.l[-1].get_v2()[li[-1]]) - self.v[-1]
            ft[-1] = sqrt(dot(vit[-1], vit[-1]) + 2 * mu1 / rad1) - sqrt(2 * mu1 / rad1)

            ft_all = sum(ft)
            if self.ignore_last:
                ft_all = ft_all - ft[-1]

            if ft_all < self.f_all:
                self.f_all = ft_all
                self.vi = vit.copy()
                self.vo = vot.copy()
                self.f = ft.copy()
                self.li_sol = li.copy()

        # check and set cost of negative altitude (using safe_radius)
        res = self.f_all
        for i in range(1, self.dim - 1):
            ta = acos(
                dot(self.vi[i], self.vo[i]) / sqrt(dot(self.vi[i], self.vi[i])) / sqrt(dot(self.vo[i], self.vo[i])))
            alt = (self.planets[i].mu_self / dot(self.vi[i], self.vi[i]) * (1 / sin(ta / 2) - 1)) - self.planets[
                i].safe_radius
            if alt < 0:
                res = res - alt

        # return the total fuel cost
        return [res]

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def plot_trajectory(self):
        from matplotlib.pyplot import figure, show
        from pykep.orbit_plots import plot_planet, plot_lambert

        fig = figure()
        ax = fig.add_subplot(projection='3d', aspect='equal')
        ax.scatter([0], [0], [0], color='y')

        colors = {'mercury': '#7B7869',
                  'venus': '#BB91A1',
                  'earth': '#0000FF',
                  'mars': '#E27B58',
                  'jupiter': '#C88B3A',
                  'saturn': '#A49B72',
                  'uranus': '#65868B',
                  'neptune': '#6081FF',
                  'pluto': '#333333'}

        d_max = 0
        for i in range(self.dim):
            r, v = self.planets[i].eph(pk.epoch(self.t[i], "mjd"))
            d = dot(r, r)
            if d > d_max:
                d_max = d

            p = keplerian(pk.epoch(self.t[i], "mjd"),
                          self.planets[i].osculating_elements(pk.epoch(self.t[i], "mjd")),
                          self.planets[i].mu_central_body,
                          self.planets[i].mu_self,
                          self.planets[i].radius,
                          self.planets[i].safe_radius,
                          self.flight_plan[i])
            plot_planet(p, pk.epoch(self.t[i], "mjd"), units=AU, color=colors[self.flight_plan[i]], axes=ax)
            if i != self.dim - 1:
                plot_lambert(self.l[i], sol=self.li_sol[i], units=AU, color='c', axes=ax)

        d_max = 1.2 * sqrt(d_max) / AU
        ax.set_xlim(-d_max, d_max)
        ax.set_ylim(-d_max, d_max)
        ax.set_zlim(-d_max, d_max)

        show()

    def print_transx(self):

        print("Date of %8s departue : " % self.flight_plan[0], pk.epoch(self.t[0], "mjd"))
        for i in range(1, self.dim - 1):
            print("Date of %8s encounter: " % self.flight_plan[i], pk.epoch(self.t[i], "mjd"))
        print("Date of %8s arrival  : " % self.flight_plan[-1], pk.epoch(self.t[-1], "mjd"))
        print("")

        for i in range(self.dim - 1):
            print("Transfer time from %8s to %8s:" % (self.flight_plan[i], self.flight_plan[i + 1]),
                  round(self.t[i + 1] - self.t[i], 4), " days")

        print("Total mission duration:                 ", round(self.t[-1] - self.t[0], 4), " days")
        print("")
        print("")

        fward = [0] * self.dim
        plane = [0] * self.dim
        oward = [0] * self.dim
        for i in range(self.dim):
            fward[i] = self.v[i] / linalg.norm(self.v[i])
            plane[i] = cross(self.v[i], self.r[i])
            plane[i] = plane[i] / linalg.norm(plane[i])
            oward[i] = cross(plane[i], fward[i])

        print("TransX escape plan -  %8s escape" % self.flight_plan[0])
        print("--------------------------------------")
        print("MJD:                  %10.4f " % round(pk.epoch(self.t[0], "mjd").mjd, 4))
        print("Prograde:             %10.4f m/s" % round(dot(fward[0], self.vo[0]), 4))
        print("Outward:              %10.4f m/s" % round(dot(oward[0], self.vo[0]), 4))
        print("Plane:                %10.4f m/s" % round(dot(plane[0], self.vo[0]), 4))
        print("Hyp. excess velocity: %10.4f m/s" % round(linalg.norm(self.vo[0]), 4))
        print("Earth escape burn:    %10.4f m/s" % round(self.f[0], 4))

        c3 = dot(self.vo[0], self.vo[0]) / 1000000
        dha = atan2(self.vo[0][2], sqrt(self.vo[0][0] * self.vo[0][0] + self.vo[0][1] * self.vo[0][1])) * RAD2DEG
        rha = atan2(self.vo[0][1], self.vo[0][0]) * RAD2DEG

        print("GMAT MJD:             ", pk.epoch(self.t[0], "mjd").mjd - 29999.5)
        print("GMAT OutgoingC3:      ", c3)
        print("GMAT OutgoingRHA:     ", rha)
        print("GMAT OutgoingDHA:     ", dha)
        print("")

        for i in range(1, self.dim - 1):
            vx = dot(fward[i], self.vo[i])
            vy = dot(oward[i], self.vo[i])
            vz = dot(plane[i], self.vo[i])
            mu = self.planets[i].mu_self
            rad = self.planets[i].radius
            print("%8s encounter" % self.flight_plan[i])
            print("--------------------------------------")
            print("MJD:                 %10.4f " % round(pk.epoch(self.t[i], "mjd").mjd, 4))
            print("Solution number:     %10d " % (1 + self.li_sol[i - 1]))
            print("Approach velocity:   %10.4f m/s" % round(linalg.norm(self.vi[i]), 4))
            print("Departure velocity:  %10.4f m/s" % round(linalg.norm(self.vo[i]), 4))
            print("Outward angle:       %10.4f deg" % round(atan2(vy, vx) * RAD2DEG, 4))
            print("Inclination:         %10.4f deg" % round(atan2(vz, sqrt(vx * vx + vy * vy)) * RAD2DEG, 4))

            a = - mu / dot(self.vi[i], self.vi[i])
            ta = acos(dot(self.vi[i], self.vo[i]) / linalg.norm(self.vi[i]) / linalg.norm(self.vo[i]))
            e = 1 / sin(ta / 2)
            rp = a * (1 - e)
            alt = (rp - rad) / 1000

            print("Turning angle:       %10.4f deg" % round(ta * RAD2DEG, 4))
            print("Periapsis altitude:  %10.4f km " % round(alt, 4))
            print("dV needed:           %10.4f m/s" % round(self.f[i], 4))
            print("GMAT MJD:             ", pk.epoch(self.t[i], "mjd").mjd - 29999.5)
            print("GMAT RadPer:          ", rp / 1000)

            c3 = dot(self.vi[i], self.vi[i]) / 1000000
            dha = atan2(-self.vi[i][2], sqrt(self.vi[i][0] * self.vi[i][0] + self.vi[i][1] * self.vi[i][1])) * RAD2DEG
            rha = atan2(-self.vi[i][1], -self.vi[i][0]) * RAD2DEG
            if rha < 0:
                rha = 360 + rha

            print("GMAT IncomingC3:      ", c3)
            print("GMAT IncomingRHA:     ", rha)
            print("GMAT IncomingDHA:     ", dha)

            e = cross([0, 0, 1], -self.vi[i])
            e = e / linalg.norm(e)
            n = cross(-self.vi[i], e)
            n = n / linalg.norm(n)
            h = cross(self.vi[i], self.vo[i])
            b = cross(h, -self.vi[i])
            b = b / linalg.norm(b)
            sinb = dot(b, e)
            cosb = dot(b, -n)
            bazi = atan2(sinb, cosb) * RAD2DEG
            if bazi < 0:
                bazi = bazi + 360
            print("GMAT IncomingBVAZI:   ", bazi)

            c3 = dot(self.vo[i], self.vo[i]) / 1000000
            dha = atan2(self.vo[i][2], sqrt(self.vo[i][0] * self.vo[i][0] + self.vo[i][1] * self.vo[i][1])) * RAD2DEG
            rha = atan2(self.vo[i][1], self.vo[i][0]) * RAD2DEG
            if rha < 0:
                rha = 360 + rha

            print("GMAT OutgoingC3:      ", c3)
            print("GMAT OutgoingRHA:     ", rha)
            print("GMAT OutgoingDHA:     ", dha)

            e = cross([0, 0, 1], self.vo[i])
            e = e / linalg.norm(e)
            n = cross(self.vo[i], e)
            n = n / linalg.norm(n)
            h = cross(self.vi[i], self.vo[i])
            b = cross(h, self.vo[i])
            b = b / linalg.norm(b)
            sinb = dot(b, e)
            cosb = dot(b, -n)
            bazi = atan2(sinb, cosb) * RAD2DEG
            if bazi < 0:
                bazi = bazi + 360
            print("GMAT OutgoingBVAZI:   ", bazi)

            print("")

        print("%8s arrival" % self.flight_plan[-1])
        print("--------------------------------------")
        print("MJD:                  %10.4f    " % round(pk.epoch(self.t[-1], "mjd").mjd, 4))
        print("Solution number:      %10d " % (1 + self.li_sol[-1] + 1))
        print("Hyp. excess velocity: %10.4f m/s" % round(sqrt(dot(self.vi[-1], self.vi[-1])), 4))
        print("Orbit insertion burn  %10.4f m/s - C3 = 0" % round(self.f[-1], 4))

        c3 = dot(self.vi[-1], self.vi[-1]) / 1000000
        dha = atan2(self.vi[-1][2], sqrt(self.vi[-1][0] * self.vi[-1][0] + self.vi[-1][1] * self.vi[-1][1])) * RAD2DEG
        rha = atan2(self.vi[-1][1], self.vi[-1][0]) * RAD2DEG

        print("GMAT MJD:             ", pk.epoch(self.t[-1], "mjd").mjd - 29999.5)
        print("GMAT IncomingC3:      ", c3)
        print("GMAT IncomingRHA:     ", rha)
        print("GMAT IncomingDHA:     ", dha)
        print("")

        print("--------------------------------------")
        print("Total fuel cost:     %10.4f m/s" % round(self.f_all, 4))

