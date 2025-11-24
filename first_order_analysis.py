# import modules

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D

import utils

import ambiance
import sys
import itertools

# create classes

class Vector:
    """Holder for vectorised forms of the utils functions."""
    # vectorise utils functions for convenience
    p = np.vectorize(
        lambda val: utils.stagnation_pressure_ratio(val)
    )
    p_invert = np.vectorize(
        lambda val: np.nan 
        if (val is np.ma.masked) or not np.isfinite(val)
        else utils.invert(utils.stagnation_pressure_ratio, val)
    )
    T = np.vectorize(
        lambda val: utils.stagnation_temperature_ratio(val)
    )
    v_cpT0 = np.vectorize(
        lambda val: utils.velocity_function(val)
    )
    m = np.vectorize(
        lambda val: utils.mass_flow_function(val)
    )
    m_invert = np.vectorize(
        lambda val: np.nan 
        if (val is np.ma.masked) or not np.isfinite(val)
        else utils.invert(utils.mass_flow_function, val)
    )

class Constants:
    """Stores constants."""
    # thermodynamic constants
    c_p = 1005
    gamma = 1.4
    gamma_ratio = (gamma - 1) / gamma
    R = 287

    # code parameters
    N = 100

    # fixed values
    hub_tip_ratio = 0.3
    thrust_target = 100

class Analysis:
    """Contains grids of values used for analysis."""
    def __init__(self, power, rpm, area, altitude, N_stages = 1, N_engines = 1):
        """Creates instance of Analysis class."""
        # store input variables
        self.power = power
        self.rpm = rpm
        self.area = area
        self.altitude = altitude
        self.N_stages = N_stages
        self.N_engines = N_engines

        # create arrays of Mach numbers
        x = np.linspace(0, 1, Constants.N)
        y = np.logspace(-4, 0, Constants.N)
        self.M_flight, self.M_1 = np.meshgrid(x, y)

        # include effects of atmosphere
        atmosphere = ambiance.Atmosphere(self.altitude)
        self.T_atm = atmosphere.temperature[0]
        self.p_atm = atmosphere.pressure[0]
        self.rho_atm = atmosphere.density[0]
        self.a_atm = atmosphere.speed_of_sound[0]

        # calculate mean radius
        self.r_mean = (
            np.sqrt(
                self.area * (1 + Constants.hub_tip_ratio)
                / (4 * np.pi * (1 - Constants.hub_tip_ratio))
            )
        )

        # convert units of motor speed
        self.omega = self.rpm * 2 * np.pi / 60

        # calculate target thrust coefficient
        self.C_th_target = Constants.thrust_target / (self.area * self.p_atm)
        
        # create cycle of colours
        self.colour_cycle = itertools.cycle(plt.cm.tab10.colors)

    def __str__(self):
        """Prints a string representation of the flight scenario."""
        string = f"\n"
        for name, value in self.__dict__.items():

            if isinstance(value, (int, float)):

                if ("alpha" in name or "angle" in name or "beta" in name) and not value == 0:

                    string += f"{name}: {utils.Colours.GREEN}{utils.rad_to_deg(value):.4g} °"
                    string += f"{utils.Colours.END}\n"

                else:

                    string += f"{name}: {utils.Colours.GREEN}{value:.4g}{utils.Colours.END}\n"

            if isinstance(value, str):

                string += f"{name}: {utils.Colours.GREEN}{value}{utils.Colours.END}\n"

        return string

    def analyse(self):
        """Analyses given parameters."""
        # find flow coefficient
        self.phi = (
            self.M_1 * np.sqrt(
                Constants.gamma * Constants.R * Vector.T(self.M_1) * self.T_atm / Vector.T(self.M_flight)
            ) / (self.omega * self.r_mean)
        )

        # find mass flow rate
        p_0_1 = self.p_atm / Vector.p(self.M_flight)
        T_0_1 = self.T_atm / Vector.T(self.M_flight)
        m_cpT0_Ap0_1 = Vector.m(self.M_1)
        self.m_dot = m_cpT0_Ap0_1 * self.area * p_0_1 / np.sqrt(Constants.c_p * T_0_1)

        # find stage loading coefficient
        self.psi = self.power / (self.m_dot * self.omega**2 * self.r_mean**2)

        # find stagnation temperature ratio
        T_03_T_01 = 1 + self.N_stages * self.psi * Vector.v_cpT0(self.M_1)**2 / self.phi**2

        # find compressor exit Mach number
        m_cpT0_Ap0_3 = (
            m_cpT0_Ap0_1 * np.power(1 / T_03_T_01, 1 / Constants.gamma_ratio)
            * np.sqrt(T_03_T_01)
        )
        self.M_3 = Vector.m_invert(m_cpT0_Ap0_3)

        # find jet Mach number via jet boundary condition
        self.M_j = (
            Vector.p_invert(
                Vector.p(self.M_flight) * np.power(1 / T_03_T_01, 1 / Constants.gamma_ratio)
            )
        )

        # find thrust coefficient
        self.C_th = (
            m_cpT0_Ap0_1 * (self.M_j - self.M_flight) * self.N_engines
            * np.sqrt((Constants.gamma - 1) * Vector.T(self.M_flight))
        )

        # find jet velocity ratio
        self.epsilon = self.M_flight / self.M_j

        # find propulsive efficiency
        self.eta_prop = 2 / (1 + 1 / self.epsilon)

        # find nozzle area ratio
        self.sigma = m_cpT0_Ap0_3 / Vector.m(self.M_j)

    def plot(self, label, max = 1, xx = None, yy = None, N_levels = 30):
        """Creates a contour plot of a given parameter."""
        # default x- and y- axes
        if xx is None:

            xx = self.M_flight

        if yy is None:

            yy = self.sigma

        # get attribute
        zz = getattr(self, label)

        # mask invalid data
        xx = np.nan_to_num(xx)
        yy = np.nan_to_num(yy)
        zz = np.ma.masked_invalid(zz)

        # create contour plot
        fig, ax = plt.subplots(figsize = (10, 6))
        norm = colors.Normalize(vmin = 0, vmax = max)
        levels = np.linspace(0, max, N_levels)
        contour = ax.contourf(xx, yy, zz, levels = levels, vmin = 0, vmax = max, alpha = 0.5)

        # configure plot
        colour_bar = fig.colorbar(contour, ax = ax)
        colour_bar.set_label(f"{label}")
        #colour_bar.set_clim(0, max)
        ax.plot([], [], linestyle = '', label = label)
        ax.set_xlabel("M_flight")
        ax.set_ylabel("Sigma")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2)
        ax.grid()
        ax.legend()
        ax.text(
            0.5, 1.02,
            f"Motor power: {self.power}W / Motor speed: {self.rpm}rpm / Area: {self.area}m^2 / "
            f"Stages: {self.N_stages} / Engines: {self.N_engines}",
            transform = ax.transAxes,
            ha = 'center',
            va = 'bottom',
            fontsize = 12
        )
        plt.tight_layout()

        return fig, ax

    def line(self, axis, label, value, linestyle = '-', xx = None, yy = None):
        """Plot a constant value line on a given axis."""
        # determine colour
        colour = next(self.colour_cycle)

        # default x- and y- axes
        if xx is None:

            xx = self.M_flight

        if yy is None:

            yy = self.sigma

        # get attribute
        zz = getattr(self, label)

        # plot line
        contour = axis.contour(
            xx,
            yy,
            zz,
            levels=[value],
            colors=[colour],
            linestyles=linestyle,
            linewidths=1.5,
        )

        # add legend entry
        axis.plot([], [], color = colour, label = f"{label} = {value}")
        axis.legend()

    def shade(self, axis, label_1, value_1, label_2, value_2, xx = None, yy = None):
        """Plots a shaded area on a given axis bounded by two contours."""
        # determine colour
        colour = next(self.colour_cycle)

        # default x- and y- axes
        if xx is None:

            xx = self.M_flight

        if yy is None:

            yy = self.sigma

        # get attributes
        zz_1 = getattr(self, label_1)
        zz_2 = getattr(self, label_2)

        # find contours
        contour_1 = axis.contour(xx, yy, zz_1, levels = [value_1], colors = [colour])
        contour_2 = axis.contour(xx, yy, zz_2, levels = [value_2], colors = [colour])

        # add legend entries
        axis.plot([], [], label = f"{label_1} > {value_1} / {label_2} < {value_2}", color = colour)

        # get contour paths
        paths_1 = contour_1.collections[0].get_paths()
        paths_2 = contour_2.collections[0].get_paths()

        # get contour coordinates
        coords_1 = paths_1[0].vertices
        coords_2 = paths_2[0].vertices

        # convert to (x, y) values
        x1, y1 = coords_1[:,0], coords_1[:,1]
        x2, y2 = coords_2[:,0], coords_2[:,1]

        # plot on common x-axis
        x_common = np.linspace(
            max(x1.min(), x2.min()),
            min(x1.max(), x2.max()),
            400
        )

        # interpolate y-values
        y1i = np.interp(x_common, x1, y1)
        y2i = np.interp(x_common, x2, y2)

        axis.legend()

        # fill bounded region
        """axis.fill_between(
            x_common,
            y1i,
            y2i,
            alpha=0.3,
            color="orange"
        )"""

def main():
    """Main function to run on script execution."""
    # create list of scenarios to analyse
    analyses = [
        Analysis(
            power = 3000,
            rpm = 12000,
            area = 0.03,
            altitude = 0,
            N_stages = 1,
            N_engines = 1
        ),
        Analysis(
            power = 3000,
            rpm = 12000,
            area = 0.03,
            altitude = 0,
            N_stages = 1,
            N_engines = 3
        ),
        Analysis(
            power = 3000,
            rpm = 12000,
            area = 0.03,
            altitude = 0,
            N_stages = 3,
            N_engines = 1
        )
    ]

    # loop over all scenarios
    for analysis in analyses:

        # analyse and create plots
        print(analysis)
        analysis.analyse()
        _, ax = analysis.plot('C_th', analysis.C_th_target)
        analysis.shade(ax, 'psi', 0.05, 'psi', 0.25)
        analysis.shade(ax, 'phi', 0.2, 'phi', 1)
        _, ax = analysis.plot('M_j', 1)
        analysis.shade(ax, 'psi', 0.05, 'psi', 0.25)
        analysis.shade(ax, 'phi', 0.2, 'phi', 1)

    # show plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    """Run main()."""
    if len(sys.argv) > 1:

        try:

            N = int(sys.argv[-1])
            Constants.N = N

        except Exception as error:

            print(error)
            print(f"{utils.Colours.RED}Please provide an integer argument!{utils.Colours.END}")

    main()
