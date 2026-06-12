# import modules
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import make_interp_spline
from scipy.integrate import cumulative_simpson

# import high speed solver modules
#from streamtube import Streamtube
#from flow_state import Flow_state
from annulus import Annulus
#from coefficients import Coefficients
from thick_aerofoils import Aerofoils
import utils
from time import perf_counter as timer

import matplotlib.pyplot as plt

# define Blade_row class
class Blade_row:
    """Represents a single row of blades (rotor or stator) and their associated parameters."""
    def __init__(self):
        """Creates an instance of the Blade_row class."""
        # assign the default colour of black
        self.colour = 'k'

        # create empty inlet and exit Annulus instances
        self.inlet = Annulus()
        self.exit = Annulus()

        # set default power and rpm as zero
        self.motor_power = 0
        self.motor_rpm = 0

    def __str__(self):
        """Prints a string representation of the blade row."""
        string = f"{self.label}\n"
        for name, value in self.__dict__.items():

            if isinstance(value, (int, float)):

                if ("alpha" in name or "angle" in name or "beta" in name) and not value == 0:

                    string += f"{name}: {utils.Colours.GREEN}{utils.rad_to_deg(value):.4g} °"
                    string += f"{utils.Colours.END}\n"

                else:

                    string += f"{name}: {utils.Colours.GREEN}{value:.4g}{utils.Colours.END}\n"

        string += "\n"

        return string

# plotting functions ------------------------------------------------------------------------------

    def draw_blades(self, max_thickness, thickness_fraction):
        """Creates a series of x- and y- coordinates based on the blade shape data."""
        # read in thickness distribution
        aerofoils = Aerofoils()
        self.zz = aerofoils.thick_aerofoil(thickness_fraction)

        # set number of sections to consider
        N = 5

        # initiialise empty arrays of x- and y-coordinates
        self.xx = np.zeros((N, 2 * len(self.zz[0])))
        self.yy = np.zeros((N, 2 * len(self.zz[0])))
        self.xx_camber = np.zeros((N, len(self.zz[0])))
        self.yy_camber = np.zeros((N, len(self.zz[0])))
        self.ll_camber = np.zeros((N, len(self.zz[0])))

        # get indices corresponding to hub, mid-span and tip
        theta = np.linspace(0, np.pi, N)
        fractions = 0.5 * (1.0 - np.cos(theta))
        rr = self.exit.rr[0] + fractions * (self.exit.rr[-1] - self.exit.rr[0])
        #rr = np.linspace(self.exit.rr[0], self.exit.rr[-1], N)
        indices = np.interp(rr, self.exit.rr, np.arange(len(self.exit.rr)))
        self.indices = np.round(indices).astype(int).tolist()

        # loop for each spanwise position
        for j, index in enumerate(self.indices):

            # find centre-point of circular camberline
            r = 1 / (self.exit.metal_angle[index] - self.inlet.metal_angle[index])
            x0 = -r * np.sin(self.inlet.metal_angle[index])
            y0 =  r * np.cos(self.inlet.metal_angle[index])

            # get list of angular values concentrated around leading edge
            theta = (
                self.inlet.metal_angle[index]
                + (self.exit.metal_angle[index] - self.inlet.metal_angle[index])
                * np.linspace(0, 1, len(self.zz[0]))**2
            )

            # get camber line coordinates
            self.xx_camber[j] = x0 + r * np.sin(theta)
            self.yy_camber[j] = y0 - r * np.cos(theta)

            # calculate cumulative distance along camber line
            self.ll_camber[j] = np.concatenate([[0], np.cumsum(
                np.sqrt(np.diff(self.xx_camber[j])**2 + np.diff(self.yy_camber[j])**2)
            )])

            # differentiate camber line with respect to cumulative distance
            dx_dl = np.gradient(self.xx_camber[j], self.ll_camber[j])
            dy_dl = np.gradient(self.yy_camber[j], self.ll_camber[j])

            # calculate components of vector normal to camber line
            norm = np.sqrt(dx_dl**2 + dy_dl**2)
            nx = -dy_dl / norm
            ny =  dx_dl / norm

            # read in thickness data
            zz = self.zz[:, self.zz[0].argsort()]

            # initialise empty arrays for upper- and lower-surface data
            xx_upper = np.zeros(self.xx_camber[j].shape)
            xx_lower = np.zeros(self.xx_camber[j].shape)
            yy_upper = np.zeros(self.xx_camber[j].shape)
            yy_lower = np.zeros(self.xx_camber[j].shape)

            # loop over each point in the camber line
            for i, (x, y, l) in enumerate(zip(self.xx_camber[j], self.yy_camber[j], self.ll_camber[j])):

                # get relevant thickness for that point along the camber line
                dy = np.interp(l, *zz)

                # scale thickness by axial chord length
                dy *= max_thickness / (0.09 * self.exit.chord[index])

                # add upper and lower surfaces
                xx_upper[i] = x + dy * nx[i]
                yy_upper[i] = y + dy * ny[i]
                xx_lower[i] = x - dy * nx[i]
                yy_lower[i] = y - dy * ny[i]

            # reverse upper surfaces
            xx_upper = xx_upper[::-1]
            yy_upper = yy_upper[::-1]

            # combine surfaces and store in 2D matrices
            self.xx[j] = np.concatenate([xx_upper, xx_lower])
            self.yy[j] = np.concatenate([yy_upper, yy_lower])

            # resize according to chord length
            self.xx[j] *= self.exit.chord[index]
            self.yy[j] *= self.exit.chord[index]

    def calculate_mass(self, radius):
        """Calculates the mass of a blade row."""
        # helper function for calcuating area of a polygon
        def polygon_area(x, y):
            """Returns the area of a polygon of x-y values via the shoelace method."""
            # calculate and return area
            A = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            return A
        
        # calculate index of mid-span
        mid_index = int(np.floor(utils.Defaults.solver_grid / 2))

        # calculate section areas at hub mid-span and tip
        section_areas = [polygon_area(xx, yy) for (xx, yy) in zip(self.xx, self.yy)]

        # approximate section area distribution by ratio of chords
        A_hub = section_areas[0] * self.exit.chord / self.exit.chord[0]
        A_mid = section_areas[1] * self.exit.chord / self.exit.chord[mid_index]
        A_tip = section_areas[2] * self.exit.chord / self.exit.chord[-1]

        # express in terms of span
        span = (self.exit.rr - self.exit.rr[0]) / (self.exit.rr[-1] - self.exit.rr[0])
        mid_span = span[mid_index]

        # find coefficients for calculating weighted average of area distributions
        w_mid = span / mid_span
        w_tip = (span - mid_span) / (1 - mid_span)

        # blend area distributions according to piecewise weighted average
        A_blend = np.where(
            span <= mid_span,
            (1 - w_mid) * A_hub + w_mid * A_mid,
            (1 - w_tip) * A_mid + w_tip * A_tip
        )

        # calculate dimensionless blade volume
        blade_volume = utils.cumulative_trapezoid(self.exit.rr, A_blend)[-1]

        #
        if self.motor_rpm > 0:

            rho = utils.rho_aluminium

        else:

            rho = utils.rho_PLA

        # calculate (dimensional, in kg) blade mass
        self.mass = blade_volume * radius**3 * rho * self.no_of_blades

    def calculate_stress(self):

        pass
