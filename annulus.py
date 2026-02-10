# import modules

import numpy as np
import utils

# import classes

from coefficients import Coefficients

# create Annulus class

class Annulus:
    """"""
    def __init__(self):
        """Creates an instance of the Annulus class."""
        # initialise empty Coefficients instances for each primary flow variable
        self.M = Coefficients()
        self.alpha = Coefficients()
        self.T_0 = Coefficients()
        self.p_0 = Coefficients()
        self.s = Coefficients()
        self.T = Coefficients()
        self.p = Coefficients()

    def set_inlet_conditions(self, M, alpha, N):
        """Sets the inlet conditions for the first stage rotor."""
        # create Coefficients instances for each quantity
        self.M.coefficients = np.r_[np.zeros(N - 1, ), M]
        self.alpha.coefficients = np.r_[np.zeros(N - 1, ), alpha]
        self.T_0.coefficients = np.r_[np.zeros(N - 1, ), 1]
        self.p_0.coefficients = np.r_[np.zeros(N - 1, ), 1]
        self.s.coefficients = np.zeros(N, )

        # define radial node positions
        self.r_hub = utils.Defaults.hub_tip_ratio
        self.r_casing = 1
        self.r_mean = 0.5 * (self.r_hub + self.r_casing)
        self.rr = np.linspace(self.r_hub, self.r_casing, utils.Defaults.solver_grid)

        # define spanwise distributions of primary flow variables
        self.M.value = np.polyval(self.M.coefficients, self.rr)
        self.alpha.value = np.polyval(self.alpha.coefficients, self.rr)
        self.T_0.value = np.polyval(self.T_0.coefficients, self.rr)
        self.p_0.value = np.polyval(self.p_0.coefficients, self.rr)
        self.s.value = np.polyval(self.s.coefficients, self.rr)

        # store static properties
        self.T.value = self.T_0.value * utils.stagnation_temperature_ratio(self.M.value)
        self.p.value = self.p_0.value * utils.stagnation_pressure_ratio(self.M.value)

    def value(self, quantity):
        """Returns the spanwise distribution of a given quantity."""
        # get relevant Coefficients instance
        coefficients = getattr(self, quantity)

        # check if value has already been assigned
        if hasattr(coefficients, "value"):

            # return existing array of values
            return coefficients.value
        
        # value has not been assigned yet
        else:

            # assign value and return
            coefficients.value = np.polyval(coefficients.coefficients, self.rr)
            return coefficients.value
