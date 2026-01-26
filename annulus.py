# import modules

import numpy as np
import utils

# create Annulus class

class Annulus:
    """"""
    def __init__(self, r_min, r_max):
        """Creates an instance of the Annulus class."""
        # store input variables
        self.r_min = r_min
        self.r_max = r_max

        # calculate mean radius and fine grid of radial positions
        self.r_mean = 0.5 * (self.r_min + self.r_max)
        self.rr = np.linspace(self.r_min, self.r_max, utils.Defaults.fine_grid)
