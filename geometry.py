# import modules
import numpy as np
import utils

###     DEPRECATED

# create Geometry class
class Geometry:
    """
    Docstring for Geometry
    """
    def __init__(self, aspect_ratio, diffusion_factor, deviation_constant):
        """Creates an instance of the Geometry class."""
        # store input variables
        self.aspect_ratio = aspect_ratio
        self.diffusion_factor = diffusion_factor
        self.deviation_constant = deviation_constant
