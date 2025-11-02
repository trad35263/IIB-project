# import modules

import numpy as np

import utils
from flow_state import Flow_state

# create Streamtube class

class Streamtube:
    """
    Used to store an instance of a streamtube at a given axial position in the engine.
    
    Facilitates 2D flow analysis throughout the engine, storing the corresponding flow state at
    that radial position as well as information relating to the size and position of the annulus of
    the streamtube. Continuity can be applied along a streamtube due to the no through-flow
    condition across streamlines.
    
    Parameters
    ----------
    None
    """
    def __init__(self, flow_state, r, dr):
        """Creates an instance of the Streamtube class."""
        # store input variables
        self.flow_state = flow_state
        self.r = r
        self.dr = dr

        # calculate annulus area
        self.A = 4 * np.pi * self.r * self.dr
        
    def __str__(self):
        """Prints a string representation of the streamtube."""
        string = ""
        for name, value in self.__dict__.items():

            if isinstance(value, (int, float)):

                if ("alpha" in name or "angle" in name or "beta" in name) and not value == 0:

                    string += f"{name}: {utils.Colours.GREEN}{utils.rad_to_deg(value):.4g} °"
                    string += f"{utils.Colours.END}\n"

                else:

                    string += f"{name}: {utils.Colours.GREEN}{value:.4g}{utils.Colours.END}\n"

        return string