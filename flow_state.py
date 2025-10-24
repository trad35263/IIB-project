# import modules

import utils

# define Flow_state class

class Flow_state:
    """
    Stores the flow properties at a given point in the system.
    
    Given two thermodynamic properties, the remaining properties can be determined via perfect gas
    relations.

    Parameters
    ----------
    M : float
        Ratio of local velocity to local speed of sound.
    alpha : float
        Angle between the velocity vector and the axial direction.
    p_0_ratio : float
        Ratio of stagnation pressure at current position in system to the inlet condition.
    T_0_ratio : float
        Ratio of stagnation temperature at current position in system to the inlet condition.
    """
    def __init__(self, M, alpha, T_0, p_0, M_rel = None, beta = None):
        self.M = M
        self.alpha = alpha
        self.T_0 = T_0
        self.p_0 = p_0
        self.M_rel = M_rel
        self.beta = beta

    def __str__(self):
        """Prints a string representation of the flow state."""
        string = ""
        for name, value in self.__dict__.items():

            if isinstance(value, (int, float)):

                if ("alpha" in name or "angle" in name or "beta" in name) and not value == 0:

                    string += f"{name}: {utils.Colours.GREEN}{utils.rad_to_deg(value):.4g} °"
                    string += f"{utils.Colours.END}\n"

                else:

                    string += f"{name}: {utils.Colours.GREEN}{value:.4g}{utils.Colours.END}\n"

        string += "\n"

        return string