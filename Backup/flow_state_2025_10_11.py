# import modules

import numpy as np

import utils_2025_10_11 as utils

# 1.2 define Flow_state class

class Flow_state:
    """
    Stores the flow properties at a given point in the system.
    
    Given two thermodynamic properties, the remaining properties can be determined via perfect gas
    relations.

    Parameters
    ----------
    M : float
        Ratio of local velocity to local speed of sound.
    phi : float
        Ratio of local axial velocity to stage blade speed.
    alpha : float
        Angle between the velocity vector and the axial direction.
    p_0_ratio : float
        Ratio of stagnation pressure at current position in system to the inlet condition.
    T_0_ratio : float
        Ratio of stagnation temperature at current position in system to the inlet condition.

    Public methods
    --------------
    relative_quantities(self, blade_speed): asdf
    """
    def __init__(self, M, phi, alpha, T_0_ratio, p_0_ratio):
        self.M = M
        self.phi = phi
        self.alpha = alpha
        self.T_0_ratio = T_0_ratio
        self.p_0_ratio = p_0_ratio
        """Create instance of the Flow_state class and store velocities and flow properties."""
        """
        # input velocities
        self.axial_velocity = axial_velocity
        self.tangential_velocity = tangential_velocity
        self.velocity = np.sqrt(self.axial_velocity**2 + self.tangential_velocity**2)

        # static quantities
        self.p = p
        self.T = T
        self.h = c_p * self.T
        self.rho = self.p / (R * self.T)

        # Mach quantities
        self.a = np.sqrt(gamma * R * self.T)
        self.M = self.velocity / self.a

        # stagnation quantities
        self.p_0 = self.p / stagnation_pressure_ratio(self.M, gamma)
        self.T_0 = self.T / stagnation_temperature_ratio(self.M, gamma)
        self.h_0 = c_p * self.T_0
        """

    def __str__(self):
        """Prints a string representation of the flow state."""
        string = ""
        for name, value in self.__dict__.items():

            if isinstance(value, (int, float)):

                if ("alpha" in name or "angle" in name) and not value == 0:

                    string += f"{name}: {utils.Colours.GREEN}{utils.rad_to_deg(value):.4g} °"
                    string += f"{utils.Colours.END}\n"

                else:

                    string += f"{name}: {utils.Colours.GREEN}{value:.4g}{utils.Colours.END}\n"

        string += "\n"

        return string

    def relative_quantities(self, blade_speed):
        """Computes various quantities relative to a given blade speed."""
        # compute relative velocities and swirl angle
        self.relative_tangential_velocity = self.tangential_velocity - blade_speed
        self.relative_velocity = (
            np.sqrt(self.axial_velocity**2 + self.relative_tangential_velocity**2)
        )
        self.relative_swirl_angle = np.arctan(self.relative_tangential_velocity / self.axial_velocity)

        # compute relative stagnation properties and Mach number
        self.M_rel = self.relative_velocity / self.a
        self.p_0_rel = self.p / utils.stagnation_pressure_ratio(self.M_rel, utils.gamma)
        self.T_0_rel = self.T / utils.stagnation_temperature_ratio(self.M_rel, utils.gamma)