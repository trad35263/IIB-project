# import modules

import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

from flow_state_2025_10_11 import Flow_state
import utils_2025_10_11 as utils

# create Nozzle class

class Nozzle:
    """
    Represents an area contraction/expansion and solves for the conditions at inlet and outlet.

    Used to model the bellmouth inlet to the engine, the exit nozzle, and inter-blade row passages
    through which there may be a change in area. Assumptions made include conservation of energy,
    isentropic process, conservation of angular momentum and compressible continuity.
    
    Parameters
    ----------
    area_ratio : float
        Ratio of outlet area to inlet area.
    inlet : object or None
        Inlet flow state (to be defined externally).
    exit : object or None
        Exit flow state (to be defined externally).
    """
    def __init__(self, area_ratio, inlet=None):
        """Create instance of the Nozzle class."""
        self.area_ratio = area_ratio
        self.inlet = inlet
        self.exit = None
        self.label = f"{utils.Colours.PURPLE}Nozzle{utils.Colours.END}"
        self.colour = 'k'

    def __str__(self):
        """Prints a string representation of the nozzle."""
        string = f"{self.label}\n"
        for name, value in self.__dict__.items():

            if isinstance(value, (int, float)):

                if ("alpha" in name or "angle" in name) and not value == 0:

                    string += f"{name}: {utils.Colours.GREEN}{utils.rad_to_deg(value):.4g} °"
                    string += f"{utils.Colours.END}\n"

                else:

                    string += f"{name}: {utils.Colours.GREEN}{value:.4g}{utils.Colours.END}\n"

        string += "\n"

        return string

    def solve_nozzle(self):
        """Solve for conditions at outlet, given the inlet conditions, due to the area change."""
        # check if inlet conditions exist
        if self.inlet == None:

            self.inlet = Flow_state(
                utils.Defaults.inlet_Mach_number,
                utils.Defaults.inlet_flow_coefficient,
                utils.Defaults.inlet_swirl,
                1, 1
            )

        # calculate inlet mass flow function
        inlet_mass_flow_function = utils.mass_flow_function(self.inlet.M)

        # solve case where there is no inlet or exit swirl
        if self.inlet.alpha == 0:

            exit_mass_flow_function = inlet_mass_flow_function / self.area_ratio
            exit_M = utils.solve_M_from_mass_flow_function(exit_mass_flow_function)
            exit_alpha = 0
        
        # solve case where there is non-zero inlet swirl via conservation of angular momentum
        else:

            # define residual function to solve the root for
            def residual(exit_swirl_angle_guess):
                """Find swirl angle which reduces residual to zero to find Mach number solution."""
                # from an exit swirl angle guess, determine the corresponding exit Mach number
                exit_mass_flow_function_guess = (
                    inlet_mass_flow_function * np.cos(exit_swirl_angle_guess)
                    / np.cos(self.inlet.alpha)
                    / self.area_ratio
                )
                exit_M_guess = utils.solve_M_from_mass_flow_function(
                    exit_mass_flow_function_guess
                )

                # compare guess for exit Mach number with calculated value and return difference
                RHS = (
                    self.inlet.M * np.sin(self.inlet.alpha) / np.sin(exit_swirl_angle_guess)
                    * np.sqrt(1 / self.area_ratio)
                    * np.sqrt(utils.stagnation_temperature_ratio(self.inlet.M))
                    * np.sqrt(1 / utils.stagnation_temperature_ratio(exit_M_guess))
                )
                return exit_M_guess - RHS

            """
            # DEBUGGING
            xx = np.linspace(-np.pi / 4, np.pi / 4, 100)
            yy = [residual(x) for x in xx]
            fig, ax = plt.subplots()
            ax.plot(xx, yy, marker = '.', linestyle = '')
            plt.show()
            """

            # solve for root of residual function
            sol = root_scalar(
                residual, x0 = self.inlet.alpha, x1 = 0.9 * self.inlet.alpha, method = "secant"
            )
            exit_alpha = sol.root

            # recalculate corresponding exit Mach number
            exit_mass_flow_function = (
                inlet_mass_flow_function * np.cos(exit_alpha)
                / np.cos(self.inlet.alpha)
                / self.area_ratio
            )
            exit_M = utils.solve_M_from_mass_flow_function(
                exit_mass_flow_function
            )

        # determine exit flow coefficient via conservation of mass
        exit_phi = (
            self.inlet.phi * utils.stagnation_density_ratio(self.inlet.M)
            / utils.stagnation_density_ratio(exit_M)
            / self.area_ratio
        )

        # return instance of Flow_state class corresponding to exit conditions
        self.exit = Flow_state(
            exit_M,
            exit_phi,
            exit_alpha,
            self.inlet.T_0_ratio,
            self.inlet.p_0_ratio
        )