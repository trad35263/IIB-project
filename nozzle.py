# import modules

import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

from flow_state import Flow_state
import utils

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
    def __init__(self, inlet=None):
        """Create instance of the Nozzle class."""
        self.area_ratio = 1
        self.inlet = inlet
        self.exit = None
        self.label = f"{utils.Colours.PURPLE}Nozzle{utils.Colours.END}"
        self.colour = 'k'

    def __str__(self):
        """Prints a string representation of the nozzle."""
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
    
    def define_nozzle_geometry(self, M_exit):
        """Determine area ratio required to achieve specified thrust coefficient."""

        # store inlet and exit mass flow functions for later
        inlet_mass_flow_function = utils.mass_flow_function(self.inlet.M)
        exit_mass_flow_function = utils.mass_flow_function(M_exit)

        # consider simple case where there is no inlet swirl
        if self.inlet.alpha == 0:

            # solve area ratio by conservation of mass
            self.area_ratio = inlet_mass_flow_function / exit_mass_flow_function
            alpha_exit = 0

        # consider case where there is inlet swirl and solve via conservation of angular momentum
        else:

            # apply conservation of mass
            A = inlet_mass_flow_function * np.cos(self.inlet.alpha) / exit_mass_flow_function

            # apply conservation of angular momentum
            B = (
                utils.stagnation_temperature_ratio(self.inlet.M)
                / utils.stagnation_temperature_ratio(M_exit)
                * (self.inlet.M * np.sin(self.inlet.alpha) / M_exit)
            )

            # form quadratic and solve - see working on ipad
            cos_alpha = (
                (
                    -B / A + np.sqrt((B / A)**2 + 4)
                ) / 2
            )

            # solve for corresponding area ratio
            self.area_ratio = (
                inlet_mass_flow_function * np.cos(self.inlet.alpha)
                / (cos_alpha * exit_mass_flow_function)
            )

            # ensure sign of swirl angle is the same through the nozzle
            alpha_exit = np.arccos(cos_alpha) * np.sign(self.inlet.alpha)

        # save exit flow at nozzle exit
        self.exit = Flow_state(
            M_exit,
            alpha_exit,
            self.inlet.T_0,
            self.inlet.p_0
        )


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
            exit_M = utils.invert(utils.mass_flow_function, exit_mass_flow_function)
            if exit_M == None:

                print(f"{utils.Colours.RED}Nozzle is choked: {self}{utils.Colours.END}")
                exit_M = 1

            #exit_M = utils.solve_M_from_mass_flow_function(exit_mass_flow_function)
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
                #exit_M_guess = utils.solve_M_from_mass_flow_function(
                #    exit_mass_flow_function_guess
                #)
                exit_M_guess = utils.invert(utils.mass_flow_function, exit_mass_flow_function_guess)
                if exit_M_guess == None:

                    return 1e3

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
            #exit_M = utils.solve_M_from_mass_flow_function(
            #    exit_mass_flow_function
            #)
            exit_M = utils.invert(utils.mass_flow_function, exit_mass_flow_function)
            if exit_M == None:

                print(f"{utils.Colours.RED}Nozzle is choked: {self}{utils.Colours.END}")
                exit_M = 1

        # determine exit flow coefficient via conservation of mass
        #exit_phi = (
        #    self.inlet.phi * utils.stagnation_density_ratio(self.inlet.M)
        #    / utils.stagnation_density_ratio(exit_M)
        #    / self.area_ratio
        #)

        # return instance of Flow_state class corresponding to exit conditions
        self.exit = Flow_state(
            exit_M,
            #exit_phi,
            exit_alpha,
            self.inlet.T_0,
            self.inlet.p_0
        )