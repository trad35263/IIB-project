# import modules

import numpy as np

from streamtube import Streamtube
from flow_state import Flow_state
import utils

# define Blade_row class

class Blade_row:
    """
    Represents a single row of blades (i.e. a rotor or a stator) and their associated parameters.
    
    Used to investigate the flow across a Rotor or a Stator. Stator is a special case of the
    Blade_row class where the blade velocity is zero. Every instance of the class will
    contain an inlet and exit flow state where all of the flow properties are stored.
    
    Parameters
    ----------
    casing_area_ratio : float
        Ratio of blade row casing area to a reference area.
    hub_area_ratio : float
        Ratio of blade row hub area to a reference area.
    Y_p : float
        Stagnation pressure loss coefficient.
    inlet : Flow_state
        Container to store inlet fluid conditions.
    exit : Flow_state
        Container to store exit fluid conditions.
    """
    def __init__(self, casing_area_ratio, hub_area_ratio, Y_p, is_rotor):
        """Create instance of the Blade_row class."""
        # assign attributes
        self.casing_area_ratio = casing_area_ratio
        self.hub_area_ratio = hub_area_ratio
        self.Y_p = Y_p

        # initialise inlet and exit variables to store a list of streamtubes
        self.inlet = None
        self.exit = None

        # assign the default colour of black
        self.colour = 'k'

        # categorise blade row
        self.categorise(is_rotor)

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

    def categorise(self, is_rotor):
        """Categorise blade row as Rotor or Stator."""
        # identify rotors
        if is_rotor:

            self.label = f"{utils.Colours.ORANGE}Rotor{utils.Colours.END}"
            self.short_label = f"{utils.Colours.ORANGE}R{utils.Colours.END}"

        # identify stators
        else:

            self.label = f"{utils.Colours.YELLOW}Stator{utils.Colours.END}"
            self.short_label = f"{utils.Colours.YELLOW}S{utils.Colours.END}"
    
    def set_inlet_conditions(self, M, alpha):
        """Distributes the given inlet conditions across several annular streamtubes."""
        self.inlet = []
        for index in range(utils.Defaults.no_of_annuli):
            
            if index == 0:

                r = (
                    (utils.Defaults.hub_tip_ratio + np.sqrt(
                        utils.Defaults.hub_tip_ratio**2 * (1 - 1 / utils.Defaults.no_of_annuli)
                        + 1 / utils.Defaults.no_of_annuli
                    )) / 2
                )
                dr = r - utils.Defaults.hub_tip_ratio
                flow_state = Flow_state(
                    M, alpha, 1, 1
                )
            
            else:

                r = (
                    (self.inlet[index - 1].r + self.inlet[index - 1].dr + np.sqrt(
                        (self.inlet[index - 1].r + self.inlet[index - 1].dr)**2
                        - (utils.Defaults.hub_tip_ratio**2 - 1) / utils.Defaults.no_of_annuli
                    )) / 2
                )
                dr = r - self.inlet[index - 1].r - self.inlet[index - 1].dr
                flow_state = Flow_state(
                    M, alpha, 1, 1
                )

            self.inlet.append(Streamtube(flow_state, r, dr))

    def solve_blade_row(self):
        """Calculates conditions at outlet to the blade row, given the inlet conditions."""
        # consider stator case
        #if self.blade_speed_ratio == 0:
        if not self.is_rotor:
            
            # find stagnation pressure ratio after stagnation pressure loss
            stagnation_pressure_ratio = (
                1 - self.Y_p * (1 - utils.stagnation_pressure_ratio(self.inlet.M))
            )

            # assume no deviation
            exit_alpha = self.exit_blade_angle

            # find exit Mach number via compressible flow relations
            exit_mass_flow_function = (
                utils.mass_flow_function(self.inlet.M) / stagnation_pressure_ratio
                * np.cos(self.inlet.alpha) / np.cos(exit_alpha)
            )
            exit_M = utils.invert(utils.mass_flow_function, exit_mass_flow_function)
            # do we need to catch errors here?

            # find exit flow coefficient via conservation of mass
            #exit_phi = (
            #    self.inlet.phi * utils.stagnation_density_ratio(self.inlet.M)
            #    / (utils.stagnation_density_ratio(exit_M) * stagnation_pressure_ratio)
            #)

            # stagnation temperature is conserved across stator row
            T_0 = self.inlet.T_0

            # stagnation pressure ratio is known from previously
            p_0 = self.inlet.p_0 * stagnation_pressure_ratio

        # consider rotor case
        else:

            # adjust flow coefficient to new blade speed
            self.inlet.phi = self.inlet.M * np.cos(self.inlet.alpha) / self.inlet_blade_Mach_number
            #self.inlet.phi = (
            #    self.inlet.phi * np.sqrt(1 / self.casing_area_ratio) / self.blade_speed_ratio
            #)

            # solve for inlet relative Mach number via vector algebra
            v1 = self.inlet.M * np.array([np.cos(self.inlet.alpha), np.sin(self.inlet.alpha)])
            v2 = self.inlet.M * np.cos(self.inlet.alpha) / self.inlet.phi * np.array([0, 1])
            v3 = v1 - v2
            self.inlet.M_rel = np.linalg.norm(v3)

            # find relative stagnation pressure ratio after stagnation pressure loss
            relative_stagnation_pressure_ratio = (
                1 - self.Y_p * (1 - utils.stagnation_pressure_ratio(self.inlet.M_rel))
            )

            # find exit relative Mach number via compressible flow relations
            exit_relative_mass_flow_function = (
                utils.mass_flow_function(self.inlet.M_rel) / relative_stagnation_pressure_ratio
                * (self.inlet.M * np.cos(self.inlet.alpha))
                / (np.cos(-self.exit_blade_angle) * self.inlet.M_rel)
            )
            exit_M_rel = (
                utils.invert(utils.mass_flow_function, exit_relative_mass_flow_function)
            )
            # do we need to catch exit_M_rel == None errors here?

            # find exit flow coefficient via conservation of mass
            exit_phi = (
                self.inlet.phi * utils.stagnation_density_ratio(self.inlet.M_rel)
                / utils.stagnation_density_ratio(exit_M_rel)
                / relative_stagnation_pressure_ratio
            )

            # find exit swirl angle via vector algebra
            v1 = exit_M_rel * np.array([np.cos(self.exit_blade_angle), np.sin(self.exit_blade_angle)])
            v2 = exit_M_rel * np.cos(self.exit_blade_angle) / exit_phi * np.array([0, 1])
            v3 = v1 + v2
            exit_M = np.linalg.norm(v3)
            exit_alpha = np.arctan2(v3[1], v3[0])

            # find stagnation temperature ratio
            T_0 = (
                self.inlet.T_0 / utils.stagnation_temperature_ratio(exit_M)
                * utils.stagnation_temperature_ratio(exit_M_rel)
                * utils.stagnation_temperature_ratio(self.inlet.M)
                / utils.stagnation_temperature_ratio(self.inlet.M_rel)
            )

            # find stagnation pressure ratio
            p_0 = (
                self.inlet.p_0
                / utils.stagnation_pressure_ratio(exit_M)
                * utils.stagnation_pressure_ratio(exit_M_rel)
                * utils.stagnation_pressure_ratio(self.inlet.M)
                / utils.stagnation_pressure_ratio(self.inlet.M_rel)
                * relative_stagnation_pressure_ratio
            )

        # return instance of Flow_state class corresponding to exit conditions
        self.exit = Flow_state(
            exit_M,
            #exit_phi,
            exit_alpha,
            T_0,
            p_0
        )

    def modify_blade_row(self):
        """Iterates over all blade row properties, offering the user to change each value."""
        # iterate over all name-value pairs associated with the class
        for name, value in list(self.__dict__.items()):

            # ignore any attributes that are not numeric
            if isinstance(value, (int, float)):

                # for angles, ask for input in degrees and convert to radians internally
                if ("alpha" in name or "angle" in name) and not value == 0:

                    print(f"{utils.Colours.RED}Please state the new {name} (in °):{utils.Colours.END}")
                    while True:

                        user_input = input()
                        if user_input == "":

                            break

                        try:

                            setattr(self, name, utils.deg_to_rad(float(user_input)))
                            break

                        except ValueError:

                            print(f"{utils.Colours.RED}Error: Please provide a valid number.{utils.Colours.END}")

                    print(f"{utils.Colours.GREEN}{name} of {utils.rad_to_deg(getattr(self, name)):.3g} ° selected!{utils.Colours.END}")

                # for non-dimensional parameters
                else:

                    print(f"{utils.Colours.RED}Please state the new {name}:{utils.Colours.END}")
                    while True:

                        user_input = input()
                        if user_input == "":

                            break

                        try:

                            setattr(self, name, float(user_input))
                            break

                        except ValueError:

                            print(f"{utils.Colours.RED}Error: Please provide a valid number.{utils.Colours.END}")

                    print(f"{utils.Colours.GREEN}{name} of {getattr(self, name):.3g} selected!{utils.Colours.END}")

        # re-categorise blade row in case blade speed has changed
        #self.categorise()