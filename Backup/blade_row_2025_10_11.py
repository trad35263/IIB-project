# import modules

import numpy as np

from flow_state_2025_10_11 import Flow_state
import utils_2025_10_11 as utils

# define Blade_row class

class Blade_row:
    """
    Represents a single row of blades (i.e. a rotor or a stator) and their associated parameters.
    
    Used to investigate the flow across a Rotor or a Stator. Stator is a special case of the
    Blade_row class where the blade velocity is zero. Every instance of the class will
    contain an inlet and exit flow state where all of the flow properties are stored.
    
    Parameters
    ----------
    blade_speed_ratio : float
        Mean-line rotating blade speed (0 for a stator).
    blade_angle : float [rad]
        Angle at which flow leaves the blade row.
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
    def __init__(self, blade_speed_ratio, blade_angle, casing_area_ratio, hub_area_ratio, Y_p):
        """Create instance of the Blade_row class."""
        # assign attributes
        self.blade_speed_ratio = blade_speed_ratio
        self.blade_angle = blade_angle
        self.casing_area_ratio = casing_area_ratio
        self.hub_area_ratio = hub_area_ratio
        self.Y_p = Y_p
        self.inlet = None
        self.exit = None

        # assign the default colour of black
        self.colour = 'k'

        # categorise blade row
        self.categorise()

    def categorise(self):
        """Categorise blade row as Rotor, Stator or Contra-Rotating."""
        # identify rotors
        if self.blade_speed_ratio > 0:

            self.label = f"{utils.Colours.ORANGE}Rotor{utils.Colours.END}"
            self.short_label = f"{utils.Colours.ORANGE}R{utils.Colours.END}"

        # identify stators
        elif self.blade_speed_ratio == 0:

            self.label = f"{utils.Colours.YELLOW}Stator{utils.Colours.END}"
            self.short_label = f"{utils.Colours.YELLOW}S{utils.Colours.END}"

        # all other cases must be counter-rotating
        else:

            self.label = f"{utils.Colours.PURPLE}Contra-Rotating{utils.Colours.END}"
            self.short_label = f"{utils.Colours.PURPLE}CR{utils.Colours.END}"

    def __str__(self):
        """Prints a string representation of the blade row."""
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

    def solve_blade_row(self):
        """Calculates conditions at outlet to the blade row, given the inlet conditions."""
        # consider stator case
        if self.blade_speed_ratio == 0:
            
            # find stagnation pressure ratio after stagnation pressure loss
            stagnation_pressure_ratio = (
                1 - self.Y_p * (1 - utils.stagnation_pressure_ratio(self.inlet.M))
            )

            # assume no deviation
            exit_alpha = self.blade_angle

            # find exit Mach number via compressible flow relations
            exit_mass_flow_function = (
                utils.mass_flow_function(self.inlet.M) / stagnation_pressure_ratio
                * np.cos(self.inlet.alpha) / np.cos(exit_alpha)
            )
            exit_M = utils.solve_M_from_mass_flow_function(exit_mass_flow_function)

            # find exit flow coefficient via conservation of mass
            exit_phi = (
                self.inlet.phi * utils.stagnation_density_ratio(self.inlet.M)
                / (utils.stagnation_density_ratio(exit_M) * stagnation_pressure_ratio)
            )

            # stagnation temperature is conserved across stator row
            T_0_ratio = self.inlet.T_0_ratio

            # stagnation pressure ratio is known from previously
            p_0_ratio = self.inlet.p_0_ratio * stagnation_pressure_ratio

        # consider rotor case
        else:

            # adjust flow coefficient to new blade speed
            self.inlet.phi = (
                self.inlet.phi * np.sqrt(1 / self.casing_area_ratio) / self.blade_speed_ratio
            )

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
                / (np.cos(-self.blade_angle) * self.inlet.M_rel)
            )
            exit_M_rel = (
                utils.solve_M_from_mass_flow_function(exit_relative_mass_flow_function)
            )

            # find exit flow coefficient via conservation of mass
            exit_phi = (
                self.inlet.phi * utils.stagnation_density_ratio(self.inlet.M_rel)
                / utils.stagnation_density_ratio(exit_M_rel)
                / relative_stagnation_pressure_ratio
            )

            # find exit swirl angle via vector algebra
            v1 = exit_M_rel * np.array([np.cos(self.blade_angle), np.sin(self.blade_angle)])
            v2 = exit_M_rel * np.cos(self.blade_angle) / exit_phi * np.array([0, 1])
            v3 = v1 + v2
            exit_M = np.linalg.norm(v3)
            exit_alpha = np.arctan2(v3[1], v3[0])

            # find stagnation temperature ratio
            T_0_ratio = (
                self.inlet.T_0_ratio / utils.stagnation_temperature_ratio(exit_M)
                * utils.stagnation_temperature_ratio(exit_M_rel)
                * utils.stagnation_temperature_ratio(self.inlet.M)
                / utils.stagnation_temperature_ratio(self.inlet.M_rel)
            )

            # find stagnation pressure ratio
            p_0_ratio = (
                self.inlet.p_0_ratio
                / utils.stagnation_pressure_ratio(exit_M)
                * utils.stagnation_pressure_ratio(exit_M_rel)
                * utils.stagnation_pressure_ratio(self.inlet.M)
                / utils.stagnation_pressure_ratio(self.inlet.M_rel)
                * relative_stagnation_pressure_ratio
            )

        # return instance of Flow_state class corresponding to exit conditions
        self.exit = Flow_state(
            exit_M,
            exit_phi,
            exit_alpha,
            T_0_ratio,
            p_0_ratio
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
        self.categorise()