# import modules

import numpy as np
from scipy.optimize import root
from scipy.optimize import least_squares

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
    is_rotor : boolean
        Reference to whether or not to categorise the blade row as a rotor or a stator.
    """
    def __init__(self, r_inlet, Y_p, is_rotor=False):
        """Create instance of the Blade_row class."""
        # assign attributes
        self.r_inlet = r_inlet
        self.Y_p = Y_p

        # derive inlet and exit areas
        self.r_exit = self.r_inlet * utils.Defaults.blade_row_radius_ratio
        self.area_inlet = np.pi *  (self.r_inlet**2 - utils.Defaults.hub_tip_ratio**2)
        self.area_exit = np.pi * (self.r_exit**2 - utils.Defaults.hub_tip_ratio**2)

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
        # create list of inlet streamtubes and iterate over each annulus of interest
        self.inlet = []
        for index in range(utils.Defaults.no_of_annuli):
            
            # consider annulus nearest the hub
            if index == 0:

                # find corresponding annulus radius and thickness
                r = (
                    (utils.Defaults.hub_tip_ratio + np.sqrt(
                        utils.Defaults.hub_tip_ratio**2 * (1 - 1 / utils.Defaults.no_of_annuli)
                        + 1 / utils.Defaults.no_of_annuli
                    )) / 2
                )
                dr = r - utils.Defaults.hub_tip_ratio
                flow_state = Flow_state(
                    M, alpha, 1, 1, 0
                )

            # consider all other annuli
            else:

                # find corresponding annulus radius and thickness
                r = (
                    (self.inlet[index - 1].r + self.inlet[index - 1].dr + np.sqrt(
                        (self.inlet[index - 1].r + self.inlet[index - 1].dr)**2
                        - (utils.Defaults.hub_tip_ratio**2 - 1) / utils.Defaults.no_of_annuli
                    )) / 2
                )
                dr = r - self.inlet[index - 1].r - self.inlet[index - 1].dr
                flow_state = Flow_state(
                    M, alpha, 1, 1, 0
                )

            # store instance of the streamtube class as an inlet condition
            self.inlet.append(Streamtube(flow_state, r, dr))

        self.mean_line()
        print(f"{utils.Colours.PURPLE}Inlet conditions:{utils.Colours.END}")
        for streamtube in self.inlet:

            print(streamtube)

        print(self.inlet_mean)

    def mean_line(self):
        """Determines the mean line inlet conditions from a series of annular streamtubes."""
        r_mean = (self.r_inlet + utils.Defaults.hub_tip_ratio) / 2
        rr = [streamtube.r for streamtube in self.inlet]
        quantities = ["M", "alpha", "T_0", "p_0", "s"]
        interp_values = {
            q: np.interp(r_mean, rr, [getattr(st.flow_state, q) for st in self.inlet])
            for q in quantities
        }
        M_mean, alpha_mean, T_0_mean, p_0_mean, s_mean = [interp_values[q] for q in quantities]
        flow_state = Flow_state(
            M_mean, alpha_mean, T_0_mean, p_0_mean, s_mean
        )
        self.inlet_mean = Streamtube(
            flow_state, r_mean, 0
        )

    def rotor_design(self, phi, psi):
        """Determines the rotor blade geometry necessary to satisfy the given stage parameters."""
        # determine variation of phi, psi and blade Mach number across the span
        for streamtube in self.inlet:

            # determine local flow coefficient
            streamtube.phi = (
                phi
                * streamtube.flow_state.M / self.inlet_mean.flow_state.M
                * np.cos(streamtube.flow_state.alpha) / np.cos(self.inlet_mean.flow_state.alpha)
                * np.sqrt(streamtube.flow_state.T / self.inlet_mean.flow_state.T)
                * self.inlet_mean.r / streamtube.r
            )

            # determine local stage loading coefficient
            streamtube.psi = (
                psi * np.power(streamtube.r / self.inlet_mean.r, utils.Defaults.vortex_exponent - 1)
            )

            # determine local blade Mach number
            streamtube.M_blade = (
                streamtube.flow_state.M * np.cos(streamtube.flow_state.alpha) / streamtube.phi
            )

            # determine relative Mach number and swirl angle via vector addition 
            v1 = streamtube.flow_state.M * np.array(
                [np.cos(streamtube.flow_state.alpha), np.sin(streamtube.flow_state.alpha)]
            )
            v2 = streamtube.M_blade * np.array([0, 1])
            v3 = v1 - v2
            streamtube.flow_state.M_rel = np.linalg.norm(v3)
            streamtube.flow_state.beta = np.arctan2(v3[1], v3[0])

            # determine relative stagnation pressure loss
            relative_stagnation_pressure_ratio = (
                1 - self.Y_p * (1 - utils.stagnation_pressure_ratio(streamtube.flow_state.M_rel))
            )

        def equations(vars):
            """Series of equations to solve the root of."""
            # reshape input vars for iteration
            vars_matrix = vars.reshape((len(self.inlet), 3))
            solutions = np.zeros_like(vars_matrix)

            # create empty list to store streamtube thicknesses and iterate
            dr_list = np.zeros(len(self.inlet))
            for index, var in enumerate(vars_matrix):

                # use hub radius for first streamtube
                if index == 0:

                    # set thickness using hub radius
                    dr_list[0] = vars_matrix[0][-1] - utils.Defaults.hub_tip_ratio

                # for all other streamtubes
                else:

                    # set thickness using previous radius
                    dr_list[index] = (
                        vars_matrix[index][-1] - vars_matrix[index - 1][-1] - dr_list[index - 1]
                    )

            # add streamtube thicknesses to matrix of input variables
            vars_matrix = np.column_stack((vars_matrix, dr_list))

            # initialise empty lists of guessed variables
            M_blade_list = np.zeros(len(self.inlet))
            M_list = np.zeros(len(self.inlet))
            alpha_list = np.zeros(len(self.inlet))
            T_0_ratio_list = np.zeros(len(self.inlet))

            # iterate over all streamtubes and sets of variables
            for index, (streamtube, var) in enumerate(zip(self.inlet, vars_matrix)):

                # separate variables out
                M_rel, beta, r, dr = var

                # find local blade Mach number at exit to the rotor row
                M_blade_list[index] = (
                    streamtube.M_blade * np.sqrt(
                        utils.stagnation_temperature_ratio(streamtube.flow_state.M_rel)
                        / utils.stagnation_temperature_ratio(M_rel)
                    )
                    * r / streamtube.r
                )

                # determine absolute Mach number and swirl angle via vector addition 
                v1 = M_rel * np.array([np.cos(beta), np.sin(beta)])
                v2 = M_blade_list[index] * np.array([0, 1])
                v3 = v1 + v2
                M_list[index] = np.linalg.norm(v3)
                alpha_list[index] = np.arctan2(v3[1], v3[0])

                # find local stagnation temperature ratio
                T_0_ratio_list[index] = (
                    1 + (utils.gamma - 1) * streamtube.psi * streamtube.M_blade**2
                    * utils.stagnation_temperature_ratio(streamtube.flow_state.M)
                )

            # repeat iteration with new values stored
            for index, (streamtube, var) in enumerate(zip(self.inlet, vars_matrix)):

                # separate variables out
                M_rel, beta, r, dr = var

                # determine residual for continuity equation
                solutions[index][0] = (
                    utils.mass_flow_function(streamtube.flow_state.M_rel)
                    * streamtube.A / (4 * np.pi * r * dr)
                    * np.cos(streamtube.flow_state.beta) / np.cos(beta)
                    / relative_stagnation_pressure_ratio
                    - utils.mass_flow_function(M_rel)
                )

                # determine residual for specified stage loading
                solutions[index][1] = (
                    1 - streamtube.psi + (
                        M_rel * np.sin(beta) * np.sqrt(
                            utils.stagnation_temperature_ratio(M_rel)
                            / utils.stagnation_temperature_ratio(streamtube.flow_state.M_rel)
                        ) - streamtube.flow_state.M * np.sin(streamtube.flow_state.alpha)
                    ) / streamtube.M_blade
                )

                # find non-dimensional entropy increase
                s_2i = (
                    streamtube.flow_state.s + np.log(
                        utils.stagnation_temperature_ratio(M_rel)
                        / utils.stagnation_temperature_ratio(streamtube.flow_state.M_rel)
                    ) / (utils.gamma - 1)
                    + np.log(
                        utils.stagnation_pressure_ratio(M_rel)
                        / utils.stagnation_pressure_ratio(streamtube.flow_state.M_rel)
                        * relative_stagnation_pressure_ratio
                    ) / utils.gamma
                )

                # determine residual for radial equilibrium equation
                if index < len(self.inlet) - 1:

                    term_1 = (
                        utils.stagnation_temperature_ratio(M_list[index])
                        * (s_2i - streamtube.flow_state.s)
                    )

                    radial_temperature_ratio = (
                        utils.stagnation_temperature_ratio(M_list[index + 1])
                        * T_0_ratio_list[index + 1]
                        / utils.stagnation_temperature_ratio(M_list[index])
                        / T_0_ratio_list[index]
                        * self.inlet[index + 1].flow_state.T_0 / streamtube.flow_state.T_0
                    )

                    term_2 = (
                        M_list[index] * np.cos(alpha_list[index])
                        * utils.stagnation_temperature_ratio(M_list[index]) * (
                            M_list[index + 1] * np.cos(alpha_list[index + 1]) * np.sqrt(
                                radial_temperature_ratio
                            )
                            - M_list[index] * np.cos(alpha_list[index])
                        )
                    )

                    term_3 = (
                        (M_list[index] * np.sin(alpha_list[index]))
                        * utils.stagnation_temperature_ratio(M_list[index]) * (
                            vars_matrix[index + 1][2] / r
                            * M_list[index + 1] * np.sin(alpha_list[index + 1])
                            * np.sqrt(radial_temperature_ratio)
                            - M_list[index] * np.cos(alpha_list[index])
                        )
                    )

                    term_4 = (
                        (
                            T_0_ratio_list[index + 1] / T_0_ratio_list[index]
                            * self.inlet[index + 1].flow_state.T_0 / streamtube.flow_state.T_0
                        )
                    )

                    solutions[index][2] = (
                        term_1 + term_2 + term_3 + term_4
                    )

            # final residual comes from constraint for all areas to sum to the exit area
            solutions[-1][-1] = (
                np.sum([4 * np.pi * var[2] * var[3] for var in vars_matrix]) - self.area_exit
            )

            # flatten solutions matrix and return
            solutions = solutions.ravel()
            return solutions
        
        x0 = np.zeros((len(self.inlet), 3))
        for index, streamtube in enumerate(self.inlet):

            x0[index] = [
                0.9 * streamtube.flow_state.M_rel,
                0.9 * streamtube.flow_state.beta,
                streamtube.r
            ]

        x0 = x0.ravel()
        lower = [0, -np.pi / 2, utils.Defaults.hub_tip_ratio]
        upper = [1, np.pi / 2, 1]
        lower = np.tile(lower, (len(self.inlet), 1)).ravel()
        upper = np.tile(upper, (len(self.inlet), 1)).ravel()
        sol = least_squares(equations, x0, bounds = (lower, upper))
        print(f"Success: {utils.Colours.PURPLE}{sol.success}{utils.Colours.END}")

        # separate out solution variables
        M_rel_list = sol.x[0::3]
        beta_list = sol.x[1::3]
        r_list = sol.x[2::3]

        self.exit = []

        for (streamtube, M_rel, beta, r) in zip(self.inlet, M_rel_list, beta_list, r_list):

            # find new blade Mach number
            M_blade = (
                streamtube.M_blade * np.sqrt(
                    utils.stagnation_temperature_ratio(streamtube.flow_state.M_rel)
                    / utils.stagnation_temperature_ratio(M_rel)
                )
                * r / streamtube.r
            )

            # determine absolute Mach number and swirl angle via vector addition
            v1 = M_rel * np.array([np.cos(beta), np.sin(beta)])
            v2 = M_blade * np.array([0, 1])
            v3 = v1 + v2
            M = np.linalg.norm(v3)
            alpha = np.arctan2(v3[1], v3[0])

            print(f"M: {M}")
            print(f"alpha: {utils.rad_to_deg(alpha)} deg")
            print(f"r: {r}")
            print(f"M_blade: {M_blade}\n")

            T_0 = (
                streamtube.flow_state.T_0 * (
                    1 + (utils.gamma - 1) * streamtube.psi * streamtube.M_blade**2
                    * utils.stagnation_temperature_ratio(streamtube.flow_state.M)
                )
            )

            print(f"T_0: {T_0}")

            T_0 = (
                streamtube.flow_state.T_0
                / utils.stagnation_temperature_ratio(M)
                * utils.stagnation_temperature_ratio(M_rel)
                / utils.stagnation_temperature_ratio(streamtube.flow_state.M_rel)
                * utils.stagnation_temperature_ratio(streamtube.flow_state.M)
            )

            p_0 = (
                streamtube.flow_state.p_0
                / utils.stagnation_pressure_ratio(M)
                * utils.stagnation_pressure_ratio(M_rel)
                / utils.stagnation_pressure_ratio(streamtube.flow_state.M_rel)
                * utils.stagnation_pressure_ratio(streamtube.flow_state.M)
                * relative_stagnation_pressure_ratio
            )

            s = (
                streamtube.flow_state.s + np.log(
                    utils.stagnation_temperature_ratio(M_rel)
                    / utils.stagnation_temperature_ratio(streamtube.flow_state.M_rel)
                ) / (utils.gamma - 1)
                + np.log(
                    utils.stagnation_pressure_ratio(M_rel)
                    / utils.stagnation_pressure_ratio(streamtube.flow_state.M_rel)
                    * relative_stagnation_pressure_ratio
                ) / utils.gamma
            )

            flow_state = Flow_state(
                M, alpha, T_0, p_0, s
            )

    def stator_design(self):
        """Determines the stator blade geometry necessary to satisfy the given stage parameters."""
        pass

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