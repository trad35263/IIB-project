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
        # determine variation of several parameters across the blade span at inlet
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
            self.exit = []

            # create empty list to store streamtube thicknesses and iterate
            for index, var in enumerate(vars_matrix):

                flow_state = Flow_state(
                    0, 0, 0, 0, 0, var[0], var[1]
                )

                # use hub radius for first streamtube
                if index == 0:

                    # set thickness using hub radius
                    dr = vars_matrix[0][2] - utils.Defaults.hub_tip_ratio
                    self.exit.append(Streamtube(flow_state, var[2], dr))

                # for all other streamtubes
                else:

                    # set thickness using previous radius
                    dr = var[2] - self.exit[index - 1].r - self.exit[index - 1].dr
                    self.exit.append(Streamtube(flow_state, var[2], dr))

            # add streamtube thicknesses to matrix of input variables
            #vars_matrix = np.column_stack((vars_matrix, dr_list))

            # initialise empty lists of guessed variables
            #M_blade_list = np.zeros(len(self.inlet))
            #M_list = np.zeros(len(self.inlet))
            #alpha_list = np.zeros(len(self.inlet))
            #T_0_ratio_list = np.zeros(len(self.inlet))

            # iterate over all streamtubes and sets of variables
            #for index, (streamtube, var) in enumerate(zip(self.inlet, vars_matrix)):
            for index, (inlet, exit) in enumerate(zip(self.inlet, self.exit)):

                # find local blade Mach number at exit to the rotor row
                exit.M_blade = (
                    inlet.M_blade * np.sqrt(
                        utils.stagnation_temperature_ratio(inlet.flow_state.M_rel)
                        / utils.stagnation_temperature_ratio(exit.flow_state.M_rel)
                    )
                    * exit.r / inlet.r
                )

                # determine absolute Mach number and swirl angle at exit via vector addition 
                v1 = exit.flow_state.M_rel * np.array([np.cos(exit.flow_state.beta), np.sin(exit.flow_state.beta)])
                v2 = exit.M_blade * np.array([0, 1])
                v3 = v1 + v2
                exit.flow_state.M = np.linalg.norm(v3)
                exit.flow_state.alpha = np.arctan2(v3[1], v3[0])

                # find local stagnation temperature ratio
                exit.flow_state.T_0 = (
                    1 + (utils.gamma - 1) * streamtube.psi * streamtube.M_blade**2
                    * utils.stagnation_temperature_ratio(streamtube.flow_state.M)
                )
                exit.flow_state.static_quantities()

                # find non-dimensional entropy
                exit.flow_state.s = (
                    inlet.flow_state.s + np.log(
                        utils.stagnation_temperature_ratio(exit.flow_state.M_rel)
                        / utils.stagnation_temperature_ratio(inlet.flow_state.M_rel)
                    ) / (utils.gamma - 1)
                    - np.log(
                        utils.stagnation_pressure_ratio(exit.flow_state.M_rel)
                        / utils.stagnation_pressure_ratio(inlet.flow_state.M_rel)
                        * relative_stagnation_pressure_ratio
                    ) / utils.gamma
                )

            # repeat iteration with new values stored
            for index, (inlet, exit) in enumerate(zip(self.inlet, self.exit)):

                # separate variables out
                #M_rel, beta, r, dr = var

                # determine residual for continuity equation
                solutions[index][0] = (
                    utils.mass_flow_function(inlet.flow_state.M_rel)
                    * inlet.A / exit.A
                    * np.cos(inlet.flow_state.beta) / np.cos(exit.flow_state.beta)
                    / relative_stagnation_pressure_ratio
                    - utils.mass_flow_function(exit.flow_state.M_rel)
                )

                # determine residual for specified stage loading
                """solutions[index][1] = (
                    1 - inlet.psi + (
                        exit.flow_state.M_rel * np.sin(exit.flow_state.beta) * np.sqrt(
                            utils.stagnation_temperature_ratio(exit.flow_state.M_rel)
                            / utils.stagnation_temperature_ratio(inlet.flow_state.M_rel)
                        ) - inlet.flow_state.M * np.sin(inlet.flow_state.alpha)
                    ) / inlet.M_blade
                )"""

                solutions[index][1] = (
                    inlet.psi - (
                        exit.flow_state.M * np.sin(exit.flow_state.alpha) * np.sqrt(
                            utils.stagnation_temperature_ratio(exit.flow_state.M_rel)
                            / utils.stagnation_temperature_ratio(inlet.flow_state.M_rel)
                        ) - inlet.flow_state.M * np.sin(inlet.flow_state.alpha)
                    ) / inlet.M_blade
                )

                # find non-dimensional entropy
                """s_2i = (
                    streamtube.flow_state.s + np.log(
                        utils.stagnation_temperature_ratio(M_rel)
                        / utils.stagnation_temperature_ratio(streamtube.flow_state.M_rel)
                    ) / (utils.gamma - 1)
                    - np.log(
                        utils.stagnation_pressure_ratio(M_rel)
                        / utils.stagnation_pressure_ratio(streamtube.flow_state.M_rel)
                        * relative_stagnation_pressure_ratio
                    ) / utils.gamma
                )"""

                # determine residual for radial equilibrium equation
                if index < len(self.inlet) - 1:

                    # find residual corresponding to thermal/entropy term
                    term_1 = (
                        exit.flow_state.T / exit.flow_state.T_0 * (
                            self.exit[index + 1].flow_state.s - exit.flow_state.s
                        )
                    )

                    # find residual corresponding to axial velocity term
                    term_2 = (
                        exit.flow_state.M * np.cos(exit.flow_state.alpha)
                        * exit.flow_state.T / exit.flow_state.T_0 * (
                            self.exit[index + 1].flow_state.M
                            * np.cos(self.exit[index + 1].flow_state.alpha) * np.sqrt(
                                self.exit[index + 1].flow_state.T / exit.flow_state.T
                            ) - exit.flow_state.M * np.cos(exit.flow_state.alpha)
                        )
                    )

                    # find residual corresponding to tangential velocity term
                    term_3 = (
                        exit.flow_state.M * np.sin(exit.flow_state.alpha)
                        * exit.flow_state.T / exit.flow_state.T_0 * (
                            self.exit[index + 1].flow_state.M
                            * np.sin(self.exit[index + 1].flow_state.alpha) * np.sqrt(
                                self.exit[index + 1].flow_state.T / exit.flow_state.T
                            ) * self.exit[index + 1].r / exit.r
                            - exit.flow_state.M * np.sin(exit.flow_state.alpha)
                        )
                    )

                    # find residual corresponding to stagnation enthalpy term
                    term_4 = (
                        (1 - self.exit[index + 1].flow_state.T_0 / exit.flow_state.T_0)
                        / (utils.gamma - 1)
                    )

                    # sum all terms together to get overall residual
                    solutions[index][2] = (
                        term_1 + term_2 + term_3 + term_4
                    )

                    print("\n---------------------------")
                    print(f"term_1: {term_1}")
                    print(f"term_2: {term_2}")
                    print(f"term_3: {term_3}")
                    print(f"term_4: {term_4}")
                    print(f"solutions[index][2]: {solutions[index][2]}")

            # final residual comes from constraint for all areas to sum to the exit area
            solutions[-1][-1] = (
                np.sum([exit.A for exit in self.exit]) - self.area_exit
            )

            # flatten solutions matrix and return
            solutions = solutions.ravel()
            return solutions
        
        # initialise array to store initial guess and iterate
        x0 = np.zeros((len(self.inlet), 3))
        for index, streamtube in enumerate(self.inlet):

            # assume solution is close to the inlet conditions
            x0[index] = [
                0.9 * streamtube.flow_state.M_rel,
                0.9 * streamtube.flow_state.beta,
                streamtube.r
            ]

        # flatten initial guess array
        x0 = x0.ravel()

        # set lower and upper guess bounds and shape correctly
        lower = [0, -np.pi / 2, utils.Defaults.hub_tip_ratio]
        upper = [1, np.pi / 2, 1]
        lower = np.tile(lower, (len(self.inlet), 1)).ravel()
        upper = np.tile(upper, (len(self.inlet), 1)).ravel()

        # solve for least squares solution
        sol = least_squares(equations, x0, bounds = (lower, upper))
        print(f"Success: {utils.Colours.PURPLE}{sol.success} {sol.status}{utils.Colours.END}")

        if sol.status == 1:

            print(sol.x)
            print(sol.fun)

        else:

            print(sol)

        # separate out solution variables
        M_rel_list = sol.x[0::3]
        beta_list = sol.x[1::3]
        r_list = sol.x[2::3]

        print(f"max M_rel: {max(M_rel_list)}")

        # find corresponding list of streamtube thicknesses
        dr_list = np.zeros_like(r_list)
        for index, r in enumerate(r_list):

            if index == 0:

                dr_list[index] = r - utils.Defaults.hub_tip_ratio
            
            else:

                dr_list[index] = r - r_list[index - 1] - dr_list[index - 1]

        self.exit = []

        for (streamtube, M_rel, beta, r, dr) in zip(self.inlet, M_rel_list, beta_list, r_list, dr_list):

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

            T_0 = (
                streamtube.flow_state.T_0 * (
                    1 + (utils.gamma - 1) * streamtube.psi * streamtube.M_blade**2
                    * utils.stagnation_temperature_ratio(streamtube.flow_state.M)
                )
            )

            print(f"T_0: {T_0}")

            # find new stagnation temperature
            T_0 = (
                streamtube.flow_state.T_0
                / utils.stagnation_temperature_ratio(M)
                * utils.stagnation_temperature_ratio(M_rel)
                / utils.stagnation_temperature_ratio(streamtube.flow_state.M_rel)
                * utils.stagnation_temperature_ratio(streamtube.flow_state.M)
            )

            # find new stagnation pressure
            p_0 = (
                streamtube.flow_state.p_0
                / utils.stagnation_pressure_ratio(M)
                * utils.stagnation_pressure_ratio(M_rel)
                / utils.stagnation_pressure_ratio(streamtube.flow_state.M_rel)
                * utils.stagnation_pressure_ratio(streamtube.flow_state.M)
                * relative_stagnation_pressure_ratio
            )

            # find new entropy
            s = (
                streamtube.flow_state.s + np.log(
                    utils.stagnation_temperature_ratio(M_rel)
                    / utils.stagnation_temperature_ratio(streamtube.flow_state.M_rel)
                ) / (utils.gamma - 1)
                - np.log(
                    utils.stagnation_pressure_ratio(M_rel)
                    / utils.stagnation_pressure_ratio(streamtube.flow_state.M_rel)
                    * relative_stagnation_pressure_ratio
                ) / utils.gamma
            )
            print(f"s: {s}")

            # create corresponding Flow_state and Streamtube objects
            flow_state = Flow_state(
                M, alpha, T_0, p_0, s, M_rel, beta
            )
            self.exit.append(Streamtube(flow_state, r, dr))
            self.exit[-1].M_blade = M_blade

        # run diagnostic checks on given solution
        for index, (inlet, exit) in enumerate(zip(self.inlet, self.exit)):

            print("\nChecking mass continuity...")
            print(f"{utils.mass_flow_function(exit.flow_state.M_rel)}")
            x = (
                utils.mass_flow_function(inlet.flow_state.M_rel)
                * inlet.A / exit.A
                * np.cos(inlet.flow_state.beta) / np.cos(exit.flow_state.beta)
                / relative_stagnation_pressure_ratio
            )
            print(f"{x}")

            print("\nChecking stage loading...")
            print(f"{inlet.psi}")
            x = (
                (
                    exit.flow_state.M * np.sin(exit.flow_state.alpha) * np.sqrt(
                        exit.flow_state.T / inlet.flow_state.T
                    )
                    - inlet.flow_state.M * np.sin(inlet.flow_state.alpha)
                ) / inlet.M_blade
            )
            print(f"{x}")

            if index < len(self.inlet) - 1:

                print("\nChecking radial equilibrium...")
                x = (
                    utils.stagnation_temperature_ratio(exit.flow_state.M)
                    * (self.exit[index + 1].flow_state.s - exit.flow_state.s)
                )
                print(f"{x}")
                y = (
                    exit.flow_state.M * np.cos(exit.flow_state.alpha)
                    * exit.flow_state.T / exit.flow_state.T_0 * (
                        self.exit[index + 1].flow_state.M
                        * np.cos(self.exit[index + 1].flow_state.alpha) * np.sqrt(
                            self.exit[index + 1].flow_state.T / exit.flow_state.T
                        )
                        - exit.flow_state.M * np.cos(exit.flow_state.alpha)
                    )
                )
                print(f"y: {y}")

                z = (
                    (exit.M_blade + exit.flow_state.M_rel * np.sin(exit.flow_state.alpha))
                    * utils.stagnation_temperature_ratio(exit.flow_state.M) * (
                        self.exit[index + 1].r / exit.r * (
                            self.exit[index + 1].M_blade
                            + self.exit[index + 1].flow_state.M_rel
                            * np.sin(self.exit[index + 1].flow_state.beta)
                        ) * np.sqrt(
                            self.exit[index + 1].flow_state.T / exit.flow_state.T
                        )
                        - exit.M_blade - exit.flow_state.M_rel * np.sin(exit.flow_state.beta)
                    )
                )
                print(f"z: {z}")

                term_3 = (
                    exit.flow_state.M * np.sin(exit.flow_state.alpha)
                    * exit.flow_state.T / exit.flow_state.T_0 * (
                        self.exit[index + 1].flow_state.M
                        * np.sin(self.exit[index + 1].flow_state.alpha) * np.sqrt(
                            self.exit[index + 1].flow_state.T / exit.flow_state.T
                        ) * self.exit[index + 1].r / exit.r
                        - exit.flow_state.M * np.sin(exit.flow_state.alpha)
                    )
                )
                print(f"term_3: {term_3}")

                # need to check if z and term_3 are the same!!!

                q = (
                    (1 - self.exit[index + 1].flow_state.T_0 / exit.flow_state.T_0)
                    / (utils.gamma - 1)
                )
                print(f"q: {q}")

                term_4 = (
                    (1 - self.exit[index + 1].flow_state.T_0 / exit.flow_state.T_0)
                    / (utils.gamma - 1)
                )
                print(f"term_4: {term_4}")

                print(f"{x + y + z + q}")

        print("\nChecking area summation criteria...")
        print(f"{self.area_exit}")
        x = np.sum([exit.A for exit in self.exit])
        print(f"{x}")


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