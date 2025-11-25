# import modules

import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import least_squares
from scipy.interpolate import make_interp_spline

from streamtube import Streamtube
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
    def __init__(self, x_inlet, x_exit, inlet = None):
        """Create instance of the Nozzle class."""
        # store input variables
        self.x_inlet = x_inlet
        self.x_exit = x_exit
        self.inlet = inlet

        # preallocate properties
        self.exit = None
        self.r_hub = 0

        # set label and colour
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

# design functions --------------------------------------------------------------------------------

    def nozzle_design(self, p):
        """Determines the nozzle area to satisfy the atmospheric pressure boundary condition."""
        # create empty array of exit streamtubes
        self.exit = np.empty((len(self.inlet),), dtype = object)

        def solve_nozzle(vars):
            """Series of equations to solve the root of."""
            # reshape input variables for iteration and create empty solutions array
            vars = vars.reshape((len(self.inlet), 3))
            solutions = np.zeros_like(vars)

            # iterate over all sets of input variables
            for index, (var, inlet, exit) in enumerate(zip(vars, self.inlet, self.exit)):

                # create a holder flow_state given that process is isentropic
                flow_state = Flow_state(
                    var[0], var[1], inlet.flow_state.T_0, inlet.flow_state.p_0, inlet.flow_state.s
                )
                flow_state.static_quantities()

                # handle inner streamtube
                if index == 0:

                    # effective hub radius is zero
                    r1 = 0
                    r2 = np.sqrt(var[2] / np.pi + r1**2)
                    r = (r1 + r2) / 2
                    dr = r - r1

                # for all other streamtubes
                else:

                    # set thickness using previous radius
                    r1 = self.exit[index - 1].r + self.exit[index - 1].dr
                    r2 = np.sqrt(var[2] / np.pi + r1**2)
                    r = (r1 + r2) / 2
                    dr = r - r1

                # create streamtube and store at exit to the nozzle
                self.exit[index] = Streamtube(flow_state, r, dr)

            # iterate over all inlet-exit pairs
            for index, (inlet, exit) in enumerate(zip(self.inlet, self.exit)):

                # find non-dimensional velocity components for radial equilibrium
                exit.flow_state.v_x = (
                    exit.flow_state.M * np.sqrt(exit.flow_state.T) * np.cos(exit.flow_state.alpha)
                )
                exit.flow_state.v_theta = (
                    exit.flow_state.M * np.sqrt(exit.flow_state.T) * np.sin(exit.flow_state.alpha)
                )

            # only compute splines for radial equilibrium if more than one streamtube exists
            if len(self.inlet) > 1:

                # fit spline for static temperature
                T_spline = make_interp_spline(
                    [exit.r for exit in self.exit],
                    [exit.flow_state.T for exit in self.exit],
                    k = min(len(self.inlet) - 1, 2)
                )

                # fit spline for entropy term
                s_spline = make_interp_spline(
                    [exit.r for exit in self.exit],
                    [exit.flow_state.s for exit in self.exit],
                    k = min(len(self.inlet) - 1, 2)
                )

                # fit spline for v_x term
                v_x_spline = make_interp_spline(
                    [exit.r for exit in self.exit],
                    [exit.flow_state.v_x for exit in self.exit],
                    k = min(len(self.inlet) - 1, 2)
                )

                # fit spline for v_theta term
                v_theta_spline = make_interp_spline(
                    [exit.r for exit in self.exit],
                    [exit.flow_state.v_theta for exit in self.exit],
                    k = min(len(self.inlet) - 1, 2)
                )

                # fit spline for T_0 term
                T_0_spline = make_interp_spline(
                    [exit.r for exit in self.exit],
                    [exit.flow_state.T_0 for exit in self.exit],
                    k = min(len(self.inlet) - 1, 2)
                )

            # repeat iteration with new values stored
            for index, (inlet, exit) in enumerate(zip(self.inlet, self.exit)):

                # determine residual for continuity equation
                solutions[index][0] = (
                    utils.mass_flow_function(inlet.flow_state.M)
                    * inlet.A / exit.A
                    * np.cos(inlet.flow_state.alpha) / np.cos(exit.flow_state.alpha)
                    - utils.mass_flow_function(exit.flow_state.M)
                )

                # determine residual for conservation of angular momentum
                solutions[index][1] = (
                    inlet.r * inlet.flow_state.M * np.sin(inlet.flow_state.alpha)
                    * np.sqrt(inlet.flow_state.T)
                    - exit.r * exit.flow_state.M * np.sin(exit.flow_state.alpha)
                    * np.sqrt(exit.flow_state.T)
                )

                # determine residual for radial equilibrium equation
                if index < len(self.inlet) - 1:

                    # choose point to evaluate radial equilibrium at
                    r = (exit.r + self.exit[index + 1].r) / 2

                    # find residual corresponding to thermal/entropy term
                    term_1 = T_spline(r) * s_spline.derivative()(r)

                    # find residual corresponding to axial velocity term
                    term_2 = v_x_spline(r) * v_x_spline.derivative()(r)

                    # find residual corresponding to tangential velocity term
                    term_3 = (
                        v_theta_spline(r) / r
                        * (v_theta_spline(r) + r * v_theta_spline.derivative()(r))
                    )

                    # find residual corresponding to stagnation enthalpy term
                    term_4 = -1 / (utils.gamma - 1) * T_0_spline.derivative()(r)

                    # sum all terms together to get overall residual
                    solutions[index][2] = (
                        term_1 + term_2 + term_3 + term_4
                    )

                    # sum all terms together to get overall residual
                    solutions[index][2] = (
                        term_1 + term_2 + term_3 + term_4
                    )

                    # debugging
                    """print("\n---------------------------")
                    print(f"term_1: {term_1}")
                    print(f"term_2: {term_2}")
                    print(f"term_3: {term_3}")
                    print(f"term_4: {term_4}")
                    print(f"exit.flow_state.T_0: {exit.flow_state.T_0}")
                    print(f"self.exit[index + 1].flow_state.T_0: {self.exit[index + 1].flow_state.T_0}")
                    print(f"solutions[index][2]: {solutions[index][2]}")"""

            # final residual comes from atmospheric pressure boundary condition
            solutions[-1][-1] = p - self.exit[-1].flow_state.p

            # flatten solutions matrix
            solutions = solutions.ravel()

            # extra residual comes from areas summing to nozzle area guess
            #solutions = np.append(solutions, A - np.sum([np.abs(exit.A) for exit in self.exit]))
            return solutions
        
        # initialise array to store initial guess and iterate
        x0 = np.zeros((len(self.inlet), 3))
        for index, inlet in enumerate(self.inlet):

            # assume solution is close to the inlet conditions
            x0[index] = [
                utils.invert(utils.stagnation_pressure_ratio, p / inlet.flow_state.p_0),
                1.5 * inlet.flow_state.alpha,
                0.8 * inlet.A
            ]

            # flatten initial guess array
            if not np.isfinite(x0[index][0]):

                x0[index][0] = 0

        x0 = x0.ravel()

        # set lower and upper guess bounds and shape correctly
        lower = [0, -np.pi / 2, 0]
        upper = [1, np.pi / 2, np.pi]
        lower = np.tile(lower, (len(self.inlet), 1)).ravel()
        upper = np.tile(upper, (len(self.inlet), 1)).ravel()

        # solve for least squares solution
        sol = least_squares(solve_nozzle, x0, bounds = (lower, upper))
        self.A_exit = np.sum([exit.A for exit in self.exit])
    
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

        # return instance of Flow_state class corresponding to exit conditions
        self.exit = Flow_state(
            exit_M,
            exit_alpha,
            self.inlet.T_0,
            self.inlet.p_0
        )

    def evaluate(self, M_1, M_flight):
        """Determine key performance metrics local to the nozzle exit conditions."""
        # find thrust coefficient
        for exit in self.exit:

            # determine local thrust coefficient
            exit.C_th = (
                utils.mass_flow_function(M_1) * np.sqrt(utils.gamma - 1) * (
                    exit.flow_state.M * np.cos(exit.flow_state.alpha)
                    * np.sqrt(exit.flow_state.T)
                    - M_flight * np.sqrt(utils.stagnation_temperature_ratio(M_flight))
                )
            )

            # determine local propulsive efficiency
            exit.eta_prop = (
                2 * exit.C_th * M_flight / utils.mass_flow_function(M_1) * np.sqrt(
                    utils.stagnation_temperature_ratio(M_flight) / (utils.gamma - 1)
                ) / (
                    exit.flow_state.M**2 * exit.flow_state.T
                    - M_flight**2 * utils.stagnation_temperature_ratio(M_flight)
                )
            )

            # determine local nozzle efficiency
            exit.eta_nozz = (
                (utils.gamma - 1) / 2 * (
                    (
                        exit.flow_state.M**2 * exit.flow_state.T
                        - M_flight**2 * utils.stagnation_temperature_ratio(M_flight)
                    ) / (np.power(exit.flow_state.p_0, 1 - 1 / utils.gamma) - 1)
                )
            )

            # determine local compressor efficiency
            exit.eta_comp = (
                (np.power(exit.flow_state.p_0, 1 - 1 / utils.gamma) - 1)
                / (exit.flow_state.T_0 - 1)
            )

            # determine local jet velocity ratio
            exit.jet_velocity_ratio = (
                M_flight / exit.flow_state.M
                * np.sqrt(utils.stagnation_temperature_ratio(M_flight) / exit.flow_state.T)
            )
