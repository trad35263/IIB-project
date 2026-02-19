# import modules

import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import least_squares
from scipy.interpolate import make_interp_spline
from scipy.integrate import cumulative_simpson
from scipy.integrate import simpson
import copy

from annulus import Annulus
from coefficients import Coefficients

from streamtube import Streamtube
from flow_state import Flow_state
import utils

# create Nozzle class

class Nozzle:
    """Represents the engine nozzle and solves for the conditions at inlet and outlet."""
    def __init__(self):
        """Create instance of the Nozzle class."""
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

    def old_nozzle_design(self, p):
        """Determines the nozzle area to satisfy the atmospheric pressure boundary condition."""
        # create empty array of exit streamtubes
        self.exit = np.empty((len(self.inlet),), dtype = object)

        def solve_nozzle(vars):
            """Series of equations to solve the root of."""
            # reshape input variables for iteration and create empty solutions array
            vars = vars.reshape((len(self.inlet), 2))
            solutions = np.zeros_like(vars)

            # iterate over all sets of input variables
            for index, (var, inlet, exit) in enumerate(zip(vars, self.inlet, self.exit)):

                # create a holder flow_state given that process is isentropic
                flow_state = Flow_state(
                    0, var[0], inlet.flow_state.T_0, inlet.flow_state.p_0, inlet.flow_state.s
                )

                # handle inner streamtube
                if index == 0:

                    # effective hub radius is zero
                    r1 = 0
                    r2 = np.sqrt(var[1] / np.pi + r1**2)
                    r = (r1 + r2) / 2
                    dr = r - r1

                # for all other streamtubes
                else:

                    # set thickness using previous radius
                    r1 = self.exit[index - 1].r + self.exit[index - 1].dr
                    r2 = np.sqrt(var[1] / np.pi + r1**2)
                    r = (r1 + r2) / 2
                    dr = r - r1

                # create streamtube and store at exit to the nozzle
                self.exit[index] = Streamtube(flow_state, r, dr)

            # iterate over all inlet-exit pairs
            for index, (inlet, exit) in enumerate(zip(self.inlet, self.exit)):

                # apply continuity to find exit Mach number
                m_cpT0_Ap0 = (
                    utils.mass_flow_function(inlet.flow_state.M)
                    * inlet.A / exit.A
                    * np.cos(inlet.flow_state.alpha) / np.cos(exit.flow_state.alpha)
                )

                # check if mass flow function is valid
                if m_cpT0_Ap0 > utils.mass_flow_function(1):

                    return 1e9 * np.ones_like(vars).ravel()

                # calculate exit Mach number
                exit.flow_state.M = utils.invert(utils.mass_flow_function, m_cpT0_Ap0)

                # solve for static quantities
                flow_state.static_quantities()

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

                # determine residual for continuity equation - RMOEVE ME
                """solutions[index][0] = (
                    utils.mass_flow_function(inlet.flow_state.M)
                    * inlet.A / exit.A
                    * np.cos(inlet.flow_state.alpha) / np.cos(exit.flow_state.alpha)
                    - utils.mass_flow_function(exit.flow_state.M)
                )"""

                # determine residual for conservation of angular momentum
                solutions[index][0] = (
                    inlet.r * inlet.flow_state.M * np.sin(inlet.flow_state.alpha)
                    * np.sqrt(inlet.flow_state.T)
                    - exit.r * exit.flow_state.M * np.sin(exit.flow_state.alpha)
                    * np.sqrt(exit.flow_state.T)
                )

                # determine residual for radial equilibrium equation
                if index < len(self.inlet) - 1:

                    # evaluate radial equilibrium at boundaries between streamtubes
                    r = exit.r + exit.dr

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
                    solutions[index][1] = (
                        term_1 + term_2 + term_3 + term_4
                    )

            # final residual comes from atmospheric pressure boundary condition
            solutions[-1][-1] = p - self.exit[-1].flow_state.p

            # flatten solutions matrix
            solutions = solutions.ravel()

            return solutions
        
        # initialise array to store initial guess and iterate
        x0 = np.zeros((len(self.inlet), 2))
        for index, inlet in enumerate(self.inlet):

            # assume solution is close to the inlet conditions
            x0[index] = [
                inlet.flow_state.alpha,
                0.8 * inlet.A
            ]

        # flatten initial guess array
        x0 = x0.ravel()

        # set lower and upper guess bounds and shape correctly
        lower = [-np.pi / 2, 0]
        upper = [np.pi / 2, np.pi]
        lower = np.tile(lower, (len(self.inlet), 1)).ravel()
        upper = np.tile(upper, (len(self.inlet), 1)).ravel()

        # solve for least squares solution
        sol = least_squares(solve_nozzle, x0, bounds = (lower, upper))
        self.A_exit = np.sum([exit.A for exit in self.exit])
        utils.debug(f"sol: {sol}")

    def old_design(self, p_atm):
        """Determines the nozzle area to satisfy the atmospheric pressure boundary condition."""
        # initialise exit annulus object to be populated
        self.exit = Annulus()

        # store variation in static properties for convenience
        T_1 = self.inlet.T_0.value * utils.stagnation_temperature_ratio(self.inlet.M.value)
        p_1 = self.inlet.p_0.value * utils.stagnation_pressure_ratio(self.inlet.M.value)

        # get incremental change in inlet mass flow
        dm_dr_1 = (
            p_1 / np.sqrt(T_1) * self.inlet.M.value * np.cos(self.inlet.alpha.value)
            * self.inlet.rr
        )
        m_dot_1 = cumulative_simpson(dm_dr_1, x = self.inlet.rr, initial = 0.0)
        dm_dot_1 = np.diff(m_dot_1)

        # FOR NOW ASSUME THAT INLET ANGLE IS ZERO AND CONSERVATION OF ANGULAR MOMENTUM IS TRIVIAL
        
        def solve_nozzle(vars):
            """Determines the matrix of residuals for a given guess of coefficients."""
            # store guess of exit conditions
            self.exit.M = Coefficients(vars)

            # set up solutions matrix to be populated
            solutions = np.zeros_like(vars)
            
            # initialise vector of new radial positions
            self.exit.rr = np.zeros_like(self.inlet.rr)
            self.exit.rr[0] = 1e-3

            # some quantities can be known because nozzle is isentropic
            self.exit.alpha = copy.deepcopy(self.inlet.alpha)
            self.exit.T_0 = copy.deepcopy(self.inlet.T_0)
            self.exit.p_0 = copy.deepcopy(self.inlet.p_0)
            self.exit.s = copy.deepcopy(self.inlet.s)

            # loop over all streamtubes
            for index, m_1 in enumerate(dm_dot_1):

                # get exit inner streamtube radius and determine extra-fine grid for interpolation
                r_2_fine = np.linspace(
                    self.exit.rr[index],
                    self.exit.rr[index] + 2 * (self.inlet.rr[index + 1] - self.inlet.rr[index]),
                    utils.Defaults.solver_grid
                )

                # evaluate relative Mach numbers and flow angles on fine, local grid
                M_2 = np.polyval(self.exit.M.coefficients, r_2_fine)
                alpha_2 = np.polyval(self.exit.alpha.coefficients, r_2_fine)

                # get variation in mass flow rate at the inlet radial nodes
                dm_dr_2 = (
                    np.power(
                        1 + 0.5 * (utils.gamma - 1) * M_2**2,
                        -utils.gamma / (utils.gamma - 1) + 0.5
                    ) * M_2 * np.cos(alpha_2) * r_2_fine
                )
                m_dot_2 = (
                    self.exit.p_0.value[index] / np.sqrt(self.exit.T_0.value[index])
                    * cumulative_simpson(dm_dr_2, x = r_2_fine, initial = 0.0)
                )

                # interpolate to find upper bound of corresponding streamtube
                self.exit.rr[index + 1] = np.interp(m_1, m_dot_2, r_2_fine)

            # expand primary flow variables onto new grid
            self.exit.value("M")

            # get variation in exit static properties
            self.exit.T.value = (
                self.exit.T_0.value * utils.stagnation_temperature_ratio(self.exit.M.value)
            )
            self.exit.p.value = (
                self.exit.p_0.value * utils.stagnation_pressure_ratio(self.exit.M.value)
            )

            # calculate dimensionless velocity components at exit
            v_x_2 = self.exit.M.value * np.sqrt(self.exit.T.value) * np.cos(self.exit.alpha.value)
            rv_theta_2 = (
                self.exit.rr * self.exit.M.value * np.sqrt(self.exit.T.value)
                * np.sin(self.exit.alpha.value)
            )

            # calculate necessary derivatives for radial equilibrium
            ds_dr = np.gradient(self.exit.s.value, self.exit.rr, edge_order = 2)
            dv_x_dr = np.gradient(v_x_2, self.exit.rr, edge_order = 2)
            drv_theta_dr = np.gradient(rv_theta_2, self.exit.rr, edge_order = 2)
            dT_0_dr = np.gradient(self.exit.T_0.value, self.exit.rr, edge_order = 2)

            # evaluate radial equilibrium
            self.exit.dr = (
                self.exit.T.value * ds_dr + v_x_2 * dv_x_dr
                + rv_theta_2 / self.exit.rr * drv_theta_dr
                - 1 / (utils.gamma - 1) * dT_0_dr
            )

            # convert stage loading residuals to a (1, N) residual array
            dr_buckets = np.array_split(self.exit.dr, solutions.shape[0] - 1)
            solutions[:-1] = np.array([
                np.sqrt(np.mean(bucket**2)) for bucket in dr_buckets
            ])

            # final residual comes from atmospheric pressure boundary condition
            solutions[-1] = self.exit.p.value[-1] - p_atm

            # return solutions
            solutions = solutions.ravel()
            return solutions

        # set list of lower and upper bounds and reshape
        lower = 100 * -2 * np.ones_like(self.inlet.M.coefficients)
        upper = 100 * 2 * np.ones_like(self.inlet.M.coefficients)

        # get initial guess based on inlet conditions
        x0 = self.inlet.M.coefficients

        # solve iteratively
        sol = least_squares(
            solve_nozzle, x0, bounds = (lower, upper),
            xtol = 1e-6, ftol = 1e-6, gtol = 1e-6,
            x_scale = 'jac'
        )
        utils.debug(f"sol: {sol}")

    def design(self, v_x_hub, hub_tip_ratio):
        """Determines the flowfield through the nozzle and solves for its geometry."""
        # hub dimensionless axial velocity and radius are known
        self.exit.v_x[0] = v_x_hub
        self.exit.rr[0] = 1e-2

        # nozzle is isentropic
        self.exit.T_0 = self.inlet.T_0
        self.exit.p_0 = self.inlet.p_0
        self.exit.s = self.inlet.s
        
        # set exit angle distribution to zero (make this more general later)
        self.exit.alpha = np.zeros(utils.Defaults.solver_grid)  # technically this is done already but included for clarity

        # determine inlet mass flow rate distribution
        self.inlet.dm_dot_dr = (
            2 * utils.gamma / ((1 - hub_tip_ratio**2) * np.sqrt(utils.gamma - 1))
            * self.inlet.p / np.sqrt(self.inlet.T)
            * self.inlet.M * np.cos(self.inlet.alpha) * self.inlet.rr
        )
        self.inlet.m_dot = utils.cumulative_trapezoid(self.inlet.rr, self.inlet.dm_dot_dr)

        # get mass flow rate through each streamtube
        dm_dot_1 = np.diff(self.inlet.m_dot)

        # loop over each streamtube
        for index in range(utils.Defaults.solver_grid):
                
            # for all cases except hub streamline
            if index > 0:
                
                # create fine grid for calculating streamtube upper bound 
                r_2_fine = np.linspace(
                    self.exit.rr[index - 1],
                    self.exit.rr[index - 1] + 10 * (self.inlet.rr[index] - self.inlet.rr[index - 1]),
                    utils.Defaults.solver_grid
                )

                # determine local exit mass flow rate distribution
                dm_dr_2 = (
                    2 * utils.gamma / ((1 - hub_tip_ratio**2) * np.sqrt(utils.gamma - 1))
                    * self.exit.p_0[index - 1] / np.sqrt(self.exit.T_0[index - 1])
                    * self.exit.M[index - 1] * np.power(
                        1 + 0.5 * (utils.gamma - 1) * self.exit.M[index - 1]**2,
                        0.5 - utils.gamma / (utils.gamma - 1)
                    ) * np.cos(self.exit.alpha[index - 1]) * r_2_fine
                )
                m_dot_2 = utils.cumulative_trapezoid(r_2_fine, dm_dr_2)

                # interpolate to find upper bound of corresponding streamtube
                self.exit.rr[index] = np.interp(dm_dot_1[index - 1], m_dot_2, r_2_fine)

                # calculate derivatives required for radial equilibrium
                dT_0 = self.exit.T_0[index] - self.exit.T_0[index - 1]
                ds = self.exit.s[index] - self.exit.s[index - 1]
                dtan_2_alpha = (np.tan(self.exit.alpha[index]))**2 - (np.tan(self.exit.alpha[index - 1]))**2

                # calculate dimensionless axial velocity via difference equation
                self.exit.v_x[index] = (
                    self.exit.v_x[index - 1] + (
                        dT_0 / (utils.gamma - 1)
                        - self.exit.T[index - 1] * ds
                        - self.exit.v_x[index - 1]**2 * (
                            (np.tan(self.exit.alpha[index - 1]))**2 / self.exit.rr[index - 1]
                            * (self.exit.rr[index] - self.exit.rr[index - 1])
                            + 0.5 * dtan_2_alpha
                        )
                    ) / self.exit.v_x[index - 1]
                )

            # get exit tangential velocity from axial velocity and flow angle
            self.exit.v_theta[index] = self.exit.v_x[index] * np.tan(self.exit.alpha[index])

            # get Mach number
            v_squared = (self.exit.v_x[index] / np.cos(self.exit.alpha[index]))**2
            self.exit.M[index] = np.sqrt(v_squared / (1 - 0.5 * (utils.gamma - 1) * v_squared))

        # get exit static conditions
        self.exit.T = self.exit.T_0 * utils.stagnation_temperature_ratio(self.exit.M)
        self.exit.p = self.exit.p_0 * utils.stagnation_temperature_ratio(self.exit.M)

        # calculate exit mass flow rate
        self.exit.dm_dot_dr = (
            2 * utils.gamma / ((1 - hub_tip_ratio**2) * np.sqrt(utils.gamma - 1))
            * self.exit.p / np.sqrt(self.exit.T)
            * self.exit.M * np.cos(self.exit.alpha) * self.exit.rr
        )
        self.exit.m_dot = utils.cumulative_trapezoid(self.exit.rr, self.exit.dm_dot_dr)

    def old_evaluate(self, hub_tip_ratio):
        """Evaluates the performance of the nozzle as part of the engine system."""
        # get cumulative mass flow rate
        dm_dot_dr = (
            2 * utils.gamma / ((1 - hub_tip_ratio**2) * np.sqrt(utils.gamma - 1))
            * self.exit.p.value / np.sqrt(self.exit.T.value)
            * self.exit.M.value * np.cos(self.exit.alpha.value) * self.exit.rr
        )
        """dm_dot_dr = (
            2 * utils.mass_flow_function(self.exit.M.value)
            * self.exit.p_0.value / np.sqrt(self.exit.T_0.value)
            * self.exit.rr / (1 - utils.Defaults.hub_tip_ratio**2)
        )"""
        self.m_dot = cumulative_simpson(dm_dot_dr, x = self.exit.rr, initial = 0)

        # find mass-averaged stagnation temperature ratio
        self.T_0_ratio = (
            simpson(dm_dot_dr * self.exit.T_0.value, x = self.exit.rr) / self.m_dot[-1]
        )

        # find mass-averaged stagnation pressure ratio
        self.p_0_ratio = (
            simpson(dm_dot_dr * self.exit.p_0.value, x = self.exit.rr) / self.m_dot[-1]
        )

        # find cumulative thrust coefficient
        dC_th_dr = (
            2 / (1 - hub_tip_ratio**2)
            * (
                utils.impulse_function(self.exit.M.value)
                - 2 * utils.dynamic_pressure_function(self.exit.M.value)
                * (np.sin(self.exit.alpha.value))**2
            )
            * self.exit.p_0.value * self.exit.rr
        )
        self.C_th = cumulative_simpson(dC_th_dr, x = self.exit.rr, initial = 0)

        self.area_ratio = (
            self.exit.rr[-1]**2 / (1 - hub_tip_ratio**2)
        )

    def evaluate(self, hub_tip_ratio):
        """Evaluates the nozzle performance."""
        # store nozzle area ratio
        self.area_ratio = self.exit.rr[-1]**2 / (1 - hub_tip_ratio**2)

        # find cumulative thrust coefficient distribution
        """dC_th_dr = (    # with pressure terms
            2 / (1 - hub_tip_ratio**2) * (
                utils.impulse_function(self.exit.M)
                - 2 * utils.dynamic_pressure_function(self.exit.M)
                * (np.sin(self.exit.alpha))**2
            ) * self.exit.p_0 * self.exit.rr
        )"""
        dC_th_dr = (    # neglecting pressure terms
            2 * utils.gamma / (1 - hub_tip_ratio**2)
            * self.exit.p * self.exit.M**2 * (np.cos(self.exit.alpha))**2 * self.exit.rr
        )
        self.C_th = utils.cumulative_trapezoid(self.exit.rr, dC_th_dr)

        # find mass-averaged stagnation temperature and pressure ratios
        self.T_0_ratio = (
            utils.cumulative_trapezoid(
                self.exit.rr, self.exit.dm_dot_dr * self.exit.T_0 # is this correct? dm/dA??
            )[-1] / self.exit.m_dot[-1]
        )
        self.p_0_ratio = (
            utils.cumulative_trapezoid(
                self.exit.rr, self.exit.dm_dot_dr * self.exit.p_0
            )[-1] / self.exit.m_dot[-1]
        )

        # store product of radius and tangential velocity for convenience
        rv_theta = self.exit.rr * self.exit.v_theta

        # calculate necessary derivatives for radial equilibrium
        ds_dr = np.gradient(self.exit.s, self.exit.rr, edge_order = 2)
        dv_x_dr = np.gradient(self.exit.v_x, self.exit.rr, edge_order = 2)
        drv_theta_dr = np.gradient(rv_theta, self.exit.rr, edge_order = 2)
        dT_0_dr = np.gradient(self.exit.T_0, self.exit.rr, edge_order = 2)

        # evaluate radial equilibrium
        self.exit.dr = (
            self.exit.T * ds_dr + self.exit.v_x * dv_x_dr
            + rv_theta / self.exit.rr * drv_theta_dr
            - 1 / (utils.gamma - 1) * dT_0_dr
        )

# unused?????
 
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

# unused???
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

    def old_evaluate(self, M_1, M_flight, A):
        """Determine key performance metrics local to the nozzle exit conditions."""
        # find thrust coefficient
        for exit in self.exit:

            # find dimensionless local mass flow rate
            exit.m = (
                exit.flow_state.M * np.cos(exit.flow_state.alpha)
                * np.sqrt(exit.flow_state.T) * exit.A * exit.flow_state.rho
                / (
                    M_1 * np.sqrt(utils.stagnation_temperature_ratio(M_1))
                    * utils.stagnation_density_ratio(M_1) * A
                )
            )

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
