# import modules

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import root_scalar
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

from streamtube import Streamtube
from flow_state import Flow_state
import utils
from time import perf_counter as timer

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
    def __init__(self, Y_p, n, is_rotor = False):
        """Create instance of the Blade_row class."""
        # assign attributes
        self.r_casing_inlet = 1
        self.r_hub = utils.Defaults.hub_tip_ratio
        self.Y_p = Y_p
        self.n = n

        # derive inlet and exit areas
        self.r_casing_exit = 1
        self.area_inlet = np.pi * (self.r_casing_inlet**2 - self.r_hub**2)
        self.area_exit = np.pi * (self.r_casing_exit**2 - self.r_hub**2)

        # assign the default colour of black
        self.colour = 'k'

        # categorise blade row
        self.categorise(is_rotor)

        # preallocate variables as None
        self.inlet = None
        self.exit = None

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

# data handling functions -------------------------------------------------------------------------

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
    
    def set_inlet_conditions(self, M, alpha, N):
        """Distributes the given inlet conditions across several annular streamtubes."""
        # create list of inlet streamtubes and iterate over each annulus of interest
        self.inlet = []
        for index in range(N):
            
            # consider annulus nearest the hub
            if index == 0:

                # find corresponding annulus radius and thickness
                r = (
                    (utils.Defaults.hub_tip_ratio + np.sqrt(
                        utils.Defaults.hub_tip_ratio**2 * (1 - 1 / N)
                        + 1 / N
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
                        - (utils.Defaults.hub_tip_ratio**2 - 1) / N
                    )) / 2
                )
                dr = r - self.inlet[index - 1].r - self.inlet[index - 1].dr
                flow_state = Flow_state(
                    M, alpha, 1, 1, 0
                )

            # store instance of the streamtube class as an inlet condition
            self.inlet.append(Streamtube(flow_state, r, dr))

        #self.mean_line()

    def mean_line(self):
        """Determines the mean line inlet conditions from a series of annular streamtubes."""
        # determine mean radius and array of radii over which to interpolate
        r_mean = (self.inlet[-1].r + self.inlet[-1].dr + self.r_hub) / 2
        rr = [inlet.r for inlet in self.inlet]

        # establish quantities of interest and interpolate
        quantities = ["M", "alpha", "T_0", "p_0", "s"]
        mean_values = []

        # loop over all quantities of interest
        for q in quantities:

            # fit spline and get corresponding mean value
            spline = make_interp_spline(
                [inlet.r for inlet in self.inlet],
                [getattr(st.flow_state, q) for st in self.inlet],
                k = min(2, len(self.inlet) - 1)
            )
            mean_values.append(spline(r_mean))

        # store mean values
        M_mean, alpha_mean, T_0_mean, p_0_mean, s_mean = mean_values

        # create Flow_state of interpolated mean-line quantities and store as a Streamtube
        flow_state = Flow_state(
            M_mean, alpha_mean, T_0_mean, p_0_mean, s_mean
        )
        self.inlet_mean = Streamtube(
            flow_state, r_mean, 0
        )

# design functions --------------------------------------------------------------------------------

    def analyse(self):
        """Determines the rotor exit conditions given defined geometry."""
        pass

    def rotor_design(self, phi, psi):
        """Determines the rotor blade geometry necessary to satisfy the given stage parameters."""
        # determine inlet mean line parameters
        self.mean_line()

        # determine variation of several parameters across the blade span at inlet
        for inlet in self.inlet:

            # determine local flow coefficient
            inlet.phi = (
                phi
                * inlet.flow_state.M / self.inlet_mean.flow_state.M
                * np.cos(inlet.flow_state.alpha) / np.cos(self.inlet_mean.flow_state.alpha)
                * np.sqrt(inlet.flow_state.T / self.inlet_mean.flow_state.T)
                * self.inlet_mean.r / inlet.r
            )

            # determine local stage loading coefficient
            inlet.psi = (
                psi * np.power(inlet.r / self.inlet_mean.r, self.n - 1)
            )

            # determine local blade Mach number
            inlet.M_blade = (
                inlet.flow_state.M * np.cos(inlet.flow_state.alpha) / inlet.phi
            )

            # determine relative Mach number and swirl angle via vector addition 
            v1 = inlet.flow_state.M * np.array(
                [np.cos(inlet.flow_state.alpha), np.sin(inlet.flow_state.alpha)]
            )
            v2 = inlet.M_blade * np.array([0, 1])
            v3 = v1 - v2
            inlet.flow_state.M_rel = np.linalg.norm(v3)
            inlet.flow_state.beta = np.arctan2(v3[1], v3[0])

        def solve_rotor(vars):
            """Series of equations to solve the root of."""
            # reshape input variables for iteration and create empty solutions array
            vars = vars.reshape((len(self.inlet), 3))
            solutions = np.zeros_like(vars)

            # iterate over all sets of input variables
            for index, var in enumerate(vars):

                # create a holder flow_state to be populated
                flow_state = Flow_state(
                    0, 0, 0, 0, 0, var[0], var[1]
                )

                # handle inner streamtube
                if index == 0:

                    # set thickness using hub radius
                    r1 = self.r_hub
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

                # create streamtube and store at exit to the rotor
                self.exit[index] = Streamtube(flow_state, r, dr)

            # iterate over all inlet-exit pairs
            for index, (inlet, exit) in enumerate(zip(self.inlet, self.exit)):

                # determine change in relative stagnation temperature
                inlet.T_0_rel_ratio = (
                    1 - inlet.M_blade**2 * (utils.gamma - 1) / 2
                    * utils.stagnation_temperature_ratio(inlet.flow_state.M_rel)
                    * (1 - (exit.r / inlet.r)**2)
                )

                # determine change in relative stagnation pressure
                inlet.p_0_rel_ratio = (
                    np.power(inlet.T_0_rel_ratio, utils.gamma / (utils.gamma - 1))
                    - self.Y_p * (1 - utils.stagnation_pressure_ratio(inlet.flow_state.M_rel))
                )

                # find local blade Mach number at exit to the rotor row
                exit.M_blade = (
                    inlet.M_blade * np.sqrt(
                        utils.stagnation_temperature_ratio(inlet.flow_state.M_rel)
                        / utils.stagnation_temperature_ratio(exit.flow_state.M_rel)
                        * inlet.T_0_rel_ratio
                    )
                    * exit.r / inlet.r
                )

                # determine absolute Mach number and swirl angle at exit via vector addition 
                v1 = exit.flow_state.M_rel * np.array([np.cos(exit.flow_state.beta), np.sin(exit.flow_state.beta)])
                v2 = exit.M_blade * np.array([0, 1])
                v3 = v1 + v2
                exit.flow_state.M = np.linalg.norm(v3)
                exit.flow_state.alpha = np.arctan2(v3[1], v3[0])

                # find exit stagnation temperature
                exit.flow_state.T_0 = (
                    inlet.flow_state.T_0 * (
                        1 + (utils.gamma - 1) * inlet.psi * inlet.M_blade**2
                        * utils.stagnation_temperature_ratio(inlet.flow_state.M)
                    )
                )
                exit.flow_state.static_quantities()

                # find non-dimensional entropy
                exit.flow_state.s = (
                    inlet.flow_state.s + np.log(
                        exit.flow_state.T / inlet.flow_state.T
                    ) / (utils.gamma - 1)
                    - np.log(
                        utils.stagnation_pressure_ratio(exit.flow_state.M_rel)
                        / utils.stagnation_pressure_ratio(inlet.flow_state.M_rel)
                        * inlet.p_0_rel_ratio
                    ) / utils.gamma
                )

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
                    utils.mass_flow_function(inlet.flow_state.M_rel)
                    * np.sqrt(inlet.T_0_rel_ratio)
                    * inlet.A / exit.A
                    * np.cos(inlet.flow_state.beta) / np.cos(exit.flow_state.beta)
                    / inlet.p_0_rel_ratio
                    - utils.mass_flow_function(exit.flow_state.M_rel)
                )

                # determine residual for specified stage loading
                solutions[index][1] = (
                    inlet.psi - (
                        exit.flow_state.M * np.sin(exit.flow_state.alpha)
                        * exit.r / inlet.r * np.sqrt(
                            exit.flow_state.T / inlet.flow_state.T
                        ) - inlet.flow_state.M * np.sin(inlet.flow_state.alpha)
                    ) / inlet.M_blade
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

            # final residual comes from constraint for all areas to sum to the exit area
            solutions[-1][-1] = (
                np.sum([exit.A for exit in self.exit]) - self.area_exit
            )
            # alternatively, final residual is constant mean axial velocity
            """solutions[-1][-1] = (
                np.mean([
                    inlet.flow_state.M * np.cos(inlet.flow_state.alpha)
                    * np.sqrt(inlet.flow_state.T) for inlet in self.inlet
                ])
                - np.mean([
                    exit.flow_state.M * np.cos(exit.flow_state.alpha)
                    * np.sqrt(inlet.flow_state.T) for exit in self.exit
                ])
            )"""

            # flatten solutions matrix and return
            solutions = solutions.ravel()
            return solutions

        def residual(M_rel, inlet):
            """Find exit relative Mach number which reduces residual to zero assuming constant radius."""
            # guess the value of cos(beta) via stage loading coefficient
            sin_beta = (
                (
                    (inlet.psi - 1) * inlet.M_blade
                    + inlet.flow_state.M * np.sin(inlet.flow_state.alpha)
                ) / (
                    M_rel * np.sqrt(
                        utils.stagnation_temperature_ratio(M_rel)
                        / utils.stagnation_temperature_ratio(inlet.flow_state.M_rel)
                    )
                )
            )

            # guess the value of cos(beta) via conservation of mass
            cos_beta = (
                utils.mass_flow_function(inlet.flow_state.M_rel)
                * np.cos(inlet.flow_state.beta)
                / utils.mass_flow_function(M_rel)
            )

            return sin_beta**2 + cos_beta**2 - 1

        # debugging
        """xx = np.linspace(1e-3, 1, 50)
        yy = [residual(x) for x in xx]
        fig, ax = plt.subplots()
        ax.plot(xx, yy)
        plt.show()"""

        # initialise array to store initial guess and loop over all inlet streamtubes
        x0 = np.zeros((len(self.inlet), 3))
        for index, inlet in enumerate(self.inlet):

            # create initial guess
            a0 = 1e-6
            a1 = inlet.flow_state.M_rel

            # solve for an exit relative Mach number recursively, assuming constant radius
            sol = root_scalar(
                residual, x0 = a0, x1 = a1, method = "secant",
                args = (inlet,)
            )
            x0[index][0] = sol.root

            # find the corresponding exit relative flow angle
            x0[index][1] = (
                np.arcsin(
                    (
                        (inlet.psi - 1) * inlet.M_blade
                        + inlet.flow_state.M * np.sin(inlet.flow_state.alpha)
                    ) / (
                        sol.root * np.sqrt(
                            utils.stagnation_temperature_ratio(sol.root)
                            / utils.stagnation_temperature_ratio(inlet.flow_state.M_rel)
                        )
                    )
                )
            )

            # key assumption behind initial guess is that radius is constant
            x0[index][2] = inlet.A

        # no previous guess is available
        if type(self.exit) == type(None):

            pass
        
        # exit conditions from a previous iteration are known
        elif len(self.inlet) == len(self.exit):

            for index, (inlet, exit) in enumerate(zip(self.inlet, self.exit)):

                x0[index] = [
                    exit.flow_state.M_rel, exit.flow_state.beta, exit.A
                ]

        # flatten initial guess array
        x0 = x0.ravel()

        # set lower and upper guess bounds and shape correctly
        lower = [0, -np.pi / 2, 0]
        upper = [1, np.pi / 2, np.pi]
        lower = np.tile(lower, (len(self.inlet), 1)).ravel()
        upper = np.tile(upper, (len(self.inlet), 1)).ravel()

        # check for out of bounds error
        for (x, low, up) in zip(x0, lower, upper):

            if x < low or x > up:

                print(f"{utils.Colours.RED}Out of bounds error occurred!{utils.Colours.END}")
                print(f"x0: {x0}")
                print(f"lower: {lower}")
                print(f"upper: {upper}")

        # empty list of exit streamtubes
        self.exit = np.empty((len(self.inlet),), dtype = object)

        # solve for least squares solution
        sol = least_squares(solve_rotor, x0, bounds = (lower, upper))
        self.A_exit = np.sum([exit.A for exit in self.exit])
        utils.debug(f"Rotor solver iterations: {utils.Colours.GREEN}{sol.nfev}{utils.Colours.END}")

        # iterate over all inlet-exit pairs
        for (inlet, exit) in zip(self.inlet, self.exit):

            # find new stagnation pressure and use to find static pressure
            exit.flow_state.p_0 = (
                inlet.flow_state.p_0
                / utils.stagnation_pressure_ratio(exit.flow_state.M)
                * utils.stagnation_pressure_ratio(exit.flow_state.M_rel)
                / utils.stagnation_pressure_ratio(inlet.flow_state.M_rel)
                * utils.stagnation_pressure_ratio(inlet.flow_state.M)
                * inlet.p_0_rel_ratio
            )
            exit.flow_state.static_quantities()

            # save blade angles in inlet and exit streamtubes
            inlet.metal_angle = inlet.flow_state.beta
            exit.metal_angle = exit.flow_state.beta

    def stator_design(self, last_stage = False):
        """Determines the stator blade geometry necessary to satisfy the given stage parameters."""
        # for now treat every stage as last stage
        last_stage = True

        # clean inlet of relative quantities
        for inlet in self.inlet:

            inlet.flow_state.M_rel = None
            inlet.flow_state.beta = None
            inlet.flow_state.T_0_rel = None
            inlet.flow_state.p_0_rel = None

        # determine variation of several parameters across the blade span at inlet
        for inlet in self.inlet:

            # determine relative stagnation pressure loss
            inlet.p_0_ratio = (
                1 - self.Y_p * (1 - utils.stagnation_pressure_ratio(inlet.flow_state.M))
            )

        # create empty array of exit streamtubes
        self.exit = np.empty((len(self.inlet),), dtype = object)

        def solve_stator(vars):
            """Series of equations to solve the root of."""
            # reshape input variables for iteration and create empty solutions array
            vars = vars.reshape((len(self.inlet), 3))
            solutions = np.zeros_like(vars)

            # iterate over all sets of input variables
            for index, var in enumerate(vars):

                # create a holder flow_state to be populated
                flow_state = Flow_state(
                    var[0], var[1], 0, 0, 0
                )

                # handle inner streamtube
                if index == 0:

                    # set thickness using hub radius
                    r1 = self.r_hub
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

                # create streamtube and store at exit to the stator
                self.exit[index] = Streamtube(flow_state, r, dr)

            # iterate over all inlet-exit pairs
            for index, (inlet, exit) in enumerate(zip(self.inlet, self.exit)):

                # stagnation temperature is conserved
                exit.flow_state.T_0 = inlet.flow_state.T_0

                # stagnation pressure is also constant apart from a small loss
                exit.flow_state.p_0 = inlet.flow_state.p_0 * inlet.p_0_ratio
                exit.flow_state.static_quantities()

                # find non-dimensional entropy
                exit.flow_state.s = (
                    inlet.flow_state.s + np.log(
                        exit.flow_state.T / inlet.flow_state.T
                    ) / (utils.gamma - 1)
                    - np.log(
                        exit.flow_state.p / inlet.flow_state.p
                    ) / utils.gamma
                )

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

                # fit spline for v_theta * r term
                v_theta_r_spline = make_interp_spline(
                    [exit.r for exit in self.exit],
                    [exit.flow_state.v_theta * exit.r for exit in self.exit],
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
                    / inlet.p_0_ratio
                    - utils.mass_flow_function(exit.flow_state.M)
                )

                # set residual to be the exit angle
                solutions[index][1] = exit.flow_state.alpha

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

            # final residual comes from constraint for all areas to sum to the exit area
            solutions[-1][-1] = (
                np.sum([exit.A for exit in self.exit]) - self.area_exit
            )
            # alternatively, final residual is constant mean axial velocity
            """solutions[-1][-1] = (
                np.mean([
                    inlet.flow_state.M * np.cos(inlet.flow_state.alpha)
                    * np.sqrt(inlet.flow_state.T) for inlet in self.inlet
                ])
                - np.mean([
                    exit.flow_state.M * np.cos(exit.flow_state.alpha)
                    * np.sqrt(inlet.flow_state.T) for exit in self.exit
                ])
            )"""

            # flatten solutions matrix and return
            solutions = solutions.ravel()
            return solutions
        
        # initialise array to store initial guess and iterate
        x0 = np.zeros((len(self.inlet), 3))
        for index, inlet in enumerate(self.inlet):

            # assume solution is close to the inlet conditions
            x0[index] = [
                0.9 * inlet.flow_state.M,
                0,
                inlet.A
            ]

        # flatten initial guess array
        x0 = x0.ravel()

        # set lower and upper guess bounds and shape correctly
        lower = [0, -np.pi / 2, 0]
        upper = [1, np.pi / 2, np.pi]
        lower = np.tile(lower, (len(self.inlet), 1)).ravel()
        upper = np.tile(upper, (len(self.inlet), 1)).ravel()

        # check for out of bounds error
        for (x, low, up) in zip(x0, lower, upper):

            if x < low or x > up:

                print(f"{utils.Colours.RED}Out of bounds error occurred!{utils.Colours.END}")
                print(f"x0: {x0}")
                print(f"lower: {lower}")
                print(f"upper: {upper}")

        # solve for least squares solution
        sol = least_squares(solve_stator, x0, bounds = (lower, upper))
        self.A_exit = np.sum([exit.A for exit in self.exit])
        utils.debug(f"Stator solver iterations: {utils.Colours.GREEN}{sol.nfev}{utils.Colours.END}")

        # save blade angles in inlet and exit streamtubes
        for (inlet, exit) in zip(self.inlet, self.exit):

            # save stator metal angles and store blade Mach number
            inlet.metal_angle = inlet.flow_state.alpha
            exit.metal_angle = exit.flow_state.alpha
            inlet.M_blade = 0
            exit.M_blade = 0

    def deviation(self):
        """Calculate local deviation across the blade span using Carter and Howell."""
        # use relative angles for rotor
        if "Rotor" in self.label:

            angle = "beta"

        # use absolute angles for stator
        else:

            angle = "alpha"

        # iterate over all inlet-exit pairs
        for index, (inlet, exit) in enumerate(zip(self.inlet, self.exit)):

            # store inlet and exit angle in degrees for convenience
            inlet_angle = utils.rad_to_deg(getattr(inlet.flow_state, angle))
            exit_angle = utils.rad_to_deg(getattr(exit.flow_state, angle))

            # calculate deviation coefficient using Howell's correlation for a circular camber line
            m = 0.23 + exit_angle / 500

            # calculate exit metal angle using Carter's correlation - CHECK SIGNS HERE
            exit.metal_angle = (
                utils.deg_to_rad(
                    exit_angle - m * inlet_angle * np.sqrt(exit.pitch_to_chord)
                    / (1 + m * np.sqrt(exit.pitch_to_chord))
                )
            )
            exit.deviation = getattr(exit.flow_state, angle) - exit.metal_angle

    def empirical_design(self):
        """Applies empirical relations to design pitch-to-chord and deviation distributions."""
        # use relative Mach numbers for rotor
        if "Rotor" in self.label:

            M = "M_rel"

        # use absolute Mach numbers for stator
        else:

            M = "M"

        # loop over all inlet-exit pairs
        for (inlet, exit) in zip(self.inlet, self.exit):

            # calculate pitch-to-chord distribution to impose constant diffusion factor distribution
            exit.DF = utils.Defaults.DF_limit
            exit.pitch_to_chord = (
                (
                    exit.DF - 1 + getattr(exit.flow_state, M) / getattr(inlet.flow_state, M)
                    * np.sqrt(exit.flow_state.T / inlet.flow_state.T)
                ) * 2 * getattr(inlet.flow_state, M) / (
                    exit.flow_state.M * np.sin(exit.flow_state.alpha)
                    * np.sqrt(exit.flow_state.T / inlet.flow_state.T)
                    - inlet.flow_state.M * np.sin(inlet.flow_state.alpha)
                )
            )

            # check if pitch to chord is unachievable
            if exit.pitch_to_chord < 0:

                # fix the pitch-to-chord and compute the new diffusion factor
                exit.pitch_to_chord = utils.Defaults.pitch_to_chord_limit
                exit.DF = (
                    1 - getattr(exit.flow_state, M) / getattr(inlet.flow_state, M)
                    * np.sqrt(exit.flow_state.T / inlet.flow_state.T) + (
                        exit.flow_state.M * np.sin(exit.flow_state.alpha)
                        * np.sqrt(exit.flow_state.T / inlet.flow_state.T)
                        - inlet.flow_state.M * np.sin(inlet.flow_state.alpha)
                    ) * exit.pitch_to_chord / (2 * getattr(inlet.flow_state, M))
                )

            # calculate the dimensionless pitch and chord distributions
            self.N = 5 # set for now
            exit.s = 2 * np.pi * exit.r / self.N
            exit.c = exit.s / exit.pitch_to_chord

        # calculate deviation and blade metal angles
        self.deviation()

# plotting functions ------------------------------------------------------------------------------

    def draw_blades(self):
        """Creates a series of x- and y- coordinates based on the blade shape data."""
        # initialise arrays to store blade shape data in
        self.xx = np.empty(len(self.inlet), dtype=object)
        self.yy = np.empty(len(self.inlet), dtype=object)

        # loop over all inlet and exit angles
        for index, (inlet, exit) in enumerate(zip(self.inlet, self.exit)):

            # construct circular camber line
            r = 1 / (exit.metal_angle - inlet.metal_angle)
            x0 = -r * np.sin(inlet.metal_angle)
            y0 = r * np.cos(inlet.metal_angle)
            theta = np.linspace(inlet.metal_angle, exit.metal_angle, 100)
            xx_0 = x0 + r * np.sin(theta)
            yy_0 = y0 - r * np.cos(theta)

            # determine cumulative length of chord line
            ll_0 = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(xx_0)**2 + np.diff(yy_0)**2))])

            # calculate derivatives in x and y with respect to l
            dx_dl = np.gradient(xx_0, ll_0)
            dy_dl = np.gradient(yy_0, ll_0)

            # compute unit normals
            norm = np.sqrt(dx_dl**2 + dy_dl**2)
            nx = -dy_dl / norm
            ny =  dx_dl / norm

            # get the upper surface only from the imported aerofoil data and sort
            zz_0 = utils.aerofoil_data
            zz_0 = zz_0[:, zz_0[0].argsort()]

            # initialise empty arrays for the upper and lower surfaces
            xx_upper = np.zeros(xx_0.shape)
            xx_lower = np.zeros(xx_0.shape)
            yy_upper = np.zeros(xx_0.shape)
            yy_lower = np.zeros(xx_0.shape)

            # iterate over each datapoint
            for i, (x, y, l) in enumerate(zip(xx_0, yy_0, ll_0)):

                # add thickness to the camberline
                dy = np.interp(l, *zz_0)
                nx_i = np.interp(l, ll_0, nx)
                ny_i = np.interp(l, ll_0, ny)
                xx_upper[i] = x + dy * nx_i
                yy_upper[i] = y + dy * ny_i
                xx_lower[i] = x - dy * nx_i
                yy_lower[i] = y - dy * ny_i

            # reverse upper surface
            xx_upper = xx_upper[::-1]
            yy_upper = yy_upper[::-1]

            # combine upper and lower surfaces
            self.xx[index] = np.concatenate([xx_upper, xx_lower])
            self.yy[index] = np.concatenate([yy_upper, yy_lower])

        # resize
        self.xx = self.xx / 2
        self.yy = self.yy / 2

    def plot_blade_row(self, ax, index, j, k, scaling = 1):
        """Plots a blade row onto a given axes at a specified spanwise position."""
        # plot blade shape
        self.draw_blades()
        ax.plot(self.xx[k] + index, self.yy[k] + j, color = self.colour)

        # store inlet and exit for convenience
        inlet = self.inlet[k]
        exit = self.exit[k]

        # get trailing edge coordinates
        x_te, y_te = self.xx[k][0], self.yy[k][0]

        # helper function to simplify plotting arrows
        def plot_arrow(z1, z2, colour = 'k'):
            """Plot an arrow from (x1, y1) to (x2, y2) in a given colour."""
            ax.annotate(
                "",
                xy = z2,
                xytext = z1,
                arrowprops = dict(
                    arrowstyle = "->", color = colour,
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )
            ax.plot([z1[0]] + [z2[0]], [z1[1]] + [z2[1]], linestyle = '')

        # only plot rotating quantities if blade is a rotor
        if "Rotor" in self.label:
                
            # display relative velocity vector at blade row inlet
            plot_arrow(
                (
                    index - scaling * inlet.flow_state.M_rel * np.cos(inlet.flow_state.beta),
                    j - scaling * inlet.flow_state.M_rel * np.sin(inlet.flow_state.beta)
                ),
                (index, j),
                colour = 'C4'
            )

            # display relative velocity vector at blade row exit
            plot_arrow(
                (x_te + index, y_te + j),
                (
                    x_te + index + scaling * exit.flow_state.M_rel * np.cos(exit.flow_state.beta),
                    y_te + j + scaling * exit.flow_state.M_rel * np.sin(exit.flow_state.beta)
                ),
                colour = 'C4'
            )

            # display blade row speed vector at blade row inlet
            plot_arrow(
                (index, j),
                (index, j + scaling * inlet.M_blade),
                colour = 'C3'
            )

            # display blade row speed vector at blade row exit
            plot_arrow(
                (
                    x_te + index + scaling * exit.flow_state.M_rel * np.cos(exit.flow_state.beta),
                    y_te + j + scaling * exit.flow_state.M_rel * np.sin(exit.flow_state.beta)
                ),
                (
                    x_te + index + scaling * exit.flow_state.M * np.cos(exit.flow_state.alpha),
                    y_te + j + scaling * exit.flow_state.M * np.sin(exit.flow_state.alpha)
                ),
                colour = 'C3'
            )

        # display absolute velocity vector at blade row inlet
        plot_arrow(
            (
                index - scaling * inlet.flow_state.M * np.cos(inlet.flow_state.alpha),
                j + scaling * (inlet.M_blade - inlet.flow_state.M * np.sin(inlet.flow_state.alpha))
            ),
            (index, j + scaling * inlet.M_blade),
            colour = 'C0'
        )

        # display absolute velocity vector at blade row exit
        plot_arrow(
            (x_te + index, y_te + j),
            (
                x_te + index + scaling * exit.flow_state.M * np.cos(exit.flow_state.alpha),
                y_te + j + scaling * exit.flow_state.M * np.sin(exit.flow_state.alpha)
            ),
            colour = 'C0'
        )

# IGNORE EVERYTHING FROM HERE -------------------------

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