# import modules
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import make_interp_spline
from scipy.integrate import cumulative_simpson

from streamtube import Streamtube
from flow_state import Flow_state
from annulus import Annulus
from coefficients import Coefficients
import utils
from time import perf_counter as timer

import matplotlib.pyplot as plt

# define Blade_row class
class Blade_row:
    """
    Represents a single row of blades (i.e. a rotor or a stator) and their associated parameters.
    
    Used to investigate the flow across a Rotor or a Stator. Stator is a special case of the
    Blade_row class where the blade velocity is zero. Every instance of the class will
    contain an inlet and exit flow state where all of the flow properties are stored.
    
    Parameters
    ----------
    Y_p : float
        Stagnation pressure loss coefficient.
    is_rotor : boolean
        Whether or not to categorise the blade row as a rotor or a stator.
    phi : float
        Flow coefficient.
    psi : float
        Stage loading coefficient.
    vortex_exponent : float
        Vortex exponent.
    """
    def __init__(self, Y_p, is_rotor = False, phi = None, psi = None, vortex_exponent = None):
        """Create instance of the Blade_row class."""
        # store input variables
        self.Y_p = Y_p
        self.phi = phi
        self.psi = psi
        self.vortex_exponent = vortex_exponent
        
        # hub radius is set by global hub-tip ratio
        self.r_hub = utils.Defaults.hub_tip_ratio

        # assign the default colour of black
        self.colour = 'k'

        # categorise blade row
        self.categorise(is_rotor)

        # preallocate variables as None
        #self.inlet = None
        #self.exit = None

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
    
    def old_set_inlet_conditions(self, M, alpha, N, edge = False):
        """Distributes the given inlet conditions across several annular streamtubes."""
        # set casing radius
        self.r_casing_inlet = 1

        # create list of inlet streamtubes
        self.inlet = []

        # for case where edge streamtubes are desired
        if edge:

            # set some infinitesimal streamtube thickness
            delta = utils.Defaults.delta

            # marginally close annulus temporarily
            self.r_hub += delta
            self.r_casing_inlet -= delta

        # loop over each annulus of interest
        for index in range(N):
            
            # consider annulus nearest the hub
            if index == 0:

                # find corresponding annulus radius and thickness
                r = (
                    0.5 * (
                        self.r_hub + np.sqrt(
                            self.r_hub**2 * (1 - 1 / N) + self.r_casing_inlet**2 / N
                        )
                    )
                )
                dr = r - self.r_hub

            # consider all other annuli
            else:

                # find corresponding annulus radius and thickness
                r = (
                    0.5 * (
                        self.inlet[index - 1].r + self.inlet[index - 1].dr + np.sqrt(
                            (self.inlet[index - 1].r + self.inlet[index - 1].dr)**2
                            + (self.r_casing_inlet**2 - self.r_hub**2) / N
                        )
                    )
                )
                dr = r - self.inlet[index - 1].r - self.inlet[index - 1].dr

            # store instance of the streamtube class as an inlet condition
            flow_state = Flow_state(M, alpha, 1, 1, 0)
            self.inlet.append(Streamtube(flow_state, r, dr))

        # for case where edge streamtubes are desired
        if edge:

            # undo closing of annulus
            self.r_hub -= delta
            self.r_casing_inlet += delta

            # create infinitesimal streamtube at hub
            dr = delta / 2
            r = self.r_hub + dr
            flow_state = Flow_state(M, alpha, 1, 1, 0)
            self.inlet = [Streamtube(flow_state, r, dr)] + self.inlet

            # create infinitesimal streamtube at tip
            r = self.r_casing_inlet - dr
            flow_state = Flow_state(M, alpha, 1, 1, 0)
            self.inlet.append(Streamtube(flow_state, r, dr))

    def mean_line(self):
        """Determines the mean line inlet conditions from a series of annular streamtubes."""
        # determine mean radius
        r_mean = (self.inlet[-1].r + self.inlet[-1].dr + self.r_hub) / 2

        # establish quantities of interest
        quantities = ["M", "alpha", "T_0", "p_0", "s"]
        mean_values = []

        # loop over quantities of interest
        for q in quantities:

            # fit spline and get corresponding mean value
            spline = make_interp_spline(
                [inlet.r for inlet in self.inlet],
                [getattr(inlet.flow_state, q) for inlet in self.inlet],
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

    def old_rotor_design(self, phi, psi):
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
                psi * np.power(inlet.r / self.inlet_mean.r, self.vortex_exponent - 1)
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
            vars = vars.reshape((len(self.inlet), 2))
            solutions = np.zeros_like(vars)

            # iterate over all sets of input variables
            for index, var in enumerate(vars):

                # create a holder flow_state to be populated
                flow_state = Flow_state(
                    0, 0, 0, 0, 0, 0, var[0]
                )

                # handle inner streamtube
                if index == 0:

                    # set thickness using hub radius
                    r1 = self.r_hub
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

                # apply continuity to find relative Mach number
                m_cpT0_Ap0 = (
                    utils.mass_flow_function(inlet.flow_state.M_rel)
                    * inlet.A / exit.A
                    * np.cos(inlet.flow_state.beta) / np.cos(exit.flow_state.beta)
                    / inlet.p_0_rel_ratio
                    * np.sqrt(inlet.T_0_rel_ratio)
                )

                # check if mass flow function is valid
                if m_cpT0_Ap0 > utils.mass_flow_function(1):

                    return 1e9 * np.ones_like(vars).ravel()
                
                # calculate exit relative Mach number
                exit.flow_state.M_rel = utils.invert(utils.mass_flow_function, m_cpT0_Ap0)

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

                # set up fine and coarse discretisations of spanwise positions
                """r_min = self.exit[0].r - self.exit[0].dr
                r_max = self.exit[-1].r + self.exit[-1].dr
                self.rr = np.linspace(r_min, r_max, 100 * int(np.sqrt(len(self.exit))))
                edges = np.linspace(r_min, r_max, len(self.exit))

                # calculate radial equilibrium residuals
                self.yy = np.array([
                    T_spline(r) * s_spline.derivative()(r)
                    + v_x_spline(r) * v_x_spline.derivative()(r)
                    + v_theta_spline(r) / r
                    * (v_theta_spline(r) + r * v_theta_spline.derivative()(r))
                    - 1 / (utils.gamma - 1) * T_0_spline.derivative()(r)
                    for r in self.rr
                ])

                # separate into indices
                indices = np.searchsorted(edges, self.rr, side = "right") - 1
                indices = np.clip(indices, 0, len(self.exit) - 2)

                # loop over all but one streamtube radii
                for index in range(len(self.exit) - 1):

                    # get residual values corresponding to bucket
                    mask = indices == index
                    rr_segmented = self.rr[mask]
                    yy_segmented = self.yy[mask]

                    # determine residual as least squares for bucket
                    solutions[index][1] = np.sqrt(np.trapz(yy_segmented**2, rr_segmented))"""

            # repeat iteration with new values stored
            for index, (inlet, exit) in enumerate(zip(self.inlet, self.exit)):

                # determine residual for specified stage loading
                solutions[index][0] = (
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

            # outside for loop
            # ...
            """r_min = self.exit[0].r + self.exit[0].dr
            r_max = self.exit[-1].r + self.exit[-1].dr
            #rr = [exit.r + exit.dr for exit in self.exit]
            self.rr = np.linspace(r_min, r_max, 10)
            self.yy = [
                T_spline(r) * s_spline.derivative()(r)
                + v_x_spline(r) * v_x_spline.derivative()(r)
                + v_theta_spline(r) / r
                * (v_theta_spline(r) + r * v_theta_spline.derivative()(r))
                - 1 / (utils.gamma - 1) * T_0_spline.derivative()(r)
                for r in self.rr
            ]
"""

            # final residual comes from constraint for all areas to sum to the exit area
            """solutions[-1][-1] = (
                np.sum([exit.A for exit in self.exit]) - np.sum([inlet.A for inlet in self.inlet])
            )"""

            # alternatively, final residual is constant mean axial velocity
            solutions[-1][-1] = (
                np.mean([
                    inlet.flow_state.M * np.cos(inlet.flow_state.alpha)
                    * np.sqrt(inlet.flow_state.T) for inlet in self.inlet
                ])
                - np.mean([
                    exit.flow_state.M * np.cos(exit.flow_state.alpha)
                    * np.sqrt(exit.flow_state.T) for exit in self.exit
                ])
            )

            # flatten solutions matrix and return
            solutions = solutions.ravel()
            return solutions

        # empty list of exit streamtubes - NECESSARY?
        self.exit = np.empty((len(self.inlet),), dtype = object)

        lower = [-np.pi / 2, 1e-6]
        upper = [np.pi / 2, np.pi]
        lower = np.tile(lower, (len(self.inlet), 1)).ravel()
        upper = np.tile(upper, (len(self.inlet), 1)).ravel()

        x0 = np.zeros((len(self.inlet), 2))
        for index, (inlet, exit) in enumerate(zip(self.inlet, self.exit)):

            x0[index][0] = inlet.flow_state.beta
            x0[index][1] = inlet.A

        x0 = x0.ravel()

        sol = least_squares(solve_rotor, x0, bounds = (lower, upper))
        utils.debug(f"Rotor solver iterations: {utils.Colours.GREEN}{sol.nfev}{utils.Colours.END}")
        utils.debug(f"sol: {sol}")

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

    def rotor_design(self, phi, psi):
        """Determines the rotor blade geometry necessary to satisfy the given stage parameters."""
        # store variation in static properties based on polynomial fits
        T_1 = self.inlet.T_0.value * utils.stagnation_temperature_ratio(self.inlet.M.value)
        p_1 = self.inlet.p_0.value * utils.stagnation_pressure_ratio(self.inlet.M.value)

        # get variation in blade Mach number
        M_1_blade_mean = self.inlet.M.value * np.cos(self.inlet.alpha.value) / phi
        T_mean = np.interp(self.inlet.r_mean, self.inlet.rr, T_1)
        M_1_blade = M_1_blade_mean * (self.inlet.rr / self.inlet.r_mean) * np.sqrt(T_mean / T_1)
        
        # initialise empty Coefficients instances for relative quantities
        self.inlet.M_rel = Coefficients()
        self.inlet.beta = Coefficients()
        self.inlet.T_0_rel = Coefficients()
        self.inlet.p_0_rel = Coefficients()

        # get variation in relative Mach number and flow angle via vector algebra
        z_x = self.inlet.M.value * np.cos(self.inlet.alpha.value)
        z_y = self.inlet.M.value * np.sin(self.inlet.alpha.value) - M_1_blade
        self.inlet.M_rel.value = np.hypot(z_x, z_y)
        self.inlet.beta.value = np.arctan2(z_y, z_x)

        # store corresponding coefficients for M_rel_1 and beta_1
        self.inlet.M_rel.calculate(self.inlet.rr, len(self.inlet.M.coefficients))
        self.inlet.beta.calculate(self.inlet.rr, len(self.inlet.M.coefficients))

        # get expected variation in stage loading coefficient
        psi_1 = psi * np.power(self.inlet.rr / self.inlet.r_mean, utils.Defaults.vortex_exponent)

        # get spanwise variation of relative stagnation properties
        self.inlet.T_0_rel.value = (
            self.inlet.T_0.value * utils.stagnation_temperature_ratio(self.inlet.M.value)
            / utils.stagnation_temperature_ratio(self.inlet.M_rel.value)
        )
        self.inlet.p_0_rel.value = (
            self.inlet.p_0.value * utils.stagnation_pressure_ratio(self.inlet.M.value)
            / utils.stagnation_pressure_ratio(self.inlet.M_rel.value)
        )

        # get cumulative inlet mass flow
        dm_dr_1 = (
            p_1 / np.sqrt(T_1) * self.inlet.M_rel.value * np.cos(self.inlet.beta.value)
            * self.inlet.rr
        )
        m_dot_1 = cumulative_simpson(dm_dr_1, x = self.inlet.rr, initial = 0.0)

        # get incremental change in inlet mass flow
        dm_dot_1 = np.diff(m_dot_1)

        # initialise exit annulus object to be populated
        self.exit = Annulus()

        def solve_rotor(vars):
            """Determines the matrix of residuals for a given guess of coefficients."""
            # regroup vars into shape (2, N) and store guess of exit conditions
            vars = vars.reshape(2, -1)
            self.exit.M_rel = Coefficients(vars[0])
            self.exit.beta = Coefficients(vars[1])
            self.exit.T_0_rel = Coefficients()
            self.exit.p_0_rel = Coefficients()

            # set up solutions matrix to be populated
            solutions = np.zeros_like(vars)

            # initialise vector of new radial positions
            self.exit.rr = np.zeros_like(self.inlet.rr)
            self.exit.rr[0] = self.inlet.rr[0]

            # initialise vector of relative stagnation quantities to be populated
            self.exit.T_0_rel.value = np.zeros_like(self.inlet.T_0_rel.value)
            self.exit.p_0_rel.value = np.zeros_like(self.inlet.p_0_rel.value)

            # loop over all streamtubes
            for index, m_1 in enumerate(dm_dot_1):

                # get exit inner streamtube radius and determine extra-fine grid to 
                r_2_fine = np.linspace(
                    self.exit.rr[index],
                    self.exit.rr[index] + 2 * (self.inlet.rr[index + 1] - self.inlet.rr[index]),
                    utils.Defaults.fine_grid
                )

                # evaluate relative Mach numbers and flow angles on fine, local grid
                M_2_rel = np.polyval(self.exit.M_rel.coefficients, r_2_fine)
                beta_2 = np.polyval(self.exit.beta.coefficients, r_2_fine)

                # get relative stagnation temperature from lower bound of streamtube
                self.exit.T_0_rel.value[index] = (
                    self.inlet.T_0_rel.value[index] - 0.5 * (utils.gamma - 1) * M_1_blade[index]**2 * T_1[index]
                    * (1 - (self.exit.rr[index] / self.inlet.rr[index])**2)
                )

                # get relative stagnation pressure from stagnation pressure loss coefficient
                self.exit.p_0_rel.value[index] = (
                    self.inlet.p_0_rel.value[index] * (
                        np.power(self.exit.T_0_rel.value[index] / self.inlet.T_0_rel.value[index], utils.gamma / (utils.gamma - 1))
                        - utils.Defaults.Y_p * (1 - p_1[index] / self.inlet.p_0_rel.value[index])
                    )
                )

                # get variation in mass flow rate at the inlet radial nodes
                dm_dr_2 = (
                    np.power(
                        1 + 0.5 * (utils.gamma - 1) * M_2_rel**2,
                        -utils.gamma / (utils.gamma - 1) + 0.5
                    ) * M_2_rel * np.cos(beta_2) * r_2_fine
                )
                m_dot_2 = (
                    self.exit.p_0_rel.value[index] / np.sqrt(self.exit.T_0_rel.value[index])
                    * cumulative_simpson(dm_dr_2, x = r_2_fine, initial = 0.0)
                )

                # interpolate to find upper bound of corresponding streamtube
                self.exit.rr[index + 1] = np.interp(m_1, m_dot_2, r_2_fine)

            # expand primary flow variables onto new grid
            self.exit.value("M_rel")
            self.exit.value("beta")

            # get final relative stagnation values for upper bound of streamtube
            self.exit.T_0_rel.value[-1] = (
                self.inlet.T_0_rel.value[-1] - 0.5 * (utils.gamma - 1) * M_1_blade[-1]**2 * T_1[-1]
                * (1 - (self.exit.rr[-1] / self.inlet.rr[-1])**2)
            )
            self.exit.p_0_rel.value[-1] = (
                self.inlet.p_0_rel.value[-1] * (
                    np.power(
                        self.exit.T_0_rel.value[-1] / self.inlet.T_0_rel.value[-1],
                        utils.gamma / (utils.gamma - 1)
                    ) - utils.Defaults.Y_p * (1 - p_1[-1] / self.inlet.p_0_rel.value[-1])
                )
            )

            # get variation in exit static properties
            T_2 = self.exit.T_0_rel.value * utils.stagnation_temperature_ratio(M_2_rel)
            p_2 = self.exit.p_0_rel.value * utils.stagnation_pressure_ratio(M_2_rel)

            # get exit blade Mach number distribution
            M_2_blade = M_1_blade * np.sqrt(T_1 / T_2) * self.exit.rr / self.inlet.rr

            # get absolute Mach number and flow angle via vector algebra
            z_x = self.exit.M_rel.value * np.cos(self.exit.beta.value)
            z_y = self.exit.M_rel.value * np.sin(self.exit.beta.value) + M_2_blade
            self.exit.M.value = np.hypot(z_x, z_y)
            self.exit.alpha.value = np.arctan2(z_y, z_x)

            # compare along each streamline to determine stage loading residual
            dpsi = (
                (
                    self.exit.rr / self.inlet.rr * np.sqrt(T_2 / T_1) * self.exit.M.value
                    * np.sin(self.exit.alpha.value)
                    - self.inlet.M.value * np.sin(self.inlet.alpha.value)
                ) / (M_1_blade * psi_1) - 1
            )

            # convert stage loading residuals to a (1, N) residual array
            dpsi_buckets = np.array_split(dpsi, solutions.shape[1])
            solutions[0] = np.array([np.mean(dpsi_bucket**2) for dpsi_bucket in dpsi_buckets])

            # calculate exit entropy distribution
            self.exit.s.value = (
                self.inlet.s.value + np.log(T_2 / T_1) / (utils.gamma - 1)
                - np.log(p_2 / p_1) / utils.gamma
            )

            # calculate exit stagnation temperature and pressure distributions
            self.exit.T_0.value = T_2 / utils.stagnation_temperature_ratio(self.exit.M.value)
            self.exit.p_0.value = p_2 / utils.stagnation_pressure_ratio(self.exit.M.value)

            # calculate dimensionless velocity components at exit
            v_x_2 = self.exit.M.value * np.sqrt(T_2) * np.cos(self.exit.alpha.value)
            rv_theta_2 = self.exit.rr * self.exit.M.value * np.sqrt(T_2) * np.sin(self.exit.alpha.value)

            # calculate necessary derivatives for radial equilibrium
            ds_dr = np.gradient(self.exit.s.value, self.exit.rr, edge_order = 2)
            dv_x_dr = np.gradient(v_x_2, self.exit.rr, edge_order = 2)
            drv_theta_dr = np.gradient(rv_theta_2, self.exit.rr, edge_order = 2)
            dT_0_dr = np.gradient(self.exit.T_0.value, self.exit.rr, edge_order = 2)

            # evaluate radial equilibrium
            dradial = (
                T_2 * ds_dr + v_x_2 * dv_x_dr + rv_theta_2 / self.exit.rr * drv_theta_dr
                - 1 / (utils.gamma - 1) * dT_0_dr
            )

            # convert stage loading residuals to a (1, N) residual array
            dradial_buckets = np.array_split(dradial, solutions.shape[1] - 1)
            solutions[1][:-1] = np.array([
                np.mean(dradial_bucket**2) for dradial_bucket in dradial_buckets
            ])

            # final residual comes from constant area
            solutions[1][-1] = self.exit.rr[-1]**2 - self.inlet.rr[-1]**2

            # calculate polynomial fit through absolute quantities for stator calculations
            self.exit.M.calculate(self.exit.rr, len(self.inlet.M.coefficients))
            self.exit.alpha.calculate(self.exit.rr, len(self.inlet.M.coefficients))

            # return solutions
            solutions = solutions.ravel()
            return solutions

        # set list of lower and upper bounds and reshape
        lower = np.concatenate((
            -2 * np.ones_like(self.inlet.M.coefficients),
            -np.pi * np.ones_like(self.inlet.M.coefficients)
        ))
        upper = np.concatenate((
            2 * np.ones_like(self.inlet.M.coefficients),
            np.pi * np.ones_like(self.inlet.M.coefficients)
        ))

        # get initial guess based on inlet conditions
        x0 = np.concatenate((self.inlet.M_rel.coefficients, self.inlet.beta.coefficients))
        print(f"x0: {x0}")

        # solve iteratively
        sol = least_squares(solve_rotor, x0, bounds = (lower, upper), max_nfev = utils.Defaults.nfev)
        print(f"sol: {sol}")

        print(f"self.exit.M.value: {self.exit.M.value}")

    def old_stator_design(self, last_stage = False):
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
            vars = vars.reshape((len(self.inlet), 2))
            solutions = np.zeros_like(vars)

            # iterate over all sets of input variables
            for index, var in enumerate(vars):

                # create a holder flow_state to be populated
                flow_state = Flow_state(
                    0, var[0], 0, 0, 0
                )

                # handle inner streamtube
                if index == 0:

                    # set thickness using hub radius
                    r1 = self.r_hub
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

                # create streamtube and store at exit to the stator
                self.exit[index] = Streamtube(flow_state, r, dr)

            # iterate over all inlet-exit pairs
            for index, (inlet, exit) in enumerate(zip(self.inlet, self.exit)):

                # apply continuity to find exit Mach number
                m_cpT0_Ap0 = (
                    utils.mass_flow_function(inlet.flow_state.M)
                    * inlet.A / exit.A
                    * np.cos(inlet.flow_state.alpha) / np.cos(exit.flow_state.alpha)
                    / inlet.p_0_ratio
                )

                # check if mass flow function is valid
                if m_cpT0_Ap0 > utils.mass_flow_function(1):

                    return 1e9 * np.ones_like(vars).ravel()
                
                # calculate exit Mach number
                exit.flow_state.M = utils.invert(utils.mass_flow_function, m_cpT0_Ap0)

                # stagnation temperature is conserved
                exit.flow_state.T_0 = inlet.flow_state.T_0

                # stagnation pressure is also constant apart from a small loss
                exit.flow_state.p_0 = inlet.flow_state.p_0 * inlet.p_0_ratio
                exit.flow_state.static_quantities()

                # find non-dimensional entropy
                exit.flow_state.s = (
                    inlet.flow_state.s
                    + np.log(exit.flow_state.T / inlet.flow_state.T) / (utils.gamma - 1)
                    - np.log(exit.flow_state.p / inlet.flow_state.p) / utils.gamma
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

                # set residual to be the exit angle
                solutions[index][0] = exit.flow_state.alpha

                # determine residual for radial equilibrium equation
                if index < len(self.inlet) - 1:

                    # choose point to evaluate radial equilibrium at
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

            # final residual comes from constraint for all areas to sum to the exit area
            """solutions[-1][-1] = (
                np.sum([exit.A for exit in self.exit]) - np.sum([inlet.A for inlet in self.inlet])
            )"""

            # alternatively, final residual is constant mean axial velocity
            solutions[-1][-1] = (
                np.mean([
                    inlet.flow_state.M * np.cos(inlet.flow_state.alpha)
                    * np.sqrt(inlet.flow_state.T) for inlet in self.inlet
                ])
                - np.mean([
                    exit.flow_state.M * np.cos(exit.flow_state.alpha)
                    * np.sqrt(exit.flow_state.T) for exit in self.exit
                ])
            )

            # flatten solutions matrix and return
            solutions = solutions.ravel()

            """y = np.mean([
                exit.flow_state.M * np.cos(exit.flow_state.alpha)
                * np.sqrt(inlet.flow_state.T) for exit in self.exit
            ])"""

            return solutions

        x0 = np.zeros((len(self.inlet), 2))
        for index, inlet in enumerate(self.inlet):

            # assume solution is close to the inlet conditions
            x0[index] = [0, inlet.A]

        # flatten initial guess array
        x0 = x0.ravel()

        # set lower and upper guess bounds and shape correctly
        lower = [-np.pi / 2, 0]
        upper = [np.pi / 2, np.pi]
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
        utils.debug(f"{sol}")

        # save blade angles in inlet and exit streamtubes
        for (inlet, exit) in zip(self.inlet, self.exit):

            # save stator metal angles and store blade Mach number
            inlet.metal_angle = inlet.flow_state.alpha
            exit.metal_angle = exit.flow_state.alpha
            inlet.M_blade = 0
            exit.M_blade = 0

    def stator_design(self):
        """Determines the stator blade geometry necessary to satisfy the given stage parameters."""
        # store variation in static properties based on polynomial fits
        T_1 = self.inlet.T_0.value * utils.stagnation_temperature_ratio(self.inlet.M.value)
        p_1 = self.inlet.p_0.value * utils.stagnation_pressure_ratio(self.inlet.M.value)

        # get cumulative inlet mass flow
        dm_dr_1 = (
            p_1 / np.sqrt(T_1) * self.inlet.M.value * np.cos(self.inlet.alpha.value)
            * self.inlet.rr
        )
        m_dot_1 = cumulative_simpson(dm_dr_1, x = self.inlet.rr, initial = 0.0)

        # get incremental change in inlet mass flow
        dm_dot_1 = np.diff(m_dot_1)

        # initialise exit annulus object to be populated
        self.exit = Annulus()
        #zeros = np.zeros_like(self.inlet.M.coefficients)
        self.exit.alpha.coefficients = np.zeros_like(self.inlet.M.coefficients)
        self.exit.alpha.value = np.zeros_like(self.inlet.M.value)


        # find stagnation quantities via no isentropic stagnation temperature change
        self.exit.T_0.value = self.inlet.T_0.value
        self.exit.p_0.value = (
            self.inlet.p_0.value * (1 - utils.Defaults.Y_p * (1 - p_1 / self.inlet.p_0.value))
        )

        
        def solve_stator(vars):
            """Determines the matrix of residuals for a given guess of coefficients."""
            # store guess of exit conditions
            self.exit.M.coefficients = vars

            # set up solutions matrix to be populated
            solutions = np.zeros_like(vars)
            
            # initialise vector of new radial positions
            self.exit.rr = np.zeros_like(self.inlet.rr)
            self.exit.rr[0] = self.inlet.rr[0]

            # loop over all streamtubes
            for index, (r_1_i, m_1) in enumerate(zip(self.inlet.rr[:-1], dm_dot_1)):

                # get exit inner streamtube radius and determine extra-fine grid to 
                self.exit.rr[index] = self.exit.rr[index]
                r_2_fine = np.linspace(
                    self.exit.rr[index],
                    self.exit.rr[index] + 2 * (self.inlet.rr[index + 1] - r_1_i),
                    utils.Defaults.fine_grid
                )

                # evaluate Mach numbers and flow angles on fine, local grid
                M_2 = np.polyval(self.exit.M.coefficients, r_2_fine)
                alpha_2 = np.polyval(self.exit.alpha.coefficients, r_2_fine)

                # get variation in mass flow rate at the inlet radial nodes
                dm_dr_2 = (
                    np.power(
                        1 + 0.5 * (utils.gamma - 1) * M_2,
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
            T_2 = self.exit.T_0.value * utils.stagnation_temperature_ratio(self.exit.M.value)
            p_2 = self.exit.p_0.value * utils.stagnation_pressure_ratio(self.exit.M.value)

            # calculate exit entropy distribution
            self.exit.s.value = (
                self.inlet.s.value + np.log(T_2 / T_1) / (utils.gamma - 1)
                - np.log(p_2 / p_1) / utils.gamma
            )

            # calculate dimensionless velocity components at exit
            v_x_2 = self.exit.M.value * np.sqrt(T_2) * np.cos(self.exit.alpha.value)
            rv_theta_2 = self.exit.rr * self.exit.M.value * np.sqrt(T_2) * np.sin(self.exit.alpha.value)

            # calculate necessary derivatives for radial equilibrium
            ds_dr = np.gradient(self.exit.s.value, self.exit.rr, edge_order = 2)
            dv_x_dr = np.gradient(v_x_2, self.exit.rr, edge_order = 2)
            drv_theta_dr = np.gradient(rv_theta_2, self.exit.rr, edge_order = 2)
            dT_0_dr = np.gradient(self.exit.T_0.value, self.exit.rr, edge_order = 2)

            # evaluate radial equilibrium
            dradial = (
                T_2 * ds_dr + v_x_2 * dv_x_dr + rv_theta_2 / self.exit.rr * drv_theta_dr
                - 1 / (utils.gamma - 1) * dT_0_dr
            )

            # convert stage loading residuals to a (1, N) residual array
            dradial_buckets = np.array_split(dradial, solutions.shape[0] - 1)
            solutions[:-1] = np.array([
                np.mean(dradial_bucket**2) for dradial_bucket in dradial_buckets
            ])

            # final residual comes from constant area
            solutions[-1] = self.exit.rr[-1]**2 - self.inlet.rr[-1]**2

            # return solutions
            solutions = solutions.ravel()
            return solutions
        
        # set list of lower and upper bounds and reshape
        lower = -2 * np.ones_like(self.inlet.M.coefficients)
        upper = 2 * np.ones_like(self.inlet.M.coefficients)

        # get initial guess based on inlet conditions
        x0 = self.inlet.M.coefficients
        print(f"x0: {x0}")

        # solve iteratively
        sol = least_squares(solve_stator, x0, bounds = (lower, upper), max_nfev = utils.Defaults.nfev)
        print(f"sol: {sol}")

        print(f"self.exit.M.value: {self.exit.M.value}")

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
        # remove references to r_casing_inlet!!!

        # use relative quantities for rotor
        if "Rotor" in self.label:

            M = "M_rel"
            angle = "beta"

        # use absolute quantities for stator
        else:

            M = "M"
            angle = "alpha"

        # loop over all inlet-exit pairs
        for (inlet, exit) in zip(self.inlet, self.exit):

            # calculate pitch-to-chord distribution to impose constant diffusion factor distribution
            exit.DF = utils.Defaults.DF_limit
            exit.pitch_to_chord = (
                2 * (
                    exit.DF - 1 + getattr(exit.flow_state, M) / getattr(inlet.flow_state, M)
                    * np.sqrt(exit.flow_state.T / inlet.flow_state.T)
                ) / (
                    np.sin(getattr(inlet.flow_state, angle))
                    - getattr(exit.flow_state, M) / getattr(inlet.flow_state, M)
                    * np.sqrt(exit.flow_state.T / inlet.flow_state.T)
                    * np.sin(getattr(exit.flow_state, angle))
                )
            )

            # check if pitch to chord is unachievable
            if exit.pitch_to_chord < utils.Defaults.pitch_to_chord_limit:

                # fix the pitch-to-chord and compute the new diffusion factor
                exit.pitch_to_chord = utils.Defaults.pitch_to_chord_limit
                exit.DF = (
                    1 - getattr(exit.flow_state, M) / getattr(inlet.flow_state, M)
                    * np.sqrt(exit.flow_state.T / inlet.flow_state.T) + 0.5 * (
                        np.sin(getattr(inlet.flow_state, angle))
                        - getattr(exit.flow_state, M) / getattr(inlet.flow_state, M)
                        * np.sqrt(exit.flow_state.T / inlet.flow_state.T)
                        * np.sin(getattr(exit.flow_state, angle))
                    ) * exit.pitch_to_chord
                )
            
        # set blade aspect ratio and calculate minimum number of blades
        self.N = 2
        while True:

            # calculate the dimensionless pitch and chord distributions
            for exit in self.exit:
                exit.s = 2 * np.pi * exit.r / self.N
                exit.c = exit.s / exit.pitch_to_chord
            r_mean = 0.5 * (self.exit[-1].r + self.exit[-1].dr + self.r_hub)
            c_mean = np.interp(
                r_mean,
                [exit.r for exit in self.exit],
                [exit.c for exit in self.exit]
            )
            self.AR = (self.exit[-1].r + self.exit[-1].dr - self.r_hub) / c_mean
            if self.AR > utils.Defaults.AR_target or self.N > 20:

                break

            self.N += 1

        for exit in self.exit:

            exit.AR = (self.exit[-1].r + self.exit[-1].dr - self.r_hub) / exit.c

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
