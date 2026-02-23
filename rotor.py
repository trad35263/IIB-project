# import modules
import numpy as np
from scipy.optimize import least_squares
from time import perf_counter as timer
import matplotlib.pyplot as plt

# import custom classes
from annulus import Annulus
from coefficients import Coefficients
import utils
from blade_row import Blade_row

# define Rotor class
class Rotor(Blade_row):
    """
    Represents a single compressor rotor and stores the associated flowfield.
    
    Used to investigate the flow across a rotor. Every instance of the class will
    contain an inlet and exit flow object where all of the flow properties are stored.
    
    Parameters
    ----------
    Y_p : float
        Stagnation pressure loss coefficient.
    phi : float
        Flow coefficient.
    psi : float
        Stage loading coefficient.
    vortex_exponent : float
        Vortex exponent.
    """
    def __init__(self, Y_p, phi, psi, vortex_exponent):
        """Create instance of the Rotor class."""
        #
        super().__init__(Y_p, phi, psi, vortex_exponent)

        # store input variables
        self.Y_p = Y_p
        self.phi_mean = phi
        self.psi_mean = psi
        self.n = vortex_exponent
        
        # hub radius is set by global hub-tip ratio
        self.r_hub = utils.Defaults.hub_tip_ratio

        # assign the default colour of black
        self.colour = 'k'

        # create empty inlet and exit Annulus instances
        self.inlet = Annulus()
        self.exit = Annulus()

    def __str__(self):
        """Prints a string representation of the rotor."""
        # create empty string and loop over all class attributes
        string = ""
        for name, value in self.__dict__.items():

            # check if attribute is numeric
            if isinstance(value, (int, float)):

                # check if attribute is an angle
                if ("alpha" in name or "angle" in name or "beta" in name) and not value == 0:

                    # append value to string in degrees
                    string += f"{name}: {utils.Colours.GREEN}{utils.rad_to_deg(value):.4g} °"
                    string += f"{utils.Colours.END}\n"

                # all other attributes
                else:

                    # append value to string
                    string += f"{name}: {utils.Colours.GREEN}{value:.4g}{utils.Colours.END}\n"

        return string

    def old_design(self, phi_mean, psi_mean, n):
        """Determines the rotor blade geometry necessary to satisfy the given stage parameters."""        
        # initialise empty Coefficients instances for relative quantities
        self.inlet.M_rel = Coefficients()
        self.inlet.beta = Coefficients()
        self.inlet.T_0_rel = Coefficients()
        self.inlet.p_0_rel = Coefficients()

        # calculate mid-span radius
        self.inlet.r_mean = np.sqrt(0.5 * (self.inlet.rr[0]**2 + self.inlet.rr[-1]**2))

        # get variation in blade Mach number
        M_1_blade_mean = self.inlet.M.value * np.cos(self.inlet.alpha.value) / phi_mean
        T_mean = np.interp(self.inlet.r_mean, self.inlet.rr, self.inlet.T.value)
        M_1_blade = M_1_blade_mean * (self.inlet.rr / self.inlet.r_mean) * np.sqrt(T_mean / self.inlet.T.value)

        # get variation in relative Mach number and flow angle via vector algebra
        z_x = self.inlet.M.value * np.cos(self.inlet.alpha.value)
        z_y = self.inlet.M.value * np.sin(self.inlet.alpha.value) - M_1_blade
        self.inlet.M_rel.value = np.hypot(z_x, z_y)
        self.inlet.beta.value = np.arctan2(z_y, z_x)

        # store corresponding coefficients for M_rel_1 and beta_1
        self.inlet.M_rel.calculate(self.inlet.rr, len(self.inlet.M.coefficients))

        # get variation in stage loading coefficient
        psi = psi_mean * np.power(self.inlet.r_mean / self.inlet.rr, n + 1)

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
            self.inlet.p.value / np.sqrt(self.inlet.T.value) * self.inlet.M_rel.value
            * np.cos(self.inlet.beta.value) * self.inlet.rr
        )
        m_dot_1 = utils.cumulative_trapezoid(dm_dr_1, self.inlet.rr)

        # get incremental change in inlet mass flow
        dm_dot_1 = np.diff(m_dot_1)

        def solve_rotor(vars):
            """Determines the matrix of residuals for a given guess of coefficients."""
            # create Coefficients objects for each of the relative quantities
            t1 = timer()
            self.exit.M_rel = Coefficients(vars)
            self.exit.beta = Coefficients()
            self.exit.T_0_rel = Coefficients()
            self.exit.p_0_rel = Coefficients()

            # set up solutions matrix to be populated
            solutions = np.zeros_like(vars)

            # initialise vector of new radial positions
            self.exit.rr = np.zeros_like(self.inlet.rr)
            self.exit.rr[0] = self.inlet.rr[0]

            # initialise vector of relative stagnation quantities to be populated
            self.exit.beta.value = np.zeros_like(self.inlet.M.value)
            self.exit.T_0_rel.value = np.zeros_like(self.inlet.M.value)
            self.exit.p_0_rel.value = np.zeros_like(self.inlet.M.value)

            # pre-compute constants used in loop
            half_gamma_1 = 0.5 * (utils.gamma - 1)
            gamma_ratio = utils.gamma / (utils.gamma - 1)

            # loop over all streamtubes
            for index, m_dot_1 in enumerate(dm_dot_1):

                # determine extra-fine grid from inner streamtube radius to outer radius 
                r_2_fine = np.linspace(
                    self.exit.rr[index],
                    self.exit.rr[index] + 5 * (self.inlet.rr[index + 1] - self.inlet.rr[index]),
                    utils.Defaults.solver_grid
                )

                # evaluate relative Mach number on fine, local grid
                M_2_rel = np.polyval(self.exit.M_rel.coefficients, r_2_fine)

                # get relative stagnation temperature from lower bound of streamtube
                self.exit.T_0_rel.value[index] = (
                    self.inlet.T_0_rel.value[index] + half_gamma_1 * M_1_blade[index]**2
                    * self.inlet.T.value[index]
                    * ((self.exit.rr[index] / self.inlet.rr[index])**2 - 1)
                )

                # get relative stagnation pressure from stagnation pressure loss coefficient
                self.exit.p_0_rel.value[index] = (
                    self.inlet.p_0_rel.value[index] * (
                        np.power(
                            self.exit.T_0_rel.value[index] / self.inlet.T_0_rel.value[index],
                            gamma_ratio
                        ) - self.Y_p
                        * (1 - self.inlet.p.value[index] / self.inlet.p_0_rel.value[index])
                    )
                )

                # get exit relative flow angle from specified stage loading
                self.exit.beta.value[index] = (
                    np.arcsin(
                        (
                            M_1_blade[index] * (
                                psi[index] - (self.exit.rr[index] / self.inlet.rr[index])**2
                            ) + self.inlet.M.value[index] * np.sin(self.inlet.alpha.value[index])
                        ) / (
                            self.exit.rr[index] / self.inlet.rr[index] * M_2_rel[0] * np.sqrt(
                                utils.stagnation_temperature_ratio(M_2_rel[0])
                                / utils.stagnation_temperature_ratio(self.inlet.M_rel.value[index])
                                * self.exit.T_0_rel.value[index] / self.inlet.T_0_rel.value[index]
                            )
                        )
                    )
                )

                # get variation in mass flow rate at the inlet radial nodes
                dm_dr_2 = (
                    np.power(
                        1 + half_gamma_1 * M_2_rel**2, - gamma_ratio + 0.5
                    ) * M_2_rel * np.cos(self.exit.beta.value[index]) * r_2_fine
                )
                m_dot_2 = (
                    self.exit.p_0_rel.value[index] / np.sqrt(self.exit.T_0_rel.value[index])
                    * utils.cumulative_trapezoid(dm_dr_2, r_2_fine)
                )

                # interpolate to find upper bound of corresponding streamtube
                self.exit.rr[index + 1] = np.interp(m_dot_1, m_dot_2, r_2_fine)

            # expand primary flow variables onto new grid
            self.exit.value("M_rel")

            # get final relative stagnation values for upper bound of streamtube
            self.exit.T_0_rel.value[-1] = (
                self.inlet.T_0_rel.value[-1] + half_gamma_1 * M_1_blade[-1]**2
                * self.inlet.T.value[-1] * ((self.exit.rr[-1] / self.inlet.rr[-1])**2 - 1)
            )
            self.exit.p_0_rel.value[-1] = (
                self.inlet.p_0_rel.value[-1] * (
                    np.power(
                        self.exit.T_0_rel.value[-1] / self.inlet.T_0_rel.value[-1],
                        gamma_ratio
                    ) - self.Y_p
                    * (1 - self.inlet.p.value[-1] / self.inlet.p_0_rel.value[-1])
                )
            )

            # get final relative flow angle for upper bound of streamtube
            self.exit.beta.value[-1] = (
                np.arcsin(
                    (
                        M_1_blade[-1] * (
                            psi[-1] - (self.exit.rr[-1] / self.inlet.rr[-1])**2
                        ) + self.inlet.M.value[-1] * np.sin(self.inlet.alpha.value[-1])
                    ) / (
                        self.exit.rr[-1] / self.inlet.rr[-1] * self.exit.M_rel.value[-1] * np.sqrt(
                            utils.stagnation_temperature_ratio(self.exit.M_rel.value[-1])
                            / utils.stagnation_temperature_ratio(self.inlet.M_rel.value[-1])
                            * self.exit.T_0_rel.value[-1] / self.inlet.T_0_rel.value[-1]
                        )
                    )
                )
            )

            # get variation in exit static properties
            self.exit.T.value = (
                self.exit.T_0_rel.value * utils.stagnation_temperature_ratio(self.exit.M_rel.value)
            )
            self.exit.p.value = (
                self.exit.p_0_rel.value * utils.stagnation_pressure_ratio(self.exit.M_rel.value)
            )

            # get exit blade Mach number distribution
            M_2_blade = (
                M_1_blade * np.sqrt(self.inlet.T.value / self.exit.T.value)
                * self.exit.rr / self.inlet.rr
            )

            # get absolute Mach number and flow angle via vector algebra
            z_x = self.exit.M_rel.value * np.cos(self.exit.beta.value)
            z_y = self.exit.M_rel.value * np.sin(self.exit.beta.value) + M_2_blade
            self.exit.M.value = np.hypot(z_x, z_y)
            self.exit.alpha.value = np.arctan2(z_y, z_x)

            # calculate exit stagnation temperature and pressure distributions
            self.exit.T_0.value = (
                self.exit.T.value / utils.stagnation_temperature_ratio(self.exit.M.value)
            )
            self.exit.p_0.value = (
                self.exit.p.value / utils.stagnation_pressure_ratio(self.exit.M.value)
            )

            # compare along each streamline to determine stage loading residual
            self.exit.psi = (
                (self.exit.T_0.value - self.inlet.T_0.value)
                / (self.inlet.T.value * (utils.gamma - 1) * M_1_blade**2)
            )
            #self.exit.dpsi = self.exit.psi / psi - 1

            # calculate exit entropy distribution
            self.exit.s.value = (
                self.inlet.s.value
                + np.log(self.exit.T.value / self.inlet.T.value) / (utils.gamma - 1)
                - np.log(self.exit.p.value / self.inlet.p.value) / utils.gamma
            )

            # calculate dimensionless velocity components at exit
            v_x_2 = self.exit.M.value * np.sqrt(self.exit.T.value) * np.cos(self.exit.alpha.value)
            rv_theta_2 = (
                self.exit.rr * self.exit.M.value * np.sqrt(self.exit.T.value)
                * np.sin(self.exit.alpha.value)
            )

            # calculate necessary derivatives for radial equilibrium
            ds_dr = np.gradient(self.exit.s.value, self.exit.rr, edge_order = 1)
            dv_x_dr = np.gradient(v_x_2, self.exit.rr, edge_order = 1)
            drv_theta_dr = np.gradient(rv_theta_2, self.exit.rr, edge_order = 1)
            dT_0_dr = np.gradient(self.exit.T_0.value, self.exit.rr, edge_order = 1)

            # evaluate radial equilibrium
            self.exit.dr = (
                self.exit.T.value * ds_dr + v_x_2 * dv_x_dr
                + rv_theta_2 / self.exit.rr * drv_theta_dr
                - 1 / (utils.gamma - 1) * dT_0_dr
            )

            # convert stage loading residuals to a (1, N) residual array
            dr_buckets = np.array_split(self.exit.dr, solutions.shape[0] - 1)
            #dr_buckets = np.array_split(self.exit.dr, solutions.shape - 1)
            """solutions[1][:-1] = np.array([
                np.sqrt(np.mean(bucket**2)) for bucket in dr_buckets
            ])"""
            solutions[:-1] = np.array([
                np.sqrt(np.mean(bucket**2)) for bucket in dr_buckets
            ])

            # final residual comes from constant area
            #solutions[1][-1] = self.exit.rr[-1]**2 - self.inlet.rr[-1]**2
            solutions[-1] = self.exit.rr[-1]**2 - self.inlet.rr[-1]**2

            # return solutions
            #solutions = solutions.ravel()
            return solutions

        # set list of lower and upper bounds and reshape
        lower = -5 * np.ones_like(self.inlet.M.coefficients)
        upper = 5 * np.ones_like(self.inlet.M.coefficients)

        # initialise exit annulus object to be populated
        self.exit = Annulus()

        # set initial guess based on inlet conditions
        #x0 = np.concatenate((self.inlet.M_rel.coefficients, self.inlet.beta.coefficients))
        x0 = self.inlet.M_rel.coefficients

        # solve iteratively
        sol = least_squares(
            solve_rotor, x0, bounds = (lower, upper),
            xtol = 1e-6, ftol = 1e-6, gtol = 1e-6,
            x_scale = 'jac'
        )
        utils.debug(f"sol: {sol}")

        # get inlet coefficients
        self.exit.M.calculate(self.exit.rr, len(self.inlet.M.coefficients))
        self.exit.alpha.calculate(self.exit.rr, len(self.inlet.M.coefficients))

        self.exit.v_x = self.exit.M.value * np.sqrt(self.exit.T.value) * np.cos(self.exit.alpha.value)

    def set_inlet_conditions(self, M_1, hub_tip_ratio):
        """Stores the flow information corresponding to the engine inlet conditions."""
        # store arrays of primary flow variables
        self.inlet.M = M_1 * np.ones(utils.Defaults.solver_grid)
        self.inlet.alpha = np.zeros(utils.Defaults.solver_grid)
        self.inlet.T_0 = np.ones(utils.Defaults.solver_grid)
        self.inlet.p_0 = np.ones(utils.Defaults.solver_grid)
        self.inlet.s = np.zeros(utils.Defaults.solver_grid)

        # calculate secondary flow variables
        self.inlet.T = self.inlet.T_0 * utils.stagnation_temperature_ratio(self.inlet.M)
        self.inlet.p = self.inlet.p_0 * utils.stagnation_pressure_ratio(self.inlet.M)

        # store axial velocity
        self.inlet.v_x = self.inlet.M * self.inlet.T

        # store grid of equally-spaced radial positions to consider
        self.inlet.rr = np.linspace(hub_tip_ratio, 1, utils.Defaults.solver_grid)

    def design(self, v_x_hub, hub_tip_ratio):
        """Solves for the rotor exit conditions and blade geometry."""
        # hub dimensionless axial velocity and radius are known
        self.exit.v_x[0] = v_x_hub
        self.exit.rr[0] = self.inlet.rr[0]

        # pre-compute constants used in loop
        half_gamma_minus_1 = 0.5 * (utils.gamma - 1)
        gamma_ratio = utils.gamma / (utils.gamma - 1)

        # get mean-line radius and conditions
        self.inlet.r_mean = np.sqrt(0.5 * (self.inlet.rr[0]**2 + self.inlet.rr[-1]**2))
        M_mean = np.interp(self.inlet.r_mean, self.inlet.rr, self.inlet.M)
        T_mean = np.interp(self.inlet.r_mean, self.inlet.rr, self.inlet.T)
        alpha_mean = np.interp(self.inlet.r_mean, self.inlet.rr, self.inlet.alpha)

        # get variation in inlet blade Mach number
        M_blade_mean = M_mean * np.cos(alpha_mean) / self.phi_mean
        self.inlet.M_blade = (
            M_blade_mean * (self.inlet.rr / self.inlet.r_mean)
            * np.sqrt(T_mean / self.inlet.T)
        )

        # careful - M_1_blade_mean is an array

        # get variation in relative Mach number and flow angle via vector algebra
        z_x = self.inlet.M * np.cos(self.inlet.alpha)
        z_y = self.inlet.M * np.sin(self.inlet.alpha) - self.inlet.M_blade
        self.inlet.M_rel = np.hypot(z_x, z_y)
        self.inlet.beta = np.arctan2(z_y, z_x)

        # get variation in stage loading coefficient
        psi = self.psi_mean * np.power(self.inlet.r_mean / self.inlet.rr, self.n + 1)

        # get spanwise variation of inlet relative stagnation properties
        self.inlet.T_0_rel = self.inlet.T / utils.stagnation_temperature_ratio(self.inlet.M_rel)
        self.inlet.p_0_rel = self.inlet.p / utils.stagnation_pressure_ratio(self.inlet.M_rel)

        # determine inlet mass flow rate distribution
        dm_dr_1 = (
            2 * utils.gamma / ((1 - hub_tip_ratio**2) * np.sqrt(utils.gamma - 1))
            * self.inlet.p / np.sqrt(self.inlet.T)
            * self.inlet.M * np.cos(self.inlet.alpha) * self.inlet.rr
        )
        self.inlet.m_dot = utils.cumulative_trapezoid(self.inlet.rr, dm_dr_1)

        # get mass flow rate through each streamtube
        dm_dot_1 = np.diff(self.inlet.m_dot)

        # create empty arrays of exit relative quantities
        self.exit.M_rel = np.zeros(utils.Defaults.solver_grid)
        self.exit.beta = np.zeros(utils.Defaults.solver_grid)
        self.exit.M_blade = np.zeros(utils.Defaults.solver_grid)
        self.exit.T_0_rel = np.zeros(utils.Defaults.solver_grid)
        self.exit.p_0_rel = np.zeros(utils.Defaults.solver_grid)

        # loop over all r values
        for index in range(len(self.inlet.rr)):

            # for all cases but hub streamtube
            if index > 0:
                
                # create fine grid for calculating streamtube upper bound 
                r_2_fine = np.linspace(
                    self.exit.rr[index - 1],
                    self.exit.rr[index - 1] + 5 * (self.inlet.rr[index] - self.inlet.rr[index - 1]),
                    utils.Defaults.solver_grid
                )

                # determine local exit mass flow rate distribution
                dm_dr_2 = (
                    2 * utils.gamma / ((1 - hub_tip_ratio**2) * np.sqrt(utils.gamma - 1))
                    * self.exit.p_0_rel[index - 1] / np.sqrt(self.exit.T_0_rel[index - 1])
                    * self.exit.M_rel[index - 1] * np.power(
                        1 + 0.5 * (utils.gamma - 1) * self.exit.M_rel[index - 1]**2,
                        0.5 - utils.gamma / (utils.gamma - 1)
                    ) * np.cos(self.exit.beta[index - 1]) * r_2_fine
                )
                m_dot_2 = utils.cumulative_trapezoid(r_2_fine, dm_dr_2)

                # interpolate to find upper bound of corresponding streamtube
                self.exit.rr[index] = np.interp(dm_dot_1[index - 1], m_dot_2, r_2_fine)

            # solve for relative stagnation temperature at upper bound of streamtube
            self.exit.T_0_rel[index] = (
                self.inlet.T_0_rel[index]
                + half_gamma_minus_1 * self.inlet.M_blade[index]**2 * self.inlet.T[index]
                * ((self.exit.rr[index] / self.inlet.rr[index])**2 - 1)
            )

            # get relative stagnation pressure from stagnation pressure loss coefficient
            self.exit.p_0_rel[index] = (
                self.inlet.p_0_rel[index] * (
                    np.power(
                        self.exit.T_0_rel[index] / self.inlet.T_0_rel[index], gamma_ratio
                    )
                    - self.Y_p * (1 - self.inlet.p[index] / self.inlet.p_0_rel[index])
                )
            )

            # get exit tangential velocity from specified stage loading coefficient
            self.exit.v_theta[index] = (
                self.inlet.rr[index] / self.exit.rr[index] * (
                    psi[index] * self.inlet.M_blade[index]
                    + self.inlet.M[index] * np.sin(self.inlet.alpha[index])
                ) * np.sqrt(self.inlet.T[index])
            )

            # get downstream entropy
            self.exit.s[index] = (
                self.inlet.s[index]
                + np.log(self.exit.T_0_rel[index] / self.inlet.T_0_rel[index]) / (utils.gamma - 1)
                - np.log(self.exit.p_0_rel[index] / self.inlet.p_0_rel[index]) / utils.gamma
            )

            # get downstream stagnation temperature
            self.exit.T_0[index] = (
                self.inlet.T_0[index] * (
                    1 + psi[index] * (utils.gamma - 1) * self.inlet.M_blade[index]**2
                    * self.inlet.T[index] / self.inlet.T_0[index]
                )
            )

            # for all cases but hub streamtube
            if index > 0:

                # calculate derivatives required for radial equilibrium
                dT_0 = self.exit.T_0[index] - self.exit.T_0[index - 1]
                dr_v_theta = (
                    self.exit.rr[index] * self.exit.v_theta[index]
                    - self.exit.rr[index - 1] * self.exit.v_theta[index - 1]
                )
                ds = self.exit.s[index] - self.exit.s[index - 1]

                # calculate dimensionless axial velocity via difference equation
                self.exit.v_x[index] = np.sqrt(
                    self.exit.v_x[index - 1]**2 + 2 * (
                        dT_0 / (utils.gamma - 1)
                        - self.exit.v_theta[index - 1] / self.exit.rr[index - 1] * dr_v_theta
                        - self.exit.T[index - 1] * ds
                    )
                )
                x = (
                    self.exit.v_x[index - 1]**2 + 2 * (
                        dT_0 / (utils.gamma - 1)
                        - self.exit.v_theta[index - 1] / self.exit.rr[index - 1] * dr_v_theta
                        - self.exit.T[index - 1] * ds
                    )
                )

            # extract Mach number from dimensionless velocity information
            M_2_T_T_0 = (
                (self.exit.v_x[index]**2 + self.exit.v_theta[index]**2)
                / self.exit.T_0[index]
            )
            self.exit.M[index] = np.sqrt(M_2_T_T_0 / (1 - M_2_T_T_0 * (utils.gamma - 1) / 2))

            # get flow angle from dimensionless velocity information
            self.exit.alpha[index] = np.arctan2(self.exit.v_theta[index], self.exit.v_x[index])

            # get static temperature and pressure
            self.exit.T[index] = (
                self.exit.T_0[index] * utils.stagnation_temperature_ratio(self.exit.M[index])
            )
            self.exit.p[index] = (
                self.exit.p_0[index] * utils.stagnation_pressure_ratio(self.exit.M[index])
            )

            # find blade Mach number
            self.exit.M_blade[index] = (
                self.inlet.M_blade[index] * self.exit.rr[index] / self.inlet.rr[index]
                * np.sqrt(self.inlet.T[index] / self.exit.T[index])
            )

            # get variation in relative Mach number and flow angle via vector algebra
            z_x = self.exit.M[index] * np.cos(self.exit.alpha[index])
            z_y = self.exit.M[index] * np.sin(self.exit.alpha[index]) - self.exit.M_blade[index]
            self.exit.M_rel[index] = np.hypot(z_x, z_y)
            self.exit.beta[index] = np.arctan2(z_y, z_x)

        # calculate exit stagnation conditions
        self.exit.T = self.exit.T_0_rel * utils.stagnation_temperature_ratio(self.exit.M_rel)
        self.exit.p = self.exit.p_0_rel * utils.stagnation_pressure_ratio(self.exit.M_rel)
        self.exit.p_0 = self.exit.p / utils.stagnation_pressure_ratio(self.exit.M)

        # calculate total mass flow rate
        dm_dr_2 = (
            2 * utils.gamma / ((1 - hub_tip_ratio**2) * np.sqrt(utils.gamma - 1))
            * self.exit.p / np.sqrt(self.exit.T)
            * self.exit.M * np.cos(self.exit.alpha) * self.exit.rr
        )
        self.exit.m_dot = utils.cumulative_trapezoid(self.exit.rr, dm_dr_2)

    def calculate_off_design(self, v_x_hub, hub_tip_ratio, phi):
        """Solves for the rotor exit conditions, given blade geometry."""
        # hub dimensionless axial velocity and radius are known
        self.exit.v_x[0] = v_x_hub
        self.exit.rr[0] = self.inlet.rr[0]

        # pre-compute constants used in loop
        half_gamma_minus_1 = 0.5 * (utils.gamma - 1)
        gamma_ratio = utils.gamma / (utils.gamma - 1)

        # get mean-line radius and conditions
        self.inlet.r_mean = np.sqrt(0.5 * (self.inlet.rr[0]**2 + self.inlet.rr[-1]**2))
        M_mean = np.interp(self.inlet.r_mean, self.inlet.rr, self.inlet.M)
        T_mean = np.interp(self.inlet.r_mean, self.inlet.rr, self.inlet.T)
        alpha_mean = np.interp(self.inlet.r_mean, self.inlet.rr, self.inlet.alpha)

        # get variation in inlet blade Mach number for the off-design flow coefficient
        M_blade_mean = M_mean * np.cos(alpha_mean) / phi
        self.inlet.M_blade = (
            M_blade_mean * (self.inlet.rr / self.inlet.r_mean)
            * np.sqrt(T_mean / self.inlet.T)
        )

        # get variation in relative Mach number and flow angle via vector algebra
        z_x = self.inlet.M * np.cos(self.inlet.alpha)
        z_y = self.inlet.M * np.sin(self.inlet.alpha) - self.inlet.M_blade
        self.inlet.M_rel = np.hypot(z_x, z_y)
        self.inlet.beta = np.arctan2(z_y, z_x)

        # get spanwise variation of inlet relative stagnation properties
        self.inlet.T_0_rel = self.inlet.T / utils.stagnation_temperature_ratio(self.inlet.M_rel)
        self.inlet.p_0_rel = self.inlet.p / utils.stagnation_pressure_ratio(self.inlet.M_rel)

        # get mass flow rate through each streamtube
        dm_dot_1 = np.diff(self.inlet.m_dot)

        # create empty arrays of exit relative quantities
        self.exit.M_rel = np.zeros(utils.Defaults.solver_grid)
        self.exit.beta = np.zeros(utils.Defaults.solver_grid)
        self.exit.M_blade = np.zeros(utils.Defaults.solver_grid)
        self.exit.T_0_rel = np.zeros(utils.Defaults.solver_grid)
        self.exit.p_0_rel = np.zeros(utils.Defaults.solver_grid)

        # loop over all r values
        for index in range(len(self.inlet.rr)):

            # for all cases but hub streamtube
            if index > 0:
                
                # create fine grid for calculating streamtube upper bound 
                r_2_fine = np.linspace(
                    self.exit.rr[index - 1],
                    self.exit.rr[index - 1] + 5 * (self.inlet.rr[index] - self.inlet.rr[index - 1]),
                    utils.Defaults.solver_grid
                )

                # determine local exit mass flow rate distribution
                dm_dr_2 = (
                    2 * utils.gamma / ((1 - hub_tip_ratio**2) * np.sqrt(utils.gamma - 1))
                    * self.exit.p_0_rel[index - 1] / np.sqrt(self.exit.T_0_rel[index - 1])
                    * self.exit.M_rel[index - 1] * np.power(
                        1 + 0.5 * (utils.gamma - 1) * self.exit.M_rel[index - 1]**2,
                        0.5 - utils.gamma / (utils.gamma - 1)
                    ) * np.cos(self.exit.beta[index - 1]) * r_2_fine
                )
                m_dot_2 = utils.cumulative_trapezoid(r_2_fine, dm_dr_2)

                # interpolate to find upper bound of corresponding streamtube
                self.exit.rr[index] = np.interp(dm_dot_1[index - 1], m_dot_2, r_2_fine)

            # solve for relative stagnation temperature at upper bound of streamtube
            self.exit.T_0_rel[index] = (
                self.inlet.T_0_rel[index]
                + half_gamma_minus_1 * self.inlet.M_blade[index]**2 * self.inlet.T[index]
                * ((self.exit.rr[index] / self.inlet.rr[index])**2 - 1)
            )

            # get relative stagnation pressure from stagnation pressure loss coefficient
            self.exit.p_0_rel[index] = (
                self.inlet.p_0_rel[index] * (
                    np.power(
                        self.exit.T_0_rel[index] / self.inlet.T_0_rel[index], gamma_ratio
                    )
                    - self.Y_p * (1 - self.inlet.p[index] / self.inlet.p_0_rel[index])
                )
            )

            # get downstream entropy
            self.exit.s[index] = (
                self.inlet.s[index]
                + np.log(self.exit.T_0_rel[index] / self.inlet.T_0_rel[index]) / (utils.gamma - 1)
                - np.log(self.exit.p_0_rel[index] / self.inlet.p_0_rel[index]) / utils.gamma
            )

            # for all cases but hub streamtube
            if index > 0:

                # calculate derivatives required for radial equilibrium
                dT_0_rel = self.exit.T_0_rel[index] - self.exit.T_0_rel[index - 1]
                ds = self.exit.s[index] - self.exit.s[index - 1]
                dbeta = self.exit.beta[index] - self.exit.beta[index - 1]

                # NEED TO CONSIDER EFFECTS OF DEVIATION HERE
                # calculate axial velocity at next index
                self.exit.v_x[index] = (
                    np.cos(self.exit.beta[index - 1])**2 / self.exit.v_x[index - 1] * (
                        dT_0_rel / (utils.gamma - 1)
                        - self.exit.T[index - 1] * ds
                        - (
                            self.exit.M_blade[index - 1] * np.sqrt(self.exit.T[index - 1])
                            + self.exit.v_x[index - 1] * np.tan(self.exit.beta[index - 1])
                        )**2 * (self.exit.rr[index] / self.exit.rr[index - 1] - 1)
                    )
                    + self.exit.v_x[index - 1] * (1 - dbeta * np.tan(self.exit.beta[index - 1]))
                )

                # calculate coefficients of quadratic derived from difference equation
                """a = (
                    np.tan(self.exit.beta[index])**2
                    * (1 - self.exit.rr[index - 1] / self.exit.rr[index])
                )
                b = (
                    2 * self.inlet.M_blade[index] * np.sqrt(self.inlet.T[index])
                    * self.exit.rr[index] / self.inlet.rr[index] * np.tan(self.exit.beta[index])
                    * (1 - self.exit.rr[index - 1] / self.exit.rr[index])
                    + self.exit.v_x[index - 1] / np.cos(self.exit.beta[index - 1])**2
                )
                c = (
                    self.exit.T[index - 1] * ds
                    + (
                        self.inlet.M_blade[index] * np.sqrt(self.inlet.T[index])
                        * self.exit.rr[index] / self.inlet.rr[index]
                    )**2 * (1 - self.exit.rr[index - 1] / self.exit.rr[index])
                    + self.exit.v_x[index - 1]**2 / np.cos(self.exit.beta[index - 1])**2 * (
                        np.tan(self.exit.beta[index - 1])
                        * (1 - self.exit.rr[index - 1] / self.exit.rr[index])
                        - 1
                    )
                )

                # solve quadratic
                self.exit.v_x[index] = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)"""
            
            # calculate relative tangential flow velocity
            v_theta_rel = self.exit.v_x[index] * np.tan(self.exit.beta[index])

            # get exit tangential velocity via vector addition
            self.exit.v_theta[index] = (
                v_theta_rel + self.inlet.M_blade[index] * np.sqrt(self.inlet.T[index])
                * self.exit.rr[index] / self.inlet.rr[index]
            )

            # get downstream stagnation temperature via Euler work equation
            self.exit.T_0[index] = (
                self.inlet.T_0[index]
                + self.inlet.M_blade[index] * np.sqrt(self.inlet.T[index]) / (utils.gamma - 1) * (
                    self.exit.rr[index] / self.inlet.rr[index] * self.exit.v_theta[index]
                    - self.inlet.v_theta[index]
                )
            )

            # extract Mach number from dimensionless velocity information
            M_2_T_T_0 = (
                (self.exit.v_x[index]**2 + self.exit.v_theta[index]**2)
                / self.exit.T_0[index]
            )
            self.exit.M[index] = np.sqrt(M_2_T_T_0 / (1 - M_2_T_T_0 * (utils.gamma - 1) / 2))

            # get flow angle from dimensionless velocity information
            self.exit.alpha[index] = np.arctan2(self.exit.v_theta[index], self.exit.v_x[index])

            # get static temperature and pressure
            self.exit.T[index] = (
                self.exit.T_0[index] * utils.stagnation_temperature_ratio(self.exit.M[index])
            )
            self.exit.p[index] = (
                self.exit.p_0[index] * utils.stagnation_pressure_ratio(self.exit.M[index])
            )

            # find blade Mach number
            self.exit.M_blade[index] = (
                self.inlet.M_blade[index] * self.exit.rr[index] / self.inlet.rr[index]
                * np.sqrt(self.inlet.T[index] / self.exit.T[index])
            )

            # get variation in relative Mach number and flow angle via vector algebra
            z_x = self.exit.M[index] * np.cos(self.exit.alpha[index])
            z_y = self.exit.M[index] * np.sin(self.exit.alpha[index]) - self.exit.M_blade[index]
            self.exit.M_rel[index] = np.hypot(z_x, z_y)
            self.exit.beta[index] = np.arctan2(z_y, z_x)

        utils.debug(f"self.exit.T: {self.exit.T}")

        # calculate exit temperature and pressure
        self.exit.T = self.exit.T_0_rel * utils.stagnation_temperature_ratio(self.exit.M_rel)
        self.exit.p = self.exit.p_0_rel * utils.stagnation_pressure_ratio(self.exit.M_rel)
        self.exit.p_0 = self.exit.p / utils.stagnation_pressure_ratio(self.exit.M)

        utils.debug(f"self.exit.T: {self.exit.T}")

        # calculate total mass flow rate
        dm_dr_2 = (
            2 * utils.gamma / ((1 - hub_tip_ratio**2) * np.sqrt(utils.gamma - 1))
            * self.exit.p / np.sqrt(self.exit.T)
            * self.exit.M * np.cos(self.exit.alpha) * self.exit.rr
        )
        self.exit.m_dot = utils.cumulative_trapezoid(self.exit.rr, dm_dr_2)

        print(f"self.inlet.m_dot: {self.inlet.m_dot}")
        print(f"self.exit.m_dot: {self.exit.m_dot}")

    def evaluate(self):
        """Evaluates performance of the rotor blade row."""
        # find inlet relative dimensionless velocity
        self.inlet.v_theta_rel = self.inlet.v_x * np.tan(self.inlet.beta)
        self.inlet.U = self.inlet.v_theta - self.inlet.v_theta_rel

        # find inlet relative dimensionless velocity
        self.exit.v_theta_rel = self.exit.v_x * np.tan(self.exit.beta)
        self.exit.U = self.exit.v_theta - self.exit.v_theta_rel

        # find flow coefficient
        self.exit.phi = self.inlet.v_x / self.inlet.U

        # find stage loading coefficient
        self.exit.psi = (
            (self.exit.T_0 - self.inlet.T_0)
            / (self.inlet.T * (utils.gamma - 1) * self.inlet.M_blade**2)
        )

    def calculate_chord(self, aspect_ratio, diffusion_factor):
        """Applies empirical relations to design the pitch-to-chord distributions."""
        # get nominal pitch-to-chord distribution
        self.exit.pitch_to_chord = (
            2 * (
                diffusion_factor - 1 + self.exit.M_rel / self.inlet.M_rel
                * np.sqrt(self.exit.T / self.inlet.T)
            ) / (
                np.sin(np.abs(self.inlet.beta))
                - self.exit.M_rel / self.inlet.M_rel
                * np.sqrt(self.exit.T / self.inlet.T)
                * np.sin(np.abs(self.exit.beta))
            )
        )

        # calculate minimum number of blades to achieve aspect ratio
        self.no_of_blades = 2
        while True:

            # calculate pitch and chord distributions
            self.exit.pitch = 2 * np.pi * self.exit.rr / self.no_of_blades
            self.exit.chord = self.exit.pitch / self.exit.pitch_to_chord

            # calculate mean-line aspect ratio
            r_mean = 0.5 * (self.exit.rr[0] + self.exit.rr[-1])
            #pitch_mean = np.interp(r_mean, self.exit.rr, self.exit.pitch)
            chord_mean = np.interp(r_mean, self.exit.rr, self.exit.chord)
            AR_mean = (self.exit.rr[-1] - self.exit.rr[0]) / chord_mean

            # check if aspect ratio criterion is met
            if AR_mean > aspect_ratio or self.no_of_blades > utils.Defaults.max_blades:

                break

            # increment number of blades
            self.no_of_blades += 1

        # calculate true aspect ratio distribution
        self.exit.aspect_ratio = (self.exit.rr[-1] - self.exit.rr[0]) / self.exit.chord

    def calculate_deviation(self, deviation_constant):
        """Calculates the deviation distribution using Carter and Howell."""
        # store inlet metal angles
        self.inlet.metal_angle = self.inlet.beta
        
        # store inlet and exit angles in degrees for convenience
        inlet_angles = utils.rad_to_deg(self.inlet.beta)
        exit_angles = utils.rad_to_deg(self.exit.beta)

        # calculate deviation coefficient using Howell's correlation for a circular camber line
        m = 0.23 + exit_angles / 500

        # calculate exit metal angles and corresponding deviation
        self.exit.metal_angle = (
            utils.deg_to_rad(
                exit_angles - m * inlet_angles * np.sqrt(self.exit.pitch_to_chord)
                / (1 + m * np.sqrt(self.exit.pitch_to_chord))
            )
        )
        self.exit.deviation = self.exit.beta - self.exit.metal_angle

        # calculate axial chord distribution
        self.exit.axial_chord = (
            self.exit.chord * (np.sin(self.exit.metal_angle) - np.sin(self.inlet.metal_angle))
            / (self.exit.metal_angle - self.inlet.metal_angle)
        )
