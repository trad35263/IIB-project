# import modules
import numpy as np
from scipy.optimize import least_squares
from time import perf_counter as timer
import matplotlib.pyplot as plt
import copy

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
        # initialise parent Blade_row class
        super().__init__()

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

        # use quadratic variation of radial positions to consider
        self.inlet.rr = (
            hub_tip_ratio
            + (1 - hub_tip_ratio) * (np.linspace(0, 1, utils.Defaults.solver_grid))**2
        )

    def design(self, v_x_hub, hub_tip_ratio):
        """Determines the flowfield through the rotor and solves for the flow angles."""
        # start timer
        t1 = timer()

        # impose bounds on hub velocity guess
        v_x_hub = utils.bound(v_x_hub)

        # create empty arrays of exit relative quantities
        self.exit.M_rel = np.zeros(utils.Defaults.solver_grid)
        self.exit.T_0_rel = np.zeros(utils.Defaults.solver_grid)
        self.exit.p_0_rel = np.zeros(utils.Defaults.solver_grid)

        # determine inlet mass flow rate distribution
        self.inlet.dm_dot_dr = (
            2 * utils.gamma / ((1 - hub_tip_ratio**2) * np.sqrt(utils.gamma - 1))
            * self.inlet.p / np.sqrt(self.inlet.T)
            * self.inlet.M * np.cos(self.inlet.alpha) * self.inlet.rr
        )
        self.inlet.m_dot = utils.cumulative_trapezoid(self.inlet.rr, self.inlet.dm_dot_dr)

        # get mass flow rate through each streamtube
        dm_dot = np.diff(self.inlet.m_dot)

        # hub dimensionless axial velocity and radius are known
        self.exit.v_x[0] = v_x_hub
        self.exit.rr[0] = self.inlet.rr[0]

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

        # loop over each streamtube
        for index in range(len(self.inlet.M)):

            if np.isnan(self.exit.rr[index]):

                input()

            # find relative stagnation temperature at upper bound of streamtube
            self.exit.T_0_rel[index] = (
                self.inlet.T_0_rel[index]
                + 0.5 * (utils.gamma - 1) * self.inlet.M_blade[index]**2 * self.inlet.T[index]
                * ((self.exit.rr[index] / self.inlet.rr[index])**2 - 1)
            )

            # find relative stagnation pressure from stagnation pressure loss coefficient
            self.exit.p_0_rel[index] = (
                self.inlet.p_0_rel[index] * (
                    np.power(
                        self.exit.T_0_rel[index] / self.inlet.T_0_rel[index],
                        utils.gamma / (utils.gamma - 1)
                    )
                    - self.Y_p * (1 - self.inlet.p[index] / self.inlet.p_0_rel[index])
                )
            )

            # find exit tangential velocity from specified stage loading coefficient
            self.exit.v_theta[index] = (
                self.inlet.rr[index] / self.exit.rr[index] * (
                    psi[index] * self.inlet.M_blade[index]
                    + self.inlet.M[index] * np.sin(self.inlet.alpha[index])
                ) * np.sqrt(self.inlet.T[index])
            )

            # find entropy
            self.exit.s[index] = (
                self.inlet.s[index]
                + np.log(self.exit.T_0_rel[index] / self.inlet.T_0_rel[index]) / (utils.gamma - 1)
                - np.log(self.exit.p_0_rel[index] / self.inlet.p_0_rel[index]) / utils.gamma
            )

            # find stagnation temperature from specified stage loading coefficient
            self.exit.T_0[index] = (
                self.inlet.T_0[index] * (
                    1 + psi[index] * (utils.gamma - 1) * self.inlet.M_blade[index]**2
                    * self.inlet.T[index] / self.inlet.T_0[index]
                )
            )

            # find Mach number
            self.exit.M[index] = (
                np.sqrt(
                    (self.exit.v_x[index]**2 + self.exit.v_theta[index]**2) / (
                        self.exit.T_0[index] - 0.5 * (utils.gamma - 1)
                        * (self.exit.v_x[index]**2 + self.exit.v_theta[index]**2)
                    )
                )
            )

            # find static temperature
            self.exit.T[index] = (
                self.exit.T_0[index] * utils.stagnation_temperature_ratio(self.exit.M[index])
            )

            # find relative Mach number
            self.exit.M_rel[index] = (
                utils.inverse_temperature_ratio(self.exit.T[index] / self.exit.T_0_rel[index])
            )

            # find static pressure
            self.exit.p[index] = (
                self.exit.p_0_rel[index] * utils.stagnation_pressure_ratio(self.exit.M_rel[index])
            )

            # for all but final streamline
            if index < len(self.inlet.M) - 1:

                # predictor step - calculate streamtube outer radius
                self.exit.rr[index + 1] = (
                    np.sqrt(
                        self.exit.rr[index]**2
                        + dm_dot[index] * np.sqrt(utils.gamma - 1) * (1 - hub_tip_ratio**2)
                        * self.exit.T[index]
                        / (utils.gamma * self.exit.p[index] * self.exit.v_x[index])
                    )
                )

                # predictor step - find relative stagnation temperature
                self.exit.T_0_rel[index + 1] = (
                    self.inlet.T_0_rel[index + 1]
                    + 0.5 * (utils.gamma - 1) * self.inlet.M_blade[index + 1]**2 * self.inlet.T[index + 1]
                    * ((self.exit.rr[index + 1] / self.inlet.rr[index + 1])**2 - 1)
                )

                # predictor step - find relative stagnation pressure
                self.exit.p_0_rel[index + 1] = (
                    self.inlet.p_0_rel[index + 1] * (
                        np.power(
                            self.exit.T_0_rel[index + 1] / self.inlet.T_0_rel[index + 1],
                            utils.gamma / (utils.gamma - 1)
                        )
                        - self.Y_p * (1 - self.inlet.p[index + 1] / self.inlet.p_0_rel[index + 1])
                    )
                )

                # predictor step - find tangential velocity
                self.exit.v_theta[index + 1] = (
                    self.inlet.rr[index + 1] / self.exit.rr[index + 1] * (
                        psi[index + 1] * self.inlet.M_blade[index + 1]
                        + self.inlet.M[index + 1] * np.sin(self.inlet.alpha[index + 1])
                    ) * np.sqrt(self.inlet.T[index + 1])
                )

                # predictor step - find entropy
                self.exit.s[index + 1] = (
                    self.inlet.s[index + 1]
                    + np.log(self.exit.T_0_rel[index + 1] / self.inlet.T_0_rel[index + 1])
                    / (utils.gamma - 1)
                    - np.log(self.exit.p_0_rel[index + 1] / self.inlet.p_0_rel[index + 1])
                    / utils.gamma
                )

                # predictor step - find stagnation temperature
                self.exit.T_0[index + 1] = (
                    self.inlet.T_0[index + 1] * (
                        1 + psi[index + 1] * (utils.gamma - 1) * self.inlet.M_blade[index + 1]**2
                        * self.inlet.T[index + 1] / self.inlet.T_0[index + 1]
                    )
                )

                # predictor step - calculate axial velocity at station index + 1
                self.exit.v_x[index + 1] = (
                    self.exit.v_x[index] + 1 / self.exit.v_x[index] * (
                        1 / (utils.gamma - 1) * (self.exit.T_0[index + 1] - self.exit.T_0[index])
                        - self.exit.T[index] * (self.exit.s[index + 1] - self.exit.s[index])
                        - self.exit.v_theta[index] / self.exit.rr[index] * (
                            self.exit.rr[index + 1] * self.exit.v_theta[index + 1]
                            - self.exit.rr[index] * self.exit.v_theta[index]
                        )
                    )
                )

                # predictor step - calculate Mach number
                self.exit.M[index + 1] = (
                    np.sqrt(
                        (self.exit.v_x[index + 1]**2 + self.exit.v_theta[index + 1]**2) / (
                            self.exit.T_0[index + 1] - 0.5 * (utils.gamma - 1)
                            * (self.exit.v_x[index + 1]**2 + self.exit.v_theta[index + 1]**2)
                        )
                    )
                )

                # find static temperature
                self.exit.T[index + 1] = (
                    self.exit.T_0[index + 1]
                    * utils.stagnation_temperature_ratio(self.exit.M[index + 1])
                )

                # find relative Mach number
                self.exit.M_rel[index + 1] = utils.inverse_temperature_ratio(
                    self.exit.T[index + 1] / self.exit.T_0_rel[index + 1]
                )

                # find static pressure
                self.exit.p[index + 1] = (
                    self.exit.p_0_rel[index + 1]
                    * utils.stagnation_pressure_ratio(self.exit.M_rel[index + 1])
                )

                # corrector step - recalculate streamtube outer radius
                self.exit.rr[index + 1] = (
                    np.sqrt(
                        self.exit.rr[index]**2
                        + 2 * dm_dot[index] * np.sqrt(utils.gamma - 1) * (1 - hub_tip_ratio**2) / (
                            utils.gamma * (
                                self.exit.p[index] * self.exit.v_x[index] / self.exit.T[index]
                                + self.exit.p[index + 1] * self.exit.v_x[index + 1]
                                / self.exit.T[index + 1]
                            )
                        )
                    )
                )

                # corrector step - find relative stagnation temperature
                self.exit.T_0_rel[index + 1] = (
                    self.inlet.T_0_rel[index + 1]
                    + 0.5 * (utils.gamma - 1) * self.inlet.M_blade[index + 1]**2 * self.inlet.T[index + 1]
                    * ((self.exit.rr[index + 1] / self.inlet.rr[index + 1])**2 - 1)
                )

                # corrector step - find relative stagnation pressure
                self.exit.p_0_rel[index + 1] = (
                    self.inlet.p_0_rel[index + 1] * (
                        np.power(
                            self.exit.T_0_rel[index + 1] / self.inlet.T_0_rel[index + 1],
                            utils.gamma / (utils.gamma - 1)
                        )
                        - self.Y_p * (1 - self.inlet.p[index + 1] / self.inlet.p_0_rel[index + 1])
                    )
                )

                # corrector step - find tangential velocity
                self.exit.v_theta[index + 1] = (
                    self.inlet.rr[index + 1] / self.exit.rr[index + 1] * (
                        psi[index + 1] * self.inlet.M_blade[index + 1]
                        + self.inlet.M[index + 1] * np.sin(self.inlet.alpha[index + 1])
                    ) * np.sqrt(self.inlet.T[index + 1])
                )

                # corrector step - find entropy
                self.exit.s[index + 1] = (
                    self.inlet.s[index + 1]
                    + np.log(self.exit.T_0_rel[index + 1] / self.inlet.T_0_rel[index + 1]) / (utils.gamma - 1)
                    - np.log(self.exit.p_0_rel[index + 1] / self.inlet.p_0_rel[index + 1]) / utils.gamma
                )

                # corrector step - find stagnation temperature
                self.exit.T_0[index + 1] = (
                    self.inlet.T_0[index + 1] * (
                        1 + psi[index + 1] * (utils.gamma - 1) * self.inlet.M_blade[index + 1]**2
                        * self.inlet.T[index + 1] / self.inlet.T_0[index + 1]
                    )
                )

                # corrector step - recalculate axial velocity at station index + 1
                self.exit.v_x[index + 1] = (
                    self.exit.v_x[index] + 1 / (self.exit.v_x[index] + self.exit.v_x[index + 1]) * (
                        2 / (utils.gamma - 1) * (self.exit.T_0[index + 1] - self.exit.T_0[index])
                        - (self.exit.T_0[index] + self.exit.T_0[index + 1])
                        * (self.exit.s[index + 1] - self.exit.s[index])
                        - (self.exit.v_theta[index] + self.exit.v_theta[index + 1])
                        / (self.exit.rr[index] + self.exit.rr[index + 1]) * (
                            self.exit.rr[index + 1] * self.exit.v_theta[index + 1]
                            - self.exit.rr[index] * self.exit.v_theta[index]
                        )
                    )
                )

        # calculate flow angle from dimensionless velocity information
        self.exit.alpha = np.arctan2(self.exit.v_theta, self.exit.v_x)

        # calculate stagnation pressure
        self.exit.p_0 = self.exit.p / utils.stagnation_pressure_ratio(self.exit.M)

        # find blade Mach number
        self.exit.M_blade = (
            self.inlet.M_blade * self.exit.rr / self.inlet.rr
            * np.sqrt(self.inlet.T / self.exit.T)
        )

        # get variation in relative Mach number and flow angle via vector algebra
        z_x = self.exit.M * np.cos(self.exit.alpha)
        z_y = self.exit.M * np.sin(self.exit.alpha) - self.exit.M_blade
        self.exit.M_rel = np.hypot(z_x, z_y)
        self.exit.beta = np.arctan2(z_y, z_x)

        # calculate geometric mean-line radius
        self.exit.r_mean = np.sqrt(0.5 * (self.exit.rr[0]**2 + self.exit.rr[-1]**2))

        # calculate exit mass flow rate
        self.exit.dm_dot_dr = (
            2 * utils.gamma / ((1 - hub_tip_ratio**2) * np.sqrt(utils.gamma - 1))
            * self.exit.p / np.sqrt(self.exit.T)
            * self.exit.M * np.cos(self.exit.alpha) * self.exit.rr
        )
        self.exit.m_dot = utils.cumulative_trapezoid(self.exit.rr, self.exit.dm_dot_dr)

        utils.debug(f"Rotor: {100 * (self.exit.m_dot[-1] / self.inlet.m_dot[-1] - 1)}")

        # end timer
        t2 = timer()
        utils.debug(
            f"Rotor design completed in {utils.Colours.GREEN}{t2 - t1:.4g}{utils.Colours.END} s!"
        )

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
        """self.exit.M_rel = np.zeros(utils.Defaults.solver_grid)
        self.exit.beta = np.zeros(utils.Defaults.solver_grid)
        self.exit.M_blade = np.zeros(utils.Defaults.solver_grid)
        self.exit.T_0_rel = np.zeros(utils.Defaults.solver_grid)
        self.exit.p_0_rel = np.zeros(utils.Defaults.solver_grid)"""

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
                + 0.5 * (utils.gamma - 1) * self.inlet.M_blade[index]**2 * self.inlet.T[index]
                * ((self.exit.rr[index] / self.inlet.rr[index])**2 - 1)
            )

            # get relative stagnation pressure from stagnation pressure loss coefficient
            self.exit.p_0_rel[index] = (
                self.inlet.p_0_rel[index] * (
                    np.power(
                        self.exit.T_0_rel[index] / self.inlet.T_0_rel[index],
                        utils.gamma / (utils.gamma - 1)
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
                + self.inlet.M_blade[index] * np.sqrt(self.inlet.T[index]) * (utils.gamma - 1) * (
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

    def calculate_chord(self, aspect_ratio, diffusion_factor, design_parameter):
        """Applies empirical relations to design the pitch-to-chord distributions."""
        # calculate mean-line chord from aspect ratio
        chord_mean = (self.exit.rr[-1] - self.exit.rr[0]) / aspect_ratio

        # calculate parabolic chord distribution
        self.exit.chord = (
            (1 - design_parameter) * chord_mean * (self.exit.rr - self.exit.rr[0])**2 / (
                (self.exit.r_mean - self.exit.rr[0])**2
                - 2 * (self.exit.rr[-1] - self.exit.rr[0]) * (self.exit.r_mean - self.exit.rr[0])
            )
            - 2 * (1 - design_parameter) * chord_mean * (self.exit.rr[-1] - self.exit.rr[0])
            * (self.exit.rr - self.exit.rr[0]) / (
                (self.exit.r_mean - self.exit.rr[0])**2
                - 2 * (self.exit.rr[-1] - self.exit.rr[0]) * (self.exit.r_mean - self.exit.rr[0])
            )
            + design_parameter * chord_mean
        )

        # calculate minimum number of blades to achieve aspect ratio
        self.no_of_blades = utils.Defaults.min_no_of_blades
        while True:

            # calculate pitch and pitch-to-chord distributions
            self.exit.pitch = 2 * np.pi * self.exit.rr / self.no_of_blades
            self.exit.pitch_to_chord = self.exit.pitch / self.exit.chord

            # calculate diffusion factor distribution
            self.exit.diffusion_factor = (
                1 - self.exit.M_rel / self.inlet.M_rel * np.sqrt(self.exit.T / self.inlet.T)
                + 0.5 * self.exit.pitch_to_chord * np.abs(
                    np.sin(self.inlet.beta)
                    - self.exit.M_rel / self.inlet.M_rel
                    * np.sqrt(self.exit.T / self.inlet.T)
                    * np.sin(self.exit.beta)
                )
            )

            # check if diffusion factor criterion is met
            if (
                np.max(self.exit.diffusion_factor) <= diffusion_factor
                or self.no_of_blades >= utils.Defaults.max_no_of_blades
            ):

                # exit while-loop
                break

            # increment number of blades
            self.no_of_blades += 1

        # calculate pitch-to-chord distribution for constant diffusion factor
        pitch_to_chord_DF = (
            2 * (
                diffusion_factor - 1 + self.exit.M_rel / self.inlet.M_rel
                * np.sqrt(self.exit.T / self.inlet.T)
            ) / np.abs(
                np.sin(self.inlet.beta)
                - self.exit.M_rel / self.inlet.M_rel
                * np.sqrt(self.exit.T / self.inlet.T)
                * np.sin(self.exit.beta)
            )
        )

        # calculate corresponding deviation
        self.calculate_deviation()

        # get mean-line deviation value
        """delta_mean = np.interp(self.exit.r_mean, self.exit.rr, self.exit.deviation)"""

        # calculate pitch-to-chord distribution for constant deviation
        """pitch_to_chord_deviation = (
            (
                utils.rad_to_deg(delta_mean) / (
                    (0.23 + np.abs(utils.rad_to_deg(self.exit.beta)) / 500)
                    * utils.rad_to_deg(self.inlet.beta - delta_mean - self.exit.beta)
                )
            )**2
        )"""

        # plot all 3 distributions
        """span = (self.exit.rr - self.exit.rr[0]) / (self.exit.rr[-1] - self.exit.rr[0])
        fig, ax = plt.subplots(figsize = (10, 5))
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 1)
        ax.plot(self.exit.pitch_to_chord, span, label = "Constant chord", color = 'C0')
        ax.plot(pitch_to_chord_DF, span, label = "Constant diffusion factor", color = 'C1')
        ax.fill_betweenx(span, pitch_to_chord_DF, ax.get_xlim()[1], alpha = 0.2, color = 'C1', label = "DF too high")
        ax.plot(pitch_to_chord_deviation, span, label = "Constant deviation", color = 'C2')
        ax.grid()
        ax.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        ax.set_title(
            f"Stage loading coefficient: {self.psi_mean}\n"
            f"No. of blades: {self.no_of_blades}\n"
            f"Mean-line DF: {diffusion_factor}\n"
            f"Mean-line AR: {aspect_ratio}"
        )
        ax.set_xlabel("Pitch-to-chord ratio")
        ax.set_ylabel("Dimensionless span")
        plt.tight_layout()
        plt.show()"""

        # calculate minimum diffusion factor
        """DF = (
            1 - self.exit.M_rel / self.inlet.M_rel * np.sqrt(self.exit.T / self.inlet.T)
            + 0.5 * utils.Defaults.min_pitch_to_chord_ratio * np.abs(
                np.sin(self.inlet.beta)
                - self.exit.M_rel / self.inlet.M_rel * np.sqrt(self.exit.T / self.inlet.T)
                * np.sin(self.exit.beta)
            )
        )

        # convert to chord distribution
        chord_DF = self.exit.pitch / pitch_to_chord_DF"""

        # calculate aspect ratio distribution
        self.exit.aspect_ratio = (self.exit.rr[-1] - self.exit.rr[0]) / self.exit.chord

        # calculate axial chord distribution
        self.exit.axial_chord = (
            self.exit.chord * (np.sin(self.exit.metal_angle) - np.sin(self.inlet.metal_angle))
            / (self.exit.metal_angle - self.inlet.metal_angle)
        )

    def calculate_deviation(self):
        """Calculates the deviation distribution using Carter and Howell."""
        # store inlet metal angle for zero nominal incidence
        self.inlet.metal_angle = self.inlet.beta
        
        # store inlet and exit angles in degrees for convenience
        inlet_angle = utils.rad_to_deg(self.inlet.beta)
        exit_angle = utils.rad_to_deg(self.exit.beta)

        # calculate deviation coefficient using Howell's correlation for a circular camber line
        m = 0.23 + np.abs(exit_angle) / 500

        # calculate two options of metal angle
        metal_angle_1 = (
            (m * np.sqrt(self.exit.pitch_to_chord) * inlet_angle + exit_angle)
            / (m * np.sqrt(self.exit.pitch_to_chord) + 1)
        )
        metal_angle_2 = (
            (-m * np.sqrt(self.exit.pitch_to_chord) * inlet_angle + exit_angle)
            / (-m * np.sqrt(self.exit.pitch_to_chord) + 1)
        )

        # calculate exit metal angles and corresponding deviation
        self.exit.metal_angle = utils.deg_to_rad(metal_angle_2)
        self.exit.deviation = self.exit.metal_angle - self.exit.beta
