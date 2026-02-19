# import modules
import numpy as np
from scipy.optimize import least_squares
#from scipy.integrate import cumulative_simpson
from time import perf_counter as timer

# import custom classes
from annulus import Annulus
from coefficients import Coefficients
import utils
from blade_row import Blade_row

# define Stator class
class Stator(Blade_row):
    """
    Represents a single compressor stator and stores the associated flowfield.
    
    Used to investigate the flow across a stator. Every instance of the class will
    contain an inlet and exit flow object where all of the flow properties are stored.
    
    Parameters
    ----------
    Y_p : float
        Stagnation pressure loss coefficient.
    """
    def __init__(self, Y_p):
        """Create instance of the Blade_row class."""
        #
        super().__init__(Y_p)

        # store input variables
        self.Y_p = Y_p
        
        # hub radius is set by global hub-tip ratio
        self.r_hub = utils.Defaults.hub_tip_ratio

        # assign the default colour of black
        self.colour = 'k'

        # create empty inlet and exit Annulus instances
        self.inlet = Annulus()
        self.exit = Annulus()

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

    def old_design(self):
        """Determines the stator blade geometry necessary to satisfy the given stage parameters."""
        # store variation in static properties based on polynomial fits
        self.inlet.T.value = self.inlet.T_0.value * utils.stagnation_temperature_ratio(self.inlet.M.value)
        self.inlet.p.value = self.inlet.p_0.value * utils.stagnation_pressure_ratio(self.inlet.M.value)

        # get cumulative inlet mass flow
        dm_dr_1 = (
            self.inlet.p.value / np.sqrt(self.inlet.T.value) * self.inlet.M.value * np.cos(self.inlet.alpha.value)
            * self.inlet.rr
        )
        m_dot_1 = utils.cumulative_trapezoid(dm_dr_1, x = self.inlet.rr, initial = 0.0)

        # get incremental change in inlet mass flow
        dm_dot_1 = np.diff(m_dot_1)

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
                    utils.Defaults.solver_grid
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
                    * utils.cumulative_trapezoid(dm_dr_2, x = r_2_fine, initial = 0.0)
                )

                # interpolate to find upper bound of corresponding streamtube
                self.exit.rr[index + 1] = np.interp(m_1, m_dot_2, r_2_fine)

            # expand primary flow variables onto new grid
            self.exit.value("M")
            if (np.abs(self.exit.M.value) > 1).any():

                print(f"Error!!!\n{self.exit.M.value}")
                solutions = 1e9 * np.random.random() * np.ones_like(solutions)
                return solutions

            # get variation in exit static properties
            self.exit.T.value = (
                self.exit.T_0.value * utils.stagnation_temperature_ratio(self.exit.M.value)
            )
            self.exit.p.value = (
                self.exit.p_0.value * utils.stagnation_pressure_ratio(self.exit.M.value)
            )

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

            # final residual comes from constant area
            solutions[-1] = self.exit.rr[-1]**2 - self.inlet.rr[-1]**2

            # return solutions
            #solutions = solutions.ravel()
            return solutions
        
        # set list of lower and upper bounds and reshape
        lower = 100 * -2 * np.ones_like(self.inlet.M.coefficients)
        upper = 100 * 2 * np.ones_like(self.inlet.M.coefficients)

        # initialise exit annulus object to be populated
        self.exit = Annulus()
        self.exit.alpha.coefficients = np.zeros_like(self.inlet.M.coefficients)
        self.exit.alpha.value = np.zeros_like(self.inlet.M.value)

        # set initial guess based on inlet conditions
        x0 = self.inlet.M.coefficients

        # find stagnation quantities via no isentropic stagnation temperature change
        self.exit.T_0.value = self.inlet.T_0.value
        self.exit.p_0.value = (
            self.inlet.p_0.value * (1 - self.Y_p * (1 - self.inlet.p.value / self.inlet.p_0.value))
        )

        # solve iteratively
        sol = least_squares(
            solve_stator, x0, bounds = (lower, upper),
            xtol = 1e-6, ftol = 1e-6, gtol = 1e-6,
            x_scale = 'jac'
        )
        utils.debug(f"sol: {sol}")

    def design(self, v_x_hub, hub_tip_ratio):
        """Solves for the stator exit conditions and blade geometry."""
        # hub dimensionless axial velocity and radius are known
        self.exit.v_x[0] = v_x_hub
        self.exit.rr[0] = self.inlet.rr[0]

        # stagnation temperature is conserved across stator row
        self.exit.T_0 = self.inlet.T_0
        
        # find exit stagnation pressure via stagnation pressure loss coefficient
        self.exit.p_0 = (
            self.inlet.p_0 * (1 - self.Y_p * (1 - utils.stagnation_pressure_ratio(self.inlet.M)))
        )

        # find corresponding change in dimensionless entropy
        self.exit.s = self.inlet.s - np.log(self.exit.p_0 / self.inlet.p_0) / utils.gamma
        
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
        dm_dot_2 = np.diff(self.inlet.m_dot)

        # loop over each streamtube
        for index in range(utils.Defaults.solver_grid):
                
            # for all cases except hub streamline
            if index > 0:
                
                # create fine grid for calculating streamtube upper bound 
                r_3_fine = np.linspace(
                    self.exit.rr[index - 1],
                    self.exit.rr[index - 1] + 5 * (self.inlet.rr[index] - self.inlet.rr[index - 1]),
                    utils.Defaults.solver_grid
                )

                # determine local exit mass flow rate distribution
                dm_dr_3 = (
                    2 * utils.gamma / ((1 - hub_tip_ratio**2) * np.sqrt(utils.gamma - 1))
                    * self.exit.p_0[index - 1] / np.sqrt(self.exit.T_0[index - 1])
                    * self.exit.M[index - 1] * np.power(
                        1 + 0.5 * (utils.gamma - 1) * self.exit.M[index - 1]**2,
                        0.5 - utils.gamma / (utils.gamma - 1)
                    ) * np.cos(self.exit.alpha[index - 1]) * r_3_fine
                )
                m_dot_3 = utils.cumulative_trapezoid(r_3_fine, dm_dr_3)

                # interpolate to find upper bound of corresponding streamtube
                self.exit.rr[index] = np.interp(dm_dot_2[index - 1], m_dot_3, r_3_fine)

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

            # solve for exit tangential velocity from axial velocity and flow angle
            self.exit.v_theta[index] = self.exit.v_x[index] * np.tan(self.exit.alpha[index])

            # solve for Mach number
            v_squared = (self.exit.v_x[index] / np.cos(self.exit.alpha[index]))**2
            self.exit.M[index] = np.sqrt(v_squared / (1 - 0.5 * (utils.gamma - 1) * v_squared))

        # solve for exit static conditions
        self.exit.T = self.exit.T_0 * utils.stagnation_temperature_ratio(self.exit.M)
        self.exit.p = self.exit.p_0 * utils.stagnation_temperature_ratio(self.exit.M)

        # calculate exit mass flow rate
        self.exit.dm_dot_dr = (
            2 * utils.gamma / ((1 - hub_tip_ratio**2) * np.sqrt(utils.gamma - 1))
            * self.exit.p / np.sqrt(self.exit.T)
            * self.exit.M * np.cos(self.exit.alpha) * self.exit.rr
        )
        self.exit.m_dot = utils.cumulative_trapezoid(self.exit.rr, self.exit.dm_dot_dr)

    def evaluate(self, T_1):
        """Evaluates performance of the stator blade row."""
        # solve for stage reaction
        self.exit.reaction = (
            (self.inlet.T - T_1)
            / (self.exit.T - T_1)
        )

    def calculate_chord(self, aspect_ratio, diffusion_factor):
        """Applies empirical relations to design the pitch-to-chord distributions."""
        # get nominal pitch-to-chord distribution
        self.exit.pitch_to_chord = (
            2 * (
                diffusion_factor - 1 + self.exit.M / self.inlet.M
                * np.sqrt(self.exit.T / self.inlet.T)
            ) / (
                np.sin(np.abs(self.inlet.alpha))
                - self.exit.M / self.inlet.M
                * np.sqrt(self.exit.T / self.inlet.T)
                * np.sin(np.abs(self.exit.alpha))
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
        self.inlet.metal_angle = self.inlet.alpha

        # store inlet and exit angles in degrees for convenience
        inlet_angles = utils.rad_to_deg(self.inlet.alpha)
        exit_angles = utils.rad_to_deg(self.exit.alpha)

        # calculate deviation coefficient using Howell's correlation for a circular camber line
        m = 0.23 + exit_angles / 500

        # calculate exit metal angles and corresponding deviation
        self.exit.metal_angle = (
            utils.deg_to_rad(
                exit_angles - m * inlet_angles * np.sqrt(self.exit.pitch_to_chord)
                / (1 + m * np.sqrt(self.exit.pitch_to_chord))
            )
        )
        self.exit.deviation = self.exit.alpha - self.exit.metal_angle

        # calculate axial chord distribution
        self.exit.axial_chord = (
            self.exit.chord * (np.sin(self.exit.metal_angle) - np.sin(self.inlet.metal_angle))
            / (self.exit.metal_angle - self.inlet.metal_angle)
        )
