# import modules
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import cumulative_simpson
from time import perf_counter as timer

# import custom classes
from annulus import Annulus
from coefficients import Coefficients
import utils

# define Stator class
class Stator:
    """
    Represents a single compressor stator and stores the associated flowfield.
    
    Used to investigate the flow across a stator. Every instance of the class will
    contain an inlet and exit flow object where all of the flow properties are stored.
    
    Parameters
    ----------
    Y_p : float
        Stagnation pressure loss coefficient.
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

    def design(self):
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
            solutions = solutions.ravel()
            return solutions
        
        # set list of lower and upper bounds and reshape
        lower = 100 * -2 * np.ones_like(self.inlet.M.coefficients)
        upper = 100 * 2 * np.ones_like(self.inlet.M.coefficients)

        # get initial guess based on inlet conditions
        x0 = self.inlet.M.coefficients
        #print(f"x0: {x0}")

        # solve iteratively
        sol = least_squares(solve_stator, x0, bounds = (lower, upper), max_nfev = utils.Defaults.nfev)
        #print(f"sol: {sol}")
