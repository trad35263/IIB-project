# import modules
import numpy as np
from scipy.optimize import least_squares
from time import perf_counter as timer

# import custom classes
from annulus import Annulus
from coefficients import Coefficients
import utils

# define Rotor class
class Rotor:
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

    def design(self, phi, psi, n):
        """Determines the rotor blade geometry necessary to satisfy the given stage parameters."""        
        # initialise empty Coefficients instances for relative quantities
        self.inlet.M_rel = Coefficients()
        self.inlet.beta = Coefficients()
        self.inlet.T_0_rel = Coefficients()
        self.inlet.p_0_rel = Coefficients()

        # calculate mid-span radius
        self.inlet.r_mean = 0.5 * (self.inlet.rr[0] + self.inlet.rr[-1])

        # get variation in blade Mach number
        M_1_blade_mean = self.inlet.M.value * np.cos(self.inlet.alpha.value) / phi
        T_mean = np.interp(self.inlet.r_mean, self.inlet.rr, self.inlet.T.value)
        M_1_blade = M_1_blade_mean * (self.inlet.rr / self.inlet.r_mean) * np.sqrt(T_mean / self.inlet.T.value)

        # get variation in relative Mach number and flow angle via vector algebra
        z_x = self.inlet.M.value * np.cos(self.inlet.alpha.value)
        z_y = self.inlet.M.value * np.sin(self.inlet.alpha.value) - M_1_blade
        self.inlet.M_rel.value = np.hypot(z_x, z_y)
        self.inlet.beta.value = np.arctan2(z_y, z_x)

        # store corresponding coefficients for M_rel_1 and beta_1
        self.inlet.M_rel.calculate(self.inlet.rr, len(self.inlet.M.coefficients))
        #self.inlet.beta.calculate(self.inlet.rr, len(self.inlet.M.coefficients))

        # get variation in stage loading coefficient
        psi_1 = psi * np.power(self.inlet.rr / self.inlet.r_mean, n - 1)

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
                    self.exit.rr[index] + 2 * (self.inlet.rr[index + 1] - self.inlet.rr[index]),
                    utils.Defaults.fine_grid
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
                                psi_1[index] - (self.exit.rr[index] / self.inlet.rr[index])**2
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
                            psi_1[-1] - (self.exit.rr[-1] / self.inlet.rr[-1])**2
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
            #self.exit.dpsi = self.exit.psi / psi_1 - 1

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
        """lower = 100 * np.concatenate((
            -2 * np.ones_like(self.inlet.M.coefficients),
            -np.pi * np.ones_like(self.inlet.M.coefficients)
        ))
        upper = 100 * np.concatenate((
            2 * np.ones_like(self.inlet.M.coefficients),
            np.pi * np.ones_like(self.inlet.M.coefficients)
        ))"""
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

    def evaluate(self):
        """Evaluates performance of the rotor blade row."""
        # find flow coefficient
        self.exit.phi = (
            self.inlet.M.value * np.cos(self.inlet.alpha.value)
            / (
                self.inlet.M.value * np.sin(self.inlet.alpha.value)
                - self.inlet.M_rel.value * np.sin(self.inlet.beta.value)
            )
        ) 
