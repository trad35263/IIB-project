# import modules

import numpy as np
from time import perf_counter as timer

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

    def design(self, v_x_hub, hub_tip_ratio):
        """Determines the flowfield through the nozzle and solves for its geometry."""
        # start timer
        t1 = timer()

        # impose bounds on hub velocity guess
        v_x_hub = utils.bound(v_x_hub)

        # ensure exit arrays have the correct length
        self.exit.rr = np.zeros(len(self.inlet.M))
        self.exit.v_x = np.zeros(len(self.inlet.M))
        self.exit.v_theta = np.zeros(len(self.inlet.M))
        self.exit.M = np.zeros(len(self.inlet.M))
        self.exit.T = np.zeros(len(self.inlet.M))
        self.exit.p = np.zeros(len(self.inlet.M))

        # hub dimensionless axial velocity and radius are known
        self.exit.v_x[0] = v_x_hub
        self.exit.rr[0] = 1e-3

        # nozzle is isentropic
        self.exit.T_0 = self.inlet.T_0
        self.exit.p_0 = self.inlet.p_0
        self.exit.s = self.inlet.s
        
        # set exit angle distribution to zero (make this more general later)
        self.exit.alpha = np.zeros(len(self.inlet.M))  # technically this is done already but included for clarity

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
        for index in range(len(self.inlet.M)):
                
            # for all cases except hub streamline
            if index > 0:
                
                # create fine grid for calculating streamtube upper bound 
                r_2_fine = np.linspace(
                    self.exit.rr[index - 1],
                    self.exit.rr[index - 1] + 10 * (self.inlet.rr[index] - self.inlet.rr[index - 1]),
                    len(self.inlet.M)
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
                d_tan_alpha = np.tan(self.exit.alpha[index]) - np.tan(self.exit.alpha[index - 1])

                # calculate all v_x terms together
                v_x_term = (
                    dT_0 / (utils.gamma - 1)
                    - self.exit.T[index - 1] * ds
                    - self.exit.v_x[index - 1]**2 * np.tan(self.exit.alpha[index - 1])**2
                    * (self.exit.rr[index] / self.exit.rr[index - 1] - 1)
                    - self.exit.v_x[index - 1]**2 * np.tan(self.exit.alpha[index - 1])
                    * d_tan_alpha
                )

                # calculate dimensionless axial velocity at new radial position
                self.exit.v_x[index] = (
                    v_x_term / (self.exit.v_x[index - 1] * (1 + np.tan(self.exit.alpha[index - 1])**2))
                    + self.exit.v_x[index - 1]
                )

            # get exit tangential velocity from axial velocity and flow angle
            self.exit.v_theta[index] = self.exit.v_x[index] * np.tan(self.exit.alpha[index])

            # get Mach number
            v_squared = (self.exit.v_x[index] / np.cos(self.exit.alpha[index]))**2
            self.exit.M[index] = np.sqrt(v_squared / (1 - 0.5 * (utils.gamma - 1) * v_squared))

        # get exit static conditions
        self.exit.T = self.exit.T_0 * utils.stagnation_temperature_ratio(self.exit.M)
        self.exit.p = self.exit.p_0 * utils.stagnation_pressure_ratio(self.exit.M)

        # calculate exit mass flow rate
        self.exit.dm_dot_dr = (
            2 * utils.gamma / ((1 - hub_tip_ratio**2) * np.sqrt(utils.gamma - 1))
            * self.exit.p / np.sqrt(self.exit.T)
            * self.exit.M * np.cos(self.exit.alpha) * self.exit.rr
        )
        self.exit.m_dot = utils.cumulative_trapezoid(self.exit.rr, self.exit.dm_dot_dr)

        # end timer
        t2 = timer()
        utils.debug(
            f"Nozzle design completed in {utils.Colours.GREEN}{t2 - t1:.4g}{utils.Colours.END} s!"
        )

    def evaluate(self, hub_tip_ratio):
        """Evaluates the nozzle performance."""
        # store nozzle area ratio
        self.area_ratio = self.exit.rr[-1]**2 / (1 - hub_tip_ratio**2)

        # find cumulative thrust coefficient distribution
        dC_th_dr = (    # with pressure terms
            2 / (1 - hub_tip_ratio**2) * (
                utils.impulse_function(self.exit.M)
                - 2 * utils.dynamic_pressure_function(self.exit.M)
                * (np.sin(self.exit.alpha))**2
            ) * self.exit.p_0 * self.exit.rr
        )
        self.C_th = utils.cumulative_trapezoid(self.exit.rr, dC_th_dr)

        """dC_th_dr = (   # without pressure terms
            2 * utils.gamma / (1 - hub_tip_ratio**2)
            * self.exit.p * self.exit.M**2 * np.cos(self.exit.alpha)**2 * self.exit.rr
        )
        self.C_th_without_p = utils.cumulative_trapezoid(self.exit.rr, dC_th_dr)"""

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
