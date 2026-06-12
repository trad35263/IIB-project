# import modules

import utils
#from blade_row import Blade_row
from rotor import Rotor
from stator import Stator
#from flow_state import Flow_state

import numpy as np
from scipy.optimize import root_scalar

import copy

# define Stage class

class Stage:
    """
    Used to store a rotor-stator pair, making up a single compressor stage.
    
    Stored within the Engine class.
    
    Parameters
    ----------
    phi : float
        Flow coefficient.
    psi : float
        Stage loading coefficient.
    vortex_exponent : float
        Vortex exponent.
    Y_p : float
        Stagnation pressure loss coefficient.
    index : int
        Stage number.
    """
    def __init__(self, phi, psi, vortex_exponent, Y_p, index):
        """Create instance of the Stage class."""
        # store input parameters
        self.phi = phi
        self.psi = psi
        self.vortex_exponent = vortex_exponent
        self.Y_p = Y_p
        self.index = index

        # create list of blade rows
        self.blade_rows = []

        # create rotor
        self.rotor = Rotor(self.Y_p, self.phi, self.psi, self.vortex_exponent)
        self.blade_rows.append(self.rotor)

        # create stator
        self.stator = Stator(self.Y_p)
        self.blade_rows.append(self.stator)

        # store off-design calculations in a dictionary
        self.off_designs = []

    def __str__(self):
        """Prints a string representation of the stage."""
        string = ""
        for name, value in self.__dict__.items():

            if isinstance(value, (int, float)):

                if ("alpha" in name or "angle" in name or "beta" in name) and not value == 0:

                    string += f"{name}: {utils.Colours.GREEN}{utils.rad_to_deg(value):.4g} °"
                    string += f"{utils.Colours.END}\n"

                else:

                    string += f"{name}: {utils.Colours.GREEN}{value:.4g}{utils.Colours.END}\n"

        string += "\n"

        return string

    def evaluate(self):
        """Evaluates performance of the individual stage."""
        # evaluate rotor and stator performance
        self.rotor.evaluate()
        self.stator.evaluate(self.rotor.inlet.T)

    def calculate_off_design(self, hub_tip_ratio, phi):
        """Calculates the off-design performance of the stage."""
        # make copy of blade rows so they can be edited freely
        rotor_copy = copy.deepcopy(self.rotor)
        stator_copy = copy.deepcopy(self.stator)

        # rotor-solving function to find zero-residual conditions for
        def solve_rotor(v_x_hub):
            """Returns the relevant residual for a blade row for a given hub axial velocity."""
            # design blade row and return residual
            rotor_copy.calculate_off_design(v_x_hub, hub_tip_ratio, phi)
            residual = self.rotor.exit.rr[-1]**2 - rotor_copy.exit.rr[-1]**2
            return residual
        
        # set initial guess
        x0 = rotor_copy.inlet.v_x[0]
        x1 = 0.9 * x0

        # solve for blade row exit conditions
        sol = root_scalar(
            solve_rotor, x0 = x0, x1 = x1, method = 'secant', maxiter = utils.Defaults.maxiter
        )
        utils.debug(f"sol: {sol}")

        # set stator inlet conditions to rotor exit conditions, preserving blade metal angles
        stator_inlet_metal_angle = self.stator.inlet.metal_angle
        stator_copy.inlet = copy.deepcopy(rotor_copy.exit)
        stator_copy.inlet.metal_angle = stator_inlet_metal_angle

        # stator-solving function to find zero-residual conditions for
        def solve_stator(v_x_hub):
            """Finds the residual for an off-design stator for a given hub axial velocity."""
            # design blade row and return residual
            stator_copy.design(v_x_hub, hub_tip_ratio)
            residual = self.stator.exit.rr[-1]**2 - stator_copy.exit.rr[-1]**2
            return residual
        
        # set initial guess
        x0 = stator_copy.inlet.v_x[0]
        x1 = 0.9 * x0

        # solve for blade row exit conditions
        sol = root_scalar(solve_stator, x0 = x0, x1 = x1, method = 'secant', maxiter = utils.Defaults.maxiter)
        utils.debug(f"sol: {sol}")

        # store as copy of stage
        stage = Stage(self.phi, self.psi, self.vortex_exponent, self.Y_p, self.index)
        stage.rotor = rotor_copy
        stage.stator = stator_copy
        stage.evaluate()
        self.off_designs.append(stage)

# obsolete

    def evaluate2(self):
        """Determines the distribution and mean values of non-dimensional stage parameters."""
        # loop over inlet-exit pairs for the stage's rotor
        for (inlet, exit) in zip(self.rotor.inlet, self.rotor.exit):

            # determine local flow coefficient
            inlet.phi = (
                inlet.flow_state.M * np.cos(inlet.flow_state.alpha) / inlet.M_blade
            )

            # determine local stage loading coefficient
            inlet.psi = (
                (
                    exit.flow_state.M * np.sin(exit.flow_state.alpha)
                    * exit.r / inlet.r * np.sqrt(
                        exit.flow_state.T / inlet.flow_state.T
                    ) - inlet.flow_state.M * np.sin(inlet.flow_state.alpha)
                ) / inlet.M_blade
            )

            # temp
            inlet.M_x = inlet.flow_state.M * np.cos(inlet.flow_state.alpha)
            exit.M_x = exit.flow_state.M * np.cos(exit.flow_state.alpha)

            # store dimensionless mass flow rate
            inlet.m = (
                inlet.flow_state.M * np.cos(inlet.flow_state.alpha)
                * np.sqrt(inlet.flow_state.T) * inlet.A * inlet.flow_state.rho
            )
            exit.m = (
                exit.flow_state.M * np.cos(exit.flow_state.alpha)
                * np.sqrt(exit.flow_state.T) * exit.A * exit.flow_state.rho
            )

        # calculate total dimensionless mass flow rates
        m_sum_inlet = np.sum([inlet.m for inlet in self.rotor.inlet])
        m_sum_exit = np.sum([exit.m for exit in self.rotor.exit])

        # loop over rotor inlet-exit pairs
        for (inlet, exit) in zip(self.rotor.inlet, self.rotor.exit):

            # normalise mass flow rates
            inlet.m /= m_sum_inlet
            exit.m /= m_sum_exit

        # loop over rotor inlet-exit and stator exit triples
        for (inlet, rotor_exit, exit) in zip(self.rotor.inlet, self.rotor.exit, self.stator.exit):

            # determine local reaction
            inlet.reaction = (
                (rotor_exit.flow_state.T - inlet.flow_state.T)
                / (exit.flow_state.T - inlet.flow_state.T)
            )
            
            # temp
            exit.M_x = exit.flow_state.M * np.cos(exit.flow_state.alpha)

        # loop over stator inlet-exit pairs
        for (inlet, exit) in zip(self.stator.inlet, self.stator.exit):

            # store dimensionless mass flow rate
            inlet.m = (
                inlet.flow_state.M * np.cos(inlet.flow_state.alpha)
                * np.sqrt(inlet.flow_state.T) * inlet.A * inlet.flow_state.rho
            )
            exit.m = (
                exit.flow_state.M * np.cos(exit.flow_state.alpha)
                * np.sqrt(exit.flow_state.T) * exit.A * exit.flow_state.rho
            )

        # calculate total dimensionless mass flow rates
        m_sum_inlet = np.sum([inlet.m for inlet in self.stator.inlet])
        m_sum_exit = np.sum([exit.m for exit in self.stator.exit])

        # loop over stator inlet-exit pairs
        for (inlet, exit) in zip(self.stator.inlet, self.stator.exit):

            # normalise mass flow rates
            inlet.m /= m_sum_inlet
            exit.m /= m_sum_exit

    def determine_efficiency(self):
        """Determine key non-dimensional parameters of the given stage."""
        # determine flow coefficient
        self.phi = (
            self.rotor.inlet.M * np.cos(self.rotor.inlet.alpha)
            / self.rotor.inlet_blade_Mach_number
        )

        # determine stage loading coefficient
        self.psi = (
            (
                self.rotor.exit.M * np.sqrt(
                    utils.stagnation_temperature_ratio(self.rotor.exit.M)
                    / utils.stagnation_temperature_ratio(self.rotor.inlet.M)
                    * self.rotor.exit.T_0 / self.rotor.inlet.T_0
                ) * np.sin(self.rotor.exit.alpha)
                - self.rotor.inlet.M * np.sin(self.rotor.inlet.alpha)
            ) / self.rotor.inlet_blade_Mach_number
        )

        # determine reaction
        self.reaction = (
            (
                utils.stagnation_temperature_ratio(self.rotor.exit.M)
                / utils.stagnation_temperature_ratio(self.rotor.inlet.M)
                * self.rotor.exit.T_0 / self.rotor.inlet.T_0 - 1
            ) / (
                utils.stagnation_temperature_ratio(self.stator.exit.M)
                / utils.stagnation_temperature_ratio(self.rotor.inlet.M)
                * self.rotor.exit.T_0 / self.rotor.inlet.T_0 - 1
            )
        )

        # solve for isentropic and polytropic efficiency
        self.eta_s = (
            (
                np.power(
                    self.stator.exit.p_0 / self.rotor.inlet.p_0, (utils.gamma - 1) / utils.gamma
                ) - 1
            )
            / (self.stator.exit.T_0 / self.rotor.inlet.T_0 - 1)
        )
        self.eta_p = (
            (utils.gamma - 1) * np.log(self.stator.exit.p_0 / self.rotor.inlet.p_0)
            / (utils.gamma * np.log(self.stator.exit.T_0 / self.rotor.inlet.T_0))
        )
