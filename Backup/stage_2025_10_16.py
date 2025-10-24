# import modules

import utils_2025_10_16 as utils
from blade_row_2025_10_17 import Blade_row
from flow_state_2025_10_16 import Flow_state

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar

# define Stage class

class Stage:
    """
    Used to store a rotor-stator pair, making up a single compressor stage.
    
    Stored within the Engine class.
    
    Parameters
    ----------
    None
    """
    def __init__(self):
        """Create instance of the Stage class."""
        self.blade_rows = []
        self.phi = utils.Defaults.flow_coefficient
        self.psi = utils.Defaults.stage_loading_coefficient
        self.reaction = utils.Defaults.reaction

        # create rotor
        self.blade_rows.append(
            Blade_row(
                utils.Defaults.blade_row_area_ratio,
                1,
                utils.Defaults.stagnation_pressure_loss_coefficient,
                True
            )
        )

        # create stator
        self.blade_rows.append(
            Blade_row(
                utils.Defaults.blade_row_area_ratio,
                1,
                utils.Defaults.stagnation_pressure_loss_coefficient,
                False
            )
        )

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

    def define_blade_geometry(self, last_stage=False):
        """Determines the blade geometry of the stage for a given inlet flow state."""
        # solve for inlet relative Mach number and swirl angle via vector algebra
        rotor = self.blade_rows[0]
        v1 = rotor.inlet.M * np.array([np.cos(rotor.inlet.alpha), np.sin(rotor.inlet.alpha)])
        v2 = rotor.inlet.M * np.cos(rotor.inlet.alpha) / self.phi * np.array([0, 1])
        v3 = v1 - v2
        rotor.inlet.M_rel = np.linalg.norm(v3)
        rotor.inlet.beta = np.arctan2(v3[1], v3[0])
        rotor.inlet_blade_Mach_number = np.linalg.norm(v2)

        # find relative stagnation pressure ratio after stagnation pressure loss
        relative_stagnation_pressure_ratio = (
            1 - rotor.Y_p * (1 - utils.stagnation_pressure_ratio(rotor.inlet.M_rel))
        )

        def residual(exit_M_rel_guess):
            """Find exit relative swirl angle which reduces residual to zero."""
            # guess the value of cos(beta) via stage loading coefficient
            sin_beta1 = (
                (
                    (self.psi - 1) * rotor.inlet_blade_Mach_number
                    + rotor.inlet.M * np.sin(rotor.inlet.alpha)
                ) / (
                    exit_M_rel_guess * np.sqrt(
                        utils.stagnation_temperature_ratio(exit_M_rel_guess)
                        / utils.stagnation_temperature_ratio(rotor.inlet.M_rel)
                    )
                )
            )
            if 1 - sin_beta1**2 < 0:

                cos_beta1 = np.nan
            
            else:

                cos_beta1 = np.sqrt(1 - sin_beta1**2)

            # guess the value of cos(beta) via conservation of mass
            cos_beta2 = (
                utils.mass_flow_function(rotor.inlet.M_rel)
                / relative_stagnation_pressure_ratio
                * np.cos(rotor.inlet.beta)
                / utils.mass_flow_function(exit_M_rel_guess)
            )
            
            if not np.isfinite(cos_beta1 - cos_beta2):

                return -1e3
            
            return cos_beta1 - cos_beta2
    
        # debugging
        """xx = np.linspace(1e-3, 1, 50)
        yy = [residual(x) for x in xx]
        fig, ax = plt.subplots()
        ax.plot(xx, yy)
        plt.show()"""

        # solve for the exit relative Mach number recursively and determine the relative swirl angle
        solution = root_scalar(
            residual, bracket = [1e-3, 1.0], method = "brentq"
        )
        M_rel_exit = solution.root
        beta_exit = (
            np.arcsin(
                (
                    (self.psi - 1) * rotor.inlet_blade_Mach_number + rotor.inlet.M * np.sin(rotor.inlet.alpha)
                ) / (
                    M_rel_exit * np.sqrt(
                        utils.stagnation_temperature_ratio(M_rel_exit)
                        / utils.stagnation_temperature_ratio(rotor.inlet.M_rel)
                    )
                )
            )
        )

        # solve for exit Mach number and swirl angle via vector algebra
        v1 = M_rel_exit * np.array([np.cos(beta_exit), np.sin(beta_exit)])
        v2 = (
            rotor.inlet_blade_Mach_number * np.sqrt(
                utils.stagnation_temperature_ratio(rotor.inlet.M_rel)
                / utils.stagnation_temperature_ratio(M_rel_exit)
            ) * np.array([0, 1])
        )
        v3 = v1 + v2
        M_exit = np.linalg.norm(v3)
        alpha_exit = np.arctan2(v3[1], v3[0])

        # find stagnation temperature ratio
        T_0 = (
            rotor.inlet.T_0
            / utils.stagnation_temperature_ratio(M_exit)
            * utils.stagnation_temperature_ratio(M_rel_exit)
            * utils.stagnation_temperature_ratio(rotor.inlet.M)
            / utils.stagnation_temperature_ratio(rotor.inlet.M_rel)
        )

        # find stagnation pressure ratio
        p_0 = (
            rotor.inlet.p_0
            / utils.stagnation_pressure_ratio(M_exit)
            * utils.stagnation_pressure_ratio(M_rel_exit)
            * utils.stagnation_pressure_ratio(rotor.inlet.M)
            / utils.stagnation_pressure_ratio(rotor.inlet.M_rel)
            * relative_stagnation_pressure_ratio
        )

        # store flow conditions at rotor exit in rotor
        rotor.exit = Flow_state(
            M_exit,
            alpha_exit,
            T_0,
            p_0,
            M_rel_exit,
            beta_exit
        )

        # save blade angles and blade Mach number to rotor
        rotor.inlet_blade_angle = rotor.inlet.beta
        rotor.exit_blade_angle = rotor.exit.beta
        rotor.blade_Mach_number = rotor.exit.M * np.sin(rotor.exit.alpha) - rotor.exit.M_rel * np.sin(rotor.exit.beta)

        # set stator inlet equal to rotor exit
        stator = self.blade_rows[-1]
        stator.inlet = rotor.exit

        # find stagnation pressure ratio after stagnation pressure loss
        stagnation_pressure_ratio = (
            1 - stator.Y_p * (1 - utils.stagnation_pressure_ratio(stator.inlet.M))
        )

        def zero_angle_stator():

            # find exit Mach number via mass flow function
            exit_mass_flow_function = (
                utils.mass_flow_function(stator.inlet.M) * np.cos(stator.inlet.alpha)
                / stagnation_pressure_ratio
            )
            return utils.invert(utils.mass_flow_function, exit_mass_flow_function), 0

        # if stator belongs to the last stage in the engine, set exit blade angle to zero
        if last_stage:

            M_exit, alpha_exit = zero_angle_stator()

        # solve exit blade angle to give defined reaction
        else:

            # find Mach number at stator exit from fixed reaction
            M_exit = (
                utils.invert(
                    utils.stagnation_temperature_ratio,
                    (
                        (
                            utils.stagnation_temperature_ratio(rotor.exit.M)
                            / utils.stagnation_temperature_ratio(rotor.inlet.M)
                            * rotor.exit.T_0 / rotor.inlet.T_0 - 1
                        ) / self.reaction + 1
                    )
                    * utils.stagnation_temperature_ratio(rotor.inlet.M)
                    * rotor.inlet.T_0 / rotor.exit.T_0
                )
            )

            # reaction cannot be achieved
            if M_exit == None:

                print(
                    f"{utils.Colours.RED}Reaction of {self.reaction} could not be achieved! "
                    f"Setting stator exit angle to zero.{utils.Colours.END}"
                )
                M_exit, alpha_exit = zero_angle_stator()

            else:

                # solve for exit angle
                alpha_exit = (
                    np.arccos(
                        utils.mass_flow_function(stator.inlet.M) / utils.mass_flow_function(M_exit)
                        * np.cos(stator.inlet.alpha) / stagnation_pressure_ratio
                    )
                )

                if not np.isfinite(alpha_exit):

                    print(
                        f"{utils.Colours.RED}Reaction of {self.reaction} could not be achieved! "
                        f"Setting stator exit angle to zero.{utils.Colours.END}"
                    )
                    M_exit, alpha_exit = zero_angle_stator()

        # stagnation temperature is conserved across stator row
        T_0 = stator.inlet.T_0

        # stagnation pressure ratio is known from stagnation_pressure_loss_coefficient
        p_0 = stator.inlet.p_0 * stagnation_pressure_ratio

        # store flow conditions at stator exit in stator
        stator.exit = Flow_state(
            M_exit,
            alpha_exit,
            T_0,
            p_0
        )

        # save blade angles to stator
        stator.inlet_blade_angle = stator.inlet.alpha
        stator.exit_blade_angle = stator.exit.alpha
        stator.inlet_blade_Mach_number = 0
        stator.blade_Mach_number = 0

        # store rotor and stator references for convenience
        self.rotor = rotor
        self.stator = stator

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