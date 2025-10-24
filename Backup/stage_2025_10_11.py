# import modules

import utils_2025_10_11 as utils
from blade_row_2025_10_11 import Blade_row

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
        # create rotor
        self.blade_rows.append(
            Blade_row(
                utils.Defaults.rotor_blade_speed_ratio,
                utils.Defaults.rotor_blade_angle,
                utils.Defaults.blade_row_area_ratio,
                1,
                utils.Defaults.stagnation_pressure_loss_coefficient
            )
        )

        # create stator
        self.blade_rows.append(
            Blade_row(
                0,
                utils.Defaults.stator_blade_angle,
                utils.Defaults.blade_row_area_ratio,
                1,
                utils.Defaults.stagnation_pressure_loss_coefficient
            )
        )