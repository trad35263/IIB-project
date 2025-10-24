# 0.0 import modules

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.optimize import brentq

# 0.1 global variables

gamma = 1.4

# 0.2 define Defaults class

class Defaults:
    """Container for default values relating to the engine system."""
    # blade angle values
    inlet_guide_vanes_blade_angle = 10 * np.pi / 180        # rad
    rotor_blade_angle = -20 * np.pi / 180                   # rad
    stator_blade_angle = 10 * np.pi / 180                   # rad
    outlet_guide_vanes_blade_angle = 0

    # area change ratios
    intake_area_ratio = 0.8
    blade_row_area_ratio = 0.95
    nozzle_area_ratio = 0.5

    # turbomachine parameters
    rotor_blade_speed_ratio = 1
    hub_tip_ratio = 0.2
    stagnation_pressure_loss_coefficient = 0.01

    # values used for determining engine inlet conditions
    inlet_Mach_number = 0.3
    inlet_flow_coefficient = 0.5
    inlet_swirl = 0

# 0.3 compressible flow perfect gas relations

def stagnation_pressure_ratio(M):
    """Calculates the ratio of static to stagnation pressure for a given Mach number."""
    return np.power(1 + (gamma - 1) * M**2 / 2, -gamma / (gamma - 1))

def stagnation_temperature_ratio(M):
    """Calculates the ratio of static to stagnation temperature for a given Mach number."""
    return np.power(1 + (gamma - 1) * M**2 / 2, -1)

def stagnation_density_ratio(M):
    """Calculates the ratio of static to stagnation density for a given Mach number."""
    return np.power(1 + (gamma - 1) * M**2 / 2, -1 / (gamma - 1))

def mass_flow_function(M):
    """Calculates the non-dimensional mass flow rate for a given Mach number."""
    return (
        gamma * M * np.power(1 + (gamma - 1) * M**2 / 2, -(gamma + 1) / (2 * (gamma - 1)))
        / np.sqrt(gamma - 1)
    )

def solve_M_from_mass_flow_function(target, M_min=1e-6, M_max=1.0):
    """Solves for Mach number given a non-dimensional mass flow rate."""
    # residual function used for inverting operation
    def residual(M):
        return mass_flow_function(M) - target

    # root-finding within given bounds
    try:

        return brentq(residual, M_min, M_max, xtol=1e-12, rtol=1e-10)
    
    except ValueError as error:

        print(f"Target mass flow function: {Colours.RED}{target}{Colours.END}\n{error}")
        return solve_M_from_mass_flow_function(1.28)



def velocity_function(M):
    """Calculates the non-dimensional velocity for a given Mach number."""
    return np.sqrt(gamma - 1) * M * np.power(1 + (gamma - 1) * M**2 / 2, -1 / 2)

# 0.4 upload NACA aerofoil for visualisation purposes

filename = 'naca0012.txt'
data = np.loadtxt(
    filename,
    skiprows = 1
)
data *= 0.6
data[:, 0] -= 0.25

# 0.5 define Colours class

class Colours:
    """Class used to store ANSI escape sequences for printing colours."""

    RED = '\033[91m'
    ORANGE = '\033[38;5;208m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    PINK = '\033[38;5;212m'
    GREY = '\033[90m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'
    END = '\033[0m'

    rainbow_rgb = np.array([
        (224,   0,   0),
        (255, 160,   0),
        (224, 192,   0),
        (  0, 192,   0),
        (  0, 127, 255),
        ( 75,   0, 130),
        (192,   0, 192)
    ], dtype=float) / 255.0
    rainbow_colour_map = LinearSegmentedColormap.from_list("smooth_rainbow", rainbow_rgb)

    def __repr__(self):
        """Return string representation of all available colours."""
        string = ""
        for name, value in self.__class__.__dict__.items():
            if name.isupper():
                string += f"{value}{name}{self.END}\n"
        return string
    
# 0.6 unit conversion functions

def rad_to_deg(x):
    """Converts a float from radians into degrees (°)."""
    return 180 * x / np.pi

def deg_to_rad(x):
    """Converts a float from degrees (°) into radians."""
    return np.pi * x / 180