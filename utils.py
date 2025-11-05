# 0.0 import modules

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.optimize import brentq
from scipy.optimize import root_scalar

# 0.1 global variables

gamma = 1.4

# 0.2 define Defaults class

class Defaults:
    """Container for default values relating to the engine system."""
    # define non-dimensional stage parameters
    flow_coefficient = 0.5
    stage_loading_coefficient = 0.4
    reaction = 0.8
    stagnation_pressure_loss_coefficient = 0.00
    vortex_exponent = 0

    # code iteration parameters
    M_min = 0.1
    M_max = 0.8
    N = 1
    no_of_annuli = 1

    # default dimensional values
    engine_diameter = 0.65
    hub_tip_ratio = 0.2

    # area change ratios
    blade_row_radius = 1
    blade_row_radius_ratio = 1

    # specify inlet swirl
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

def impulse_function(M):
    """Calculates the non-dimensional impulse function for a given Mach number."""
    return (1 + gamma * M**2) * np.power(1 + (gamma - 1) * M**2 / 2, -gamma / (gamma - 1))

def dynamic_pressure_function(M):
    """Calculates the ratio of dynamic pressure to stagnation pressure for a given Mach number."""
    return gamma * M**2 / 2 * np.power(1 + (gamma - 1) * M**2 / 2, -gamma / (gamma - 1))

def velocity_function(M):
    """Calculates the non-dimensional velocity for a given Mach number."""
    return np.sqrt(gamma - 1) * M * np.power(1 + (gamma - 1) * M**2 / 2, -1 / 2)

def invert(function, target, bracket = [1e-6, 1], method = "brentq"):
    """
    Numerically inverts a 1D function f(x), solving f(x) = y_target.
    
    Parameters
    ----------
    function : callable
        Function to invert, must take a single float and return a float.
    target : float
        Target output value of f(x).
    bracket : (float, float), optional
        Lower and upper x bounds where the root is searched.
        Required for bracketed methods like 'brentq' or 'bisect'.
    method : str, default 'brentq'
        Root-finding method ('brentq', 'bisect', 'secant', 'newton', etc.)
    
    Returns
    -------
    M : float
        The M value such that function(M_inverse) = target.
    """

    # define residual function
    def residual(x):

        return function(x) - target

    # solve for residual root
    try:
        
        sol = root_scalar(residual, bracket = bracket, method = method)

    except:

        #print(f"Unable to invert: {target}")
        return None

    if not sol.converged:

        raise RuntimeError("Inversion failed to converge.")
    
    return sol.root

# 0.4 upload NACA aerofoil for visualisation purposes

filename = 'naca0009.txt'
data = np.loadtxt(
    filename,
    skiprows = 1
)
#print(f"data: {data}")
#data *= 0.6
#data[:, 0] -= 0.25

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
        (  0, 160, 192),
        ( 75,   75, 255),
        (127,   0, 192)
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