# 0.0 import modules

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.optimize import root_scalar
from scipy.interpolate import make_interp_spline

# 0.1 global variables

gamma = 1.4

# 0.2 define Defaults class

class Defaults:
    """Container for default values relating to the engine system."""
    # default flight scenario parameters
    label = ""

    # default engine input parameters
    no_of_stages = 1
    vortex_exponent = 0.5
    no_of_annuli = 3
    hub_tip_ratio = 0.025 / 0.070
    Y_p = 0.00
    phi = 0.6
    psi = 0.15

    # default plotting parameters
    quantity_list = [
        [
            'M', 'Mach number', True,
            'M_rel', 'Relative Mach number', True
        ],
        [
            'alpha', 'Flow angle (°)', True,
            'beta', 'Relative flow angle (°)', True,
            'metal_angle', 'Metal angle (°)', False
        ],
        ['M_x', 'Axial Mach number', False],
        [
            'pitch_to_chord', 'Pitch-to-chord ratio', False,
            's', 'Pitch', False,
            'c', 'Chord', False
        ],
        [
            'phi', 'Flow coefficient', False,
            'psi', 'Stage loading coefficient', False,
            'reaction', 'Reaction', False
        ],
        [
            'DF', 'Diffusion factor', False
        ]
    ]


    flow_coefficient = 0.6
    stage_loading_coefficient = 0.18


    stagnation_pressure_loss_coefficient = 0.00

    # code iteration parameters
    delta = 1e-6

    # default dimensional values
    diameter = 0.14
    altitude = 10000
    flight_speed = 30
    thrust = 20

    # placeholder for now
    blade_row_axial_depth = 0.5

    # specify inlet swirl
    inlet_swirl = 0

    # specify maximum permissible diffusion factor
    DF_limit = 0.4
    pitch_to_chord_limit = 1
    AR_target = 2.5

    # whether or not debug mode is active
    debug = False
    loading_bar = True

    # default flight scenarios
    flight_scenarios = {
        "Take-off": ["Take-off", 0, 20, diameter, hub_tip_ratio, 30],
        "Static": ["Static", 0, 0, diameter, hub_tip_ratio, 50],
        "Cruise": ["Cruise", 3000, 40, diameter, hub_tip_ratio, 20]
    }

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

def invert(function, target, bracket = [0, 1], method = "brentq"):
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
    """
    # define residual function
    def residual(x):

        return function(x) - target

    # solve for residual root
    try:
        
        sol = root_scalar(residual, bracket = bracket, method = method)

    except:

        return np.nan

    if not sol.converged:

        print(f"target: {target}")
        raise RuntimeError("Inversion failed to converge.")
    
    return sol.root

# 0.4 upload NACA aerofoil for visualisation purposes

filename = 'naca0009.txt'
data = np.loadtxt(
    filename,
    skiprows = 1
)
z = np.array([x for x in data if x[1] >= 0])
x = np.array(z[:-1, 0])[::-1]
y = np.array(z[:-1, 1])[::-1]
spline = make_interp_spline(x, y, k = 2)
x_fine = np.linspace(x.min(), x.max(), 10000)
y_fine = spline(x_fine)
aerofoil_data = np.array([x_fine, y_fine])

# 0.5 define Colours class

class Colours:
    """Class used to store ANSI escape sequences for printing colours."""
    # store ASCII codes for selected colours as class attributes
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
    # handle case where None is passed
    if x == None:

        return None

    # convert to degrees and return
    return 180 * x / np.pi

def deg_to_rad(x):
    """Converts a float from degrees (°) into radians."""
    return np.pi * x / 180

# 0.7 debugging function

def debug(string):
    """Prints a message to the terminal only if debug mode is activated."""
    if Defaults.debug:

        print(f"{string}")

M_infinity = 0.05877
M_1 = 0.152
M_1A_rel = 0.2413
M_2A_rel = 0.2133

M_1B_rel = 0.3167
M_2B_rel = 0.2750

M_1C_rel = 0.3759
M_2C_rel = 0.3283

print(f"stagnation_temperature_ratio(M_1): {stagnation_temperature_ratio(M_1)}")


print(f"mass_flow_function(M_1A_rel): {mass_flow_function(M_1A_rel)}")
print(f"stagnation_temperature_ratio(M_1A_rel): {stagnation_temperature_ratio(M_1A_rel)}")
print(f"invert(mass_flow_function, 0.4595): {invert(mass_flow_function, 0.4595)}")
print(f"stagnation_temperature_ratio(M_2A_rel): {stagnation_temperature_ratio(M_2A_rel)}")


print(f"mass_flow_function(M_1B_rel): {mass_flow_function(M_1B_rel)}")
print(f"stagnation_temperature_ratio(M_1B_rel): {stagnation_temperature_ratio(M_1B_rel)}")
print(f"invert(mass_flow_function, 0.5526): {invert(mass_flow_function, 0.5820)}")
print(f"stagnation_temperature_ratio(M_2B_rel): {stagnation_temperature_ratio(M_2B_rel)}")


print(f"mass_flow_function(M_1C_rel): {mass_flow_function(M_1C_rel)}")
print(f"stagnation_temperature_ratio(M_1C_rel): {stagnation_temperature_ratio(M_1C_rel)}")
print(f"invert(mass_flow_function, 0.6817): {invert(mass_flow_function, 0.6817)}")
print(f"stagnation_temperature_ratio(M_2C_rel): {stagnation_temperature_ratio(M_2C_rel)}")