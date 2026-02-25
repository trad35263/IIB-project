# 0.0 import modules

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.optimize import root_scalar
from scipy.interpolate import make_interp_spline

# 0.1 global variables

gamma = 1.4
c_p = 1005
R = 287

# 0.2 define Defaults class

class Defaults:
    """Container for default values relating to the engine system."""
    # default flight scenario parameters
    label = ""
    diameter = 0.14
    hub_tip_ratio = 0.3077
    flight_scenarios = {
        "Take-off": ["Take-off", 0, 10, diameter, hub_tip_ratio, 10],
        "Static": ["Static", 0, 0, diameter, hub_tip_ratio, 50],
        "Cruise": ["Cruise", 3000, 40, diameter, hub_tip_ratio, 20]
    }

    # default engine input parameters
    no_of_stages = 2
    vortex_exponent = 0.5
    solver_order = 2
    Y_p = 0.02
    phi = 0.6
    psi = 0.1596

    # default geometry parameters
    aspect_ratio = 2.5
    diffusion_factor = 0.3
    deviation_constant = 0.23
    max_blades = 20

    # default off_design parameters
    phi_min = 0.4
    phi_max = 1

    # default figure size tuple
    figsize = (10, 6)

    # default plotting parameters
    quantity_list = [
        [
            'M', 'Mach number',
            'M_rel', 'Relative Mach number'
        ],
        [
            'alpha', 'Flow angle (°)',
            'beta', 'Relative flow angle (°)',
            'metal_angle', 'Metal angle (°)'
        ],
        [
            'chord', 'Chord',
            'axial_chord', 'Axial chord'
        ],
        [
            'phi', 'Flow coefficient',
            'psi', 'Stage loading coefficient',
            'reaction', 'Reaction'
        ],
        [
            'p_0', 'Stagnation pressure',
            'T_0', 'Stagnation temperature'
        ],
        [
            'v_x', 'Axial velocity',
            'v_theta', 'Tangential velocity'
        ],
        [
            'diffusion_factor', 'Diffusion factor'
        ],
        [
            'deviation', 'Deviation angle (°)'
        ],
    ]

    # code iteration parameters
    solver_grid = 101
    export_grid = 51
    off_design_grid = 10
    maxiter = 50

    # default dimensional values
    altitude = 10000
    flight_speed = 30
    thrust = 20

    # specify inlet swirl
    inlet_swirl = 0

    # whether or not debug mode is active
    debug = False
    loading_bar = True

    nfev = 500

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
    #if x == None:

        #return None

    # convert to degrees and return
    return 180 * x / np.pi

def deg_to_rad(x):
    """Converts a float from degrees (°) into radians."""
    return np.pi * x / 180

def rad_s_to_rpm(x):
    """Converts a float from rad/s to rpm."""
    return x * 60 / (2 * np.pi)

# 0.7 debugging function

def debug(string):
    """Prints a message to the terminal only if debug mode is activated."""
    if Defaults.debug:

        print(f"{string}")

# 0.8 logistic curve

def bound(x, a = 0.01, b = 0.5, c = 0.6):
    """Bounds a value between 0 and c using a logistic curve."""
    # check if x is too small
    if x < a:

        # return lower bound
        return a * np.exp((x - a) / a)

    # check if x is too large
    elif x > b:

        # return upper bound
        return c - (c - b) * np.exp(-(x - b) / (c - b))

    # intermediate cases
    else:

        # return as is
        return x
    
# 0.9 faster cumulative trapezoid function

def cumulative_trapezoid(x, y, initial = 0):
    """Computes the cumulative trapezoidal integral for arrays x and y = y(x)."""
    # extract necessary x-coordinate information.
    dx = np.diff(x)
    mids = 0.5 * (y[:-1] + y[1:])

    # cumulative sum of mids * dx, prepend initial
    result = np.zeros(len(x))
    result[0] = initial
    result[1:] = np.cumsum(mids * dx) + initial
    return result

M_infinity = 0.05877
M_1 = 0.152
M_1A_rel = 0.2413
M_2A_rel = 0.2133

M_1B_rel = 0.3167
M_2B_rel = 0.2750

M_1C_rel = 0.3759
M_2C_rel = 0.3283

"""print(f"stagnation_temperature_ratio(M_1): {stagnation_temperature_ratio(M_1)}")


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
print(f"stagnation_temperature_ratio(M_2C_rel): {stagnation_temperature_ratio(M_2C_rel)}")"""
