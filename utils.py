# 0.0 import modules

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.interpolate import make_interp_spline

# 0.1 global variables

gamma = 1.4
c_p = 1005
R = 287

rho_aluminium = 2700
resistivity_copper = 1.68e-8

# 0.2 define Defaults class

class Defaults:
    """Container for default values relating to the engine system."""
    # default dimensional values
    altitude = 0
    flight_speed = 170
    thrust = 1

    # default flight scenario parameters
    label = ""
    diameter = 0.2
    hub_tip_ratio = 0.3
    flight_scenarios = {
        "Default": ["Default", altitude, flight_speed, diameter, hub_tip_ratio, thrust],
        "High speed": ["High speed", 0, 170, diameter, hub_tip_ratio, 100],
        "Take-off": ["Take-off", 0, 20, 0.14, 0.3571, 30],
        "Static": ["Static", 0, 0, 0.14, 0.3571, 50],
        "Cruise": ["Cruise", 3000, 40, 0.14, 0.3571, 20],
    }

    # default engine input parameters
    no_of_stages = 1
    vortex_exponent = 0.5
    Y_p = 0.04
    phi = 0.6
    psi = 0.15
    area_ratio = 1

    # default geometry parameters
    aspect_ratio = 2.5
    diffusion_factor = 0.4
    design_parameter = 1.15
    min_no_of_blades = 6
    max_no_of_blades = 20

    # guardrails
    min_pitch_to_chord_ratio = 0.2
    axial_separation = 0.15

    # chord distribution limits
    max_chord_limit = 0.8
    chord_ratio_limit = 0.5

    # default motor parameters
    motor_power = 1000
    motor_rpm = 10000
    motor_diameter = 0.1
    cable_diameter = 1
    max_current_density = 5

    # default off_design parameters
    phi_min = 0.4
    phi_max = 1

    # fonts
    fontsize = 12

    # default figure size tuple
    figsize = (10, 6)

    # default plotting parameters
    quantity_list = [
        [
            'M', 'Mach number',
            'M_rel', 'Relative Mach number',
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
        [
            'p', 'Static pressure'
        ]
    ]

    # code iteration parameters
    solver_grid = 101
    export_grid = 51
    off_design_grid = 10
    maxiter = 50

    M_1 = ""

    # specify inlet swirl
    inlet_swirl = 0

    # whether or not debug mode is active
    debug = False

    # default dimensional blade thickness (in mm)
    max_thickness_mm = 2
    thickness_fraction = 0.5

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

def soft_clip(x, a_min=None, a_max=None, sharpness = 8):
    if a_max is not None:
        x = a_max - (1 / sharpness) * np.log1p(np.exp(sharpness * (a_max - x)))
    if a_min is not None:
        x = a_min + (1 / sharpness) * np.log1p(np.exp(sharpness * (x - a_min)))
    return x

# 0.4 upload NACA aerofoil for visualisation purposes

"""filename = 'naca0009.txt'
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
aerofoil_data = np.array([x_fine, y_fine])"""

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

def bound(x, a = 0.01, b = 0.5, c = 0.7):
    """Bounds a value between 0 and c using a logistic curve."""
    # check if x is too small
    if x < a:

        # return lower bound
        return a * np.exp((x - a) / a) + 1e-3

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

# create Labels class
class Labels:
    """Stores various labels used in the gui for displaying engine design information."""
    # list of label-pairs required to create a scenario object
    scenario_input_labels = [
        ["Label", "label"],
        ["Altitude (m)", "altitude"],
        ["Flight Speed (m/s)", "flight_speed"],
        ["Diameter (m)", "diameter"],
        ["Hub-tip Ratio", "hub_tip_ratio"],
        ["Thrust (N)", "thrust"]
    ]

    # list of scenario label-pairs to display (computed/output values)
    scenario_output_labels = [
        ["Mach Number", "M"],
        ["Thrust Coefficient", "C_th"]
    ]
    
    # list of label-pairs required to create an engine object
    engine_input_labels = [
        ["No. of Stages", "no_of_stages"],
        ["Flow Coefficient", "phi"],
        ["Stage Loading Coefficient", "psi"],
        ["Vortex Exponent", "vortex_exponent"],
        ["Stagnation Pressure Loss Coefficient", "Y_p"],
        ["Blade Row Area Ratio", "area_ratio"],
        ["Inlet Mach Number", "M_1"],
    ]

    # list of engine label-pairs to display (computed/output values)
    engine_output_labels = [
        ["Temperature Ratio", "T_0_ratio"],
        ["Pressure Ratio", "p_0_ratio"],
        ["Nozzle Area Ratio", "nozzle_area_ratio"],
        ["Jet Velocity Ratio", "jet_velocity_ratio"],
        ["Compressor Efficiency", "eta_comp"],
        ["Propulsive Efficiency", "eta_prop"],
        ["Overall Efficiency", "eta_overall"]
    ]

    # list of extra (dimensional) label-pairs to display
    engine_extra_labels = [
        ["Mass flow rate (kg/s)", "m_dot"]
    ]
    
    # list of label-pairs required to create a geometry object
    geometry_input_labels = [
        ["Aspect Ratio", "aspect_ratio"],
        ["Diffusion Factor", "diffusion_factor"],
        ["Design Parameter", "design_parameter"]
    ]
    geometry_output_labels = []

    # list of label-pairs required to create a thickness object
    thickness_input_labels = [
        ["Max. Blade Thickness", "max_thickness_mm"],
        ["Thickness Fraction", "thickness_fraction"]
    ]
    thickness_output_labels = []

    # list of label-pairs required to make a motor object
    motor_input_labels = [
        ["Min. Power (W)", "motor_power"],
        ["Min. RPM", "motor_rpm"],
        ["Max. Diameter (m)", "motor_diameter"],
        ["Cable Diameter (mm)", "cable_diameter"]
    ]
    motor_output_labels = [
        ["No. of Motors (Power)", "no_of_motors_power"],
        ["No. of Motors (RPM)", "no_of_motors_rpm"],
        ["No. of Motors (Diameter)", "no_of_motors_diameter"],
        ["Min. Motor Mass (g)", "motor_mass"],
        ["No. of Variants", "no_of_variants"],
        ["No. of Installable", "no_of_installable"],
        ["Phase Voltage (V)", "phase_voltage"],
        ["Phase Current (A)", "phase_current"],
        ["Current Density (A/mm^2)", "current_density"],
        ["Cable Voltage Drop (V/m)", "voltage_drop"]
    ]

    # list of label-pairs required to create an off_design object
    off_design_input_labels = [
        ["Min. Flow Coefficient", "phi_min"],
        ["Max. Flow Coefficient", "phi_max"]
    ]
    off_design_output_labels = []

