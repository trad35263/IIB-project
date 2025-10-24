# 0.0 import modules

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.optimize import brentq

# 0.1 global variables

p_atm = 1e5                 # Pa
T_atm = 288                 # K
c_p = 1005                  # J/kgK
gamma = 1.4
R = c_p * (1 - 1 / gamma)   # J/kgK

# 0.2 define Defaults class

class Defaults:
    """Container for default values relating to the engine system."""
    # values used in constructing blade rows
    rotor_blade_angle = -45 * np.pi / 180           # rad
    stator_blade_angle = 10 * np.pi / 180           # rad
    blade_speed = 100                               # m/s
    inlet_tip_radius = 0.5                          # m
    exit_tip_radius = 0.4                           # m
    hub_tip_ratio = 0.2
    stagnation_pressure_loss_coefficient = 0.1

    # values used for determining engine inlet conditions
    inlet_swirl = 0
    air_speed = 20                                 # m/s

# 0.3 compressible flow perfect gas relations

def stagnation_pressure_ratio(M, gamma):
    """Calculates the ratio of static to stagantion pressure for a given Mach number."""
    return np.power(1 + (gamma - 1) * M**2 / 2, -gamma / (gamma - 1))

def stagnation_temperature_ratio(M, gamma):
    """Calculates the ratio of static to stagantion temperature for a given Mach number."""
    return np.power(1 + (gamma - 1) * M**2 / 2, -1)

def non_dim_m_dot(M, gamma):
    """Calculates the non-dimensional mass flow rate for a given Mach number."""
    return (
        gamma * M * np.power(1 + (gamma - 1) * M**2 / 2, -(gamma + 1) / (2 * (gamma - 1)))
        / np.sqrt(gamma - 1)
    )

def solve_M_from_m_dot(target, gamma, M_min=1e-6, M_max=1.0):
    """Solves for Mach number given a non-dimensional mass flow rate."""
    # residual function used for inverting operation
    def residual(M):
        return non_dim_m_dot(M, gamma) - target

    # root-finding within given bounds
    M = brentq(residual, M_min, M_max, xtol=1e-12, rtol=1e-10)
    return M

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

# 1.0 define Engine class

class Engine:
    """
    Used to store multiple (if applicable) stages and determine the overall engine performance.
    
    Parameters
    ----------
    None

    Public Methods
    --------------

    solve_velocity_triangles(self): solves for velocity at each point in the engine
    solve_flow(self, blade_row): solves for the velocities at exit to a blade row
    visualise_velocity_triangles(self): asdf
    calculate_mass_flow_rate(self):
    """
    def __init__(self):
        """Create instance of the Engine class."""
        self.blade_rows = []

    def __str__(self):
        """Print a simplifed summary of the information contained in the class."""
        # print header and state number of blade rows
        string = f"""
{Colours.UNDERLINE}Engine Summary{Colours.END}

Number of blade rows: {Colours.GREEN}{len(self.blade_rows)}{Colours.END}
Blade row configuration:
"""
        
        # print abbreviated form of each type of blade row and return final string
        for blade_row in self.blade_rows:
            string += f"{blade_row.type}-"
        string = string[:-1]
        return string

    def __repr__(self):
        """Print a detailed summary of the information contained in the class."""
        string = f"""
{Colours.UNDERLINE}Engine Details{Colours.END}
"""
        for index, blade_row in enumerate(self.blade_rows):
            string += f"\n[{index + 1}] {blade_row}"
        return string

    def assign_stages(self):
        """"Determines which blade rows to count as one stage and assigns a unique colour."""
        # create empty list to store stages
        self.stages = []

        # set counters to zero and iterate over all blade rows
        found_rotor = False
        for index, blade_row in enumerate(self.blade_rows):
            
            # if blade row is a rotor and there hasn't been a rotor previously, create a new stage
            if "Rotor" in blade_row.row_type and not found_rotor:

                stage = Stage()
                self.stages.append(stage)
                found_rotor = True

            # store blade row in appropriate stage
            if len(self.stages) > 0:

                self.stages[-1].blade_rows.append(blade_row)

            # if blade row is not a rotor, start looking for a new rotor again
            if "S" in blade_row.type or "CR" in blade_row.type:

                found_rotor = False

        # assign colours
        for index, stage in enumerate(self.stages):

            if len(self.stages) == 1:
                
                stage.colour = Colours.rainbow_colour_map(0)

            else:

                stage.colour = Colours.rainbow_colour_map(index / (len(self.stages) - 1))

            # assign stage number and colour to blade row
            for blade_row in stage.blade_rows:

                blade_row.colour = stage.colour
    
    def calculate_mass_flow_rate(self):
        """Calculates mass flow rate given the flow state at engine inlet."""
        inlet = self.blade_rows[0].inlet
        self.m_dot = inlet.rho * inlet.axial_velocity * self.blade_rows[0].area

    def solve_velocity_triangles(self):
        """Solves for axial and tangential velocity at inlet and exit to each blade row."""

        # iterate over all stages and solve for flow
        for index, blade_row in enumerate(self.blade_rows):

            if index > 0:

                blade_row.inlet = self.solve_area_change(*self.blade_rows[index - 1:index + 1])

            blade_row.exit = self.solve_flow(blade_row)

    def solve_flow(self, blade_row):
        """Solves for velocities at exit to a given blade row."""
        # work out relative flow quantities and apply stagnation pressure loss coefficient
        blade_row.inlet.relative_quantities(blade_row.blade_speed)
        p_0_rel = (
            blade_row.inlet.p_0_rel - blade_row.Y_p * (blade_row.inlet.p_0_rel - blade_row.inlet.p)
        )

        # relative stagnation temperature is constant and find non-dimensional mass flow rate 
        T_0_rel = blade_row.inlet.T_0_rel
        non_dim_m_dot_rel = (
            self.m_dot * np.sqrt(c_p * T_0_rel) * np.cos(blade_row.metal_angle)
            / (blade_row.area * p_0_rel)
        )
        M_rel = solve_M_from_m_dot(non_dim_m_dot_rel, gamma)

        # solve for static quantities and apply continuity to determine new axial velocity
        p = p_0_rel * stagnation_pressure_ratio(M_rel, gamma)
        T = T_0_rel * stagnation_temperature_ratio(M_rel, gamma)
        rho = p / (R * T)
        v_x = blade_row.inlet.axial_velocity * blade_row.inlet.rho / rho

        # determine tangential velocity in the relative frame and then convert to absolute
        v_t_rel = (
            v_x * np.tan(blade_row.metal_angle)
        )
        v_t = v_t_rel + blade_row.blade_speed

        # store results as instance of Flow_state class and return
        exit_flow = Flow_state(v_x, v_t, p, T)
        return exit_flow
    
    def solve_area_change(self, blade_row1, blade_row2):
        """Solve for the flow state after an annulus area change with no work done."""
        # find exit tangential velocity via conservation of angular momentum
        v_t = (
            blade_row1.exit.tangential_velocity * (blade_row1.tip_radius + blade_row1.hub_radius)
            / (blade_row2.tip_radius + blade_row2.hub_radius)
        )
        
        # stagnation pressure and temperature are conserved through area change
        p_0 = blade_row1.exit.p_0
        T_0 = blade_row1.exit.T_0

        # establish velocity bounds, limiting v_max to the speed of sound
        v_min = 1e-16
        v_max = blade_row1.exit.a

        # establish LHS quantity and solve iteratively
        lhs = self.m_dot * R * T_0 / (blade_row2.area * p_0)
        def residual(vx):
            rhs = vx * np.power(1.0 - (vx**2 + v_t**2) / (2.0 * c_p * T_0), 1.0 / (gamma - 1.0))
            return rhs - lhs

        # uncomment for debugging
        """xx = np.linspace(v_min, v_max, 20)
        yy = residual(xx)
        fig, ax = plt.subplots()
        ax.plot(xx, yy)
        ax.grid()
        plt.show()"""

        # solve numerically for vx and determine static quantities
        v_x = brentq(residual, v_min, v_max)
        Vr = np.sqrt(v_x**2 + v_t**2)
        p = p_0 * np.power(1.0 - Vr**2 / (2.0 * c_p * T_0), gamma / (gamma - 1.0))
        T = T_0 * (1.0 - Vr**2 / (2.0 * c_p * T_0))

        # create instance of Flow_state class and return
        exit_flow = Flow_state(v_x, v_t, p, T)
        return exit_flow
    
    def visualise_velocity_triangles(self):
        """Function to plot the velocity triangles and pressure and temperature distributions."""
        # create matplotlib plot with multiple axes
        height_ratio = 24 / (len(self.blade_rows) + 2)
        height_ratio = 24 / (2 * len(self.blade_rows))
        fig, (ax_upper, ax_lower) = plt.subplots(
            2, 1, figsize = (12, 7),
            gridspec_kw={'height_ratios': [height_ratio, 1]},
            sharex = True
        )
        ax_lower1 = ax_lower.twinx()
        ax_lower2 = ax_lower.twinx()

        # set title and subtitle
        ax_upper.title.set_text(
            f"Air speed: {self.blade_rows[0].inlet.axial_velocity:.3g} m/s\n"
            f"Nominal blade speed: {Defaults.blade_speed:.3g} m/s"
        )
        ax_upper.title.set_fontsize(12)
        ax_lower.title.set_text(
            f"Stagnation pressure ratio = {self.p_r:.3g}\n"
            f"Stagnation temperature ratio = {self.T_r:.3g}\n"
            f"Isentropic efficiency = {self.eta_s:.3g}\n"
            f"Polytropic efficiency = {self.eta_p:.3g}"
        )
        ax_lower.title.set_fontsize(10)

        # store pressure and temperature distributions and tip radii
        pressures = [
            p for blade_row in self.blade_rows
            for p in (blade_row.inlet.p_0, blade_row.exit.p_0)
        ]
        temperatures = [
            T for blade_row in self.blade_rows
            for T in (blade_row.inlet.T_0, blade_row.exit.T_0)
        ]
        radii = [
            r for blade_row in self.blade_rows
            for r in (blade_row.tip_radius, blade_row.tip_radius)
        ]

        # plot stored quantities against blade row number
        xx = np.arange(2 * len(self.blade_rows)) / 2 + 0.75
        ax_lower.plot(
            xx, pressures, color = 'C1', label = 'Pressure (Pa)'
        )
        ax_lower1.plot(
            xx, temperatures, color = 'C2', label = 'Temperature (K)'
        )
        ax_lower2.plot(
            xx, radii, color = 'k', label = 'Tip radius (m)'
        )
        ax_lower2.plot(
            xx, -np.array(radii), color = 'k'
        )

        # iterate over each blade row
        for index, blade_row in enumerate(self.blade_rows):

            # construct rotation matrix and carry out matrix multiplication
            R = np.array([
                [np.cos(blade_row.metal_angle), -np.sin(blade_row.metal_angle)],
                [np.sin(blade_row.metal_angle),  np.cos(blade_row.metal_angle)]
            ])
            coords = data.copy()
            data_rot = coords @ R.T

            # display rotated NACA aerofoils in position
            ax_upper.plot(data_rot[:, 0] + index + 1, data_rot[:, 1], color = blade_row.colour)

            # display velocity triangles as arrows
            ax_upper.annotate(
                "",
                xy = (
                    index + 1 + blade_row.exit.axial_velocity / Defaults.blade_speed,
                    blade_row.exit.tangential_velocity / Defaults.blade_speed
                ),
                xytext = (index + 1, 0),
                arrowprops = dict(
                    arrowstyle = "->", color = 'C0',
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )
            ax_upper.plot(
                [0], [blade_row.exit.tangential_velocity / Defaults.blade_speed], linestyle = ''
            )

            # display blade row speeds
            ax_upper.annotate(
                "",
                xy = (
                    index + 1,
                    blade_row.blade_speed / Defaults.blade_speed
                ),
                xytext = (index + 1, 0),
                arrowprops = dict(
                    arrowstyle = "->", color = 'C3',
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )
            ax_upper.plot(
                [0], [blade_row.blade_speed / Defaults.blade_speed], linestyle = ''
            )

            # store pressure and temperature for plotting
            pressures.append(blade_row.exit.p)
            temperatures.append(blade_row.exit.T)

        # label velocity arrows in legend
        ax_upper.plot([], [], color = 'C0', label = 'Absolute flow velocity')
        ax_upper.plot([], [], color = 'C3', label = 'Blade speeds')

        # set axis limits and aspect ratio
        ax_upper.set_xlim(1e-6, len(self.blade_rows) + 1 - 1e-6)
        ax_upper.set_aspect('equal')

        # set x-axis grid lines to be at integer intervals
        loc = plticker.MultipleLocator(base=1.0)
        ax_upper.xaxis.set_major_locator(loc)
        ax_lower.xaxis.set_major_locator(loc)

        # gather all legend handles on the lower plot
        lines_1, labels_1 = ax_lower.get_legend_handles_labels()
        lines_2, labels_2 = ax_lower1.get_legend_handles_labels()
        lines_3, labels_3 = ax_lower2.get_legend_handles_labels()

        # create legends for both plots
        ax_upper.legend()
        ax_lower.legend(
            lines_1 + lines_2 + lines_3,
            labels_1 + labels_2 + labels_3,
            loc = 'right'
        )

        # colour plot ticks
        ax_upper.tick_params(axis = 'y', labelcolor = 'C0')
        ax_lower.tick_params(axis = 'y', labelcolor = 'C1')
        ax_lower1.tick_params(axis = 'y', labelcolor = 'C2')
        ax_lower2.tick_params(axis = 'y', labelcolor = 'k')

        # ensure third y-axis labels appear without overlapping
        ax_lower1.spines["right"].set_position(("axes", 1.01))
        ax_lower1.spines["right"].set_visible(True)
        ax_lower2.spines["right"].set_position(("axes", 1.08))
        ax_lower2.spines["right"].set_visible(True)

        # set axis labels
        ax_lower.set_xlabel("Blade row number")
        ax_upper.set_ylabel("Dimensionless tangential velocity", color = 'C0')
        ax_lower.set_ylabel("Stagnation Pressure (Pa)", color = 'C1')
        ax_lower1.set_ylabel("Stagnation Temperature (K)", color = 'C2')
        ax_lower2.set_ylabel("Tip radius (m)", color = 'k')

        # set grids on both plots
        ax_upper.grid()
        ax_lower.grid()

        plt.tight_layout()

    def determine_efficiency(self):
        """Determine key performance metrics for the engine system and individual stages."""
        # solve for pressure ratio and coefficient and temperature ratio
        self.p_r = self.blade_rows[-1].exit.p_0 / self.blade_rows[0].inlet.p_0
        self.C_p = (
            (self.blade_rows[-1].exit.p_0 - self.blade_rows[0].inlet.p_0)
            / (self.blade_rows[0].inlet.p_0 - self.blade_rows[0].inlet.p)
        )
        self.T_r = self.blade_rows[-1].exit.T_0 / self.blade_rows[0].inlet.T_0

        # solve for isentropic and polytropic efficiency
        T_0s = (
            self.blade_rows[0].inlet.T_0 * np.power(self.p_r, (gamma - 1) / gamma)
        )
        self.eta_s = (
            (T_0s - self.blade_rows[0].inlet.T_0)
            / (self.blade_rows[-1].exit.T_0 - self.blade_rows[0].inlet.T_0)
        )
        self.eta_p = (gamma - 1) * np.log(self.p_r) / (gamma * np.log(self.T_r))

        # for each stage, determine flow and stage loading coefficients and reaction
        for stage in self.stages:

            stage.flow_coefficient = (
                stage.blade_rows[0].inlet.axial_velocity / stage.blade_rows[0].blade_speed
            )
            stage.stage_loading_coefficient = (
                (stage.blade_rows[-1].exit.h_0 - stage.blade_rows[0].inlet.h_0)
                / stage.blade_rows[0].blade_speed**2
            )
            stage.reaction = (
                (stage.blade_rows[0].exit.h - stage.blade_rows[0].inlet.h)
                / (stage.blade_rows[-1].exit.h - stage.blade_rows[0].inlet.h)
            )

# 1.1 define Stage class

class Stage:
    def __init__(self):
        self.blade_rows = []

# 1.2 define Blade_row class

class Blade_row:
    """
    Stores a single row of blades and their associated parameters, used to investigate the flow
    across a Blade_row or a Stator. Stator is a special case of the Blade_row class where the
    blade velocity is zero. Every instance of the rotor class will contain and inlet and exit
    flow state where all of the flow properties are stored.
    
    Parameters
    ----------
    blade_speed (float) [m/s]: mean-line rotating blade speed (0 for a stator).
    metal_angle (float) [rad]: angle at which flow leaves the blade row.
    tip_radius (float) [m]
    hub_radius (float) [m]
    inlet (Flow_state class): container to store inlet fluid conditions.
    exit (Flow_state class): container to store exit fluid conditions.

    Public Methods
    --------------
    None
    """
    def __init__(self, blade_speed, metal_angle, tip_radius, hub_radius, Y_p, inlet=None, exit=None):
        """Create instance of the Blade_row class."""
        self.blade_speed = blade_speed
        self.metal_angle = metal_angle
        self.tip_radius = tip_radius
        self.hub_radius = hub_radius
        self.Y_p = Y_p
        self.inlet = inlet
        self.exit = exit

        # derived quantities
        self.area = np.pi * (self.tip_radius**2 - self.hub_radius**2)

        # categorise blade row as Rotor, Stator or Contra-Rotating
        if self.blade_speed > 0:

            self.row_type = f"{Colours.ORANGE}Rotor{Colours.END}"
            self.type = f"{Colours.ORANGE}R{Colours.END}"

        elif self.blade_speed == 0:

            self.row_type = f"{Colours.YELLOW}Stator{Colours.END}"
            self.type = f"{Colours.YELLOW}S{Colours.END}"

        else:

            self.row_type = f"{Colours.PURPLE}Contra-Rotating{Colours.END}"
            self.type = f"{Colours.PURPLE}CR{Colours.END}"

        # assign the default colour of black
        self.colour = 'k'

    def __str__(self):
        """Print string representation of blade row."""
        string = f"""{self.row_type}
Blade speed: {Colours.GREEN}{self.blade_speed:.3g}{Colours.END} m/s
Metal angle: {Colours.GREEN}{rad_to_deg(self.metal_angle):.3g}{Colours.END} °
Tip radius: {Colours.GREEN}{self.tip_radius:.3g}{Colours.END} m
Hub radius: {Colours.GREEN}{self.hub_radius:.3g}{Colours.END} m
Stagnation pressure loss coefficient: {Colours.GREEN}{self.Y_p:.3g}{Colours.END}
"""
        return string

# 1.2 define Flow_state class

class Flow_state:
    """
    Stores the flow properties at a given point in the system. Given two thermodynamic properties,
    the remaining properties can be determined via perfect gas relations.

    Parameters
    ----------
    pressure (Pa)
    temperature (K)
    specific enthalpy (J/kg)
    axial velocity (m/s)
    tangential velocity (m/s)
    Mach number (-)
    density (kg/m^3)

    Public methods
    --------------
    relative_quantities(self, blade_speed): asdf
    """
    def __init__(self, axial_velocity, tangential_velocity, p, T):
        """Create instance of the Flow_state class and store velocities and flow properties."""
        # input velocities
        self.axial_velocity = axial_velocity
        self.tangential_velocity = tangential_velocity
        self.velocity = np.sqrt(self.axial_velocity**2 + self.tangential_velocity**2)

        # static quantities
        self.p = p
        self.T = T
        self.h = c_p * self.T
        self.rho = self.p / (R * self.T)

        # Mach quantities
        self.a = np.sqrt(gamma * R * self.T)
        self.M = self.velocity / self.a

        # stagnation quantities
        self.p_0 = self.p / stagnation_pressure_ratio(self.M, gamma)
        self.T_0 = self.T / stagnation_temperature_ratio(self.M, gamma)
        self.h_0 = c_p * self.T_0

    def __str__(self):
        string = f"{self.__class__.__name__}:\n"
        for name, value in self.__dict__.items():
            string += f"  {name}: {value}\n"
        return string

    def relative_quantities(self, blade_speed):
        """Computes various quantities relative to a given blade speed."""
        # compute relative velocities and swirl angle
        self.relative_tangential_velocity = self.tangential_velocity - blade_speed
        self.relative_velocity = (
            np.sqrt(self.axial_velocity**2 + self.relative_tangential_velocity**2)
        )
        self.relative_swirl_angle = np.arctan(self.relative_tangential_velocity / self.axial_velocity)

        # compute relative stagnation properties and Mach number
        self.M_rel = self.relative_velocity / self.a
        self.p_0_rel = self.p / stagnation_pressure_ratio(self.M_rel, gamma)
        self.T_0_rel = self.T / stagnation_temperature_ratio(self.M_rel, gamma)

# 2.0 main function

def main():
    """Function to run on script execution."""
    # visualise colour options for convenience
    """colours = Colours()
    print(colours)"""
    print(f"{Colours.CYAN}Constructing engine for analysis...{Colours.END}")

    # determine from user how many stages to construct the engine with
    print(f"{Colours.RED}Please state the desired number of stages:{Colours.END}")
    while True:

        user_input = input()
        try:

            no_of_stages = int(user_input)
            break

        except ValueError:

            print(f"{Colours.RED}Error: Please provide a positive integer.{Colours.END}")

    print(f"{Colours.GREEN}{no_of_stages} stages selected!{Colours.END}")

    # construct engine class for the appropriate number of stages
    engine = Engine()
    for i in range(no_of_stages):

        r_t_in = Defaults.inlet_tip_radius
        r_t_out = Defaults.exit_tip_radius
        rotor = Blade_row(
            Defaults.blade_speed,
            Defaults.rotor_blade_angle,
            r_t_in * np.power(r_t_out / r_t_in, i / (no_of_stages - 0.5)),
            Defaults.hub_tip_ratio * r_t_in,
            Defaults.stagnation_pressure_loss_coefficient
        )
        stator = Blade_row(
            0,
            Defaults.stator_blade_angle,
            r_t_in * np.power(r_t_out / r_t_in, (i + 0.5) / (no_of_stages - 0.5)),
            Defaults.hub_tip_ratio * r_t_in,
            Defaults.stagnation_pressure_loss_coefficient
        )
        engine.blade_rows.extend([rotor, stator])

    # display default engine information to user
    print(engine)
    print(repr(engine))

    # determine from user whether or not to use the default blade row configurations
    print(
        f"{Colours.RED}Would you like to accept the default blade row configurations? [y / n]"
        f"{Colours.END}"
    )
    while True:

        user_input = input()
        if user_input == "y" or user_input == "n":
            
            break
        
        else:

            print(f"{Colours.RED}Error: Please respond with [y / n].{Colours.END}")

    # ask user which blade row they would like to make changes to
    if user_input == "n":

        while True:

            print(
                f"{Colours.RED}Which blade row would you like to edit? "
                f"[1 - {len(engine.blade_rows)}]{Colours.END}"
            )
            while True:

                user_input = input()
                try:

                    index = int(user_input)
                    break

                except ValueError:

                    print(
                        f"{Colours.RED}Error: Please provide a positive integer. "
                        f"[1 - {len(engine.blade_rows)}]{Colours.END}"
                    )

            print(f"\n[{index}] {engine.blade_rows[index - 1]}")

            # get new blade speed from user
            print(f"{Colours.RED}Please state the new blade speed (m/s):{Colours.END}")
            while True:

                user_input = input()
                if user_input == "":
                    
                    blade_speed = engine.blade_rows[index - 1].blade_speed
                    break

                try:
                
                    blade_speed = float(user_input)
                    break
                
                except ValueError:
                
                    print(f"{Colours.RED}Error: Please provide a valid number.{Colours.END}")

            print(f"{Colours.GREEN}Blade speed of {blade_speed:.3g} m/s selected!{Colours.END}")

            # get new metal angle from user
            print(f"{Colours.RED}Please state the new metal angle (°):{Colours.END}")
            while True:

                user_input = input()
                if user_input == "":

                    metal_angle = engine.blade_rows[index - 1].metal_angle
                    break

                try:

                    metal_angle = deg_to_rad(float(user_input))
                    break

                except ValueError:

                    print(f"{Colours.RED}Error: Please provide a valid number.{Colours.END}")

            print(f"{Colours.GREEN}Metal angle of {rad_to_deg(metal_angle):.3g} ° selected!{Colours.END}")

            # get new tip radius from user
            print(f"{Colours.RED}Please state the new tip radius (m):{Colours.END}")
            while True:

                user_input = input()
                if user_input == "":
  
                    tip_radius = engine.blade_rows[index - 1].tip_radius
                    break

                try:

                    tip_radius = float(user_input)
                    break

                except ValueError:

                    print(f"{Colours.RED}Error: Please provide a valid number.")

            print(f"{Colours.GREEN}Tip radius of {tip_radius:.3g} m selected!{Colours.END}")

            # get new hub radius from user
            print(f"{Colours.RED}Please state the new hub radius (m):{Colours.END}")
            while True:

                user_input = input()
                if user_input == "":

                    hub_radius = engine.blade_rows[index - 1].hub_radius
                    break

                try:

                    hub_radius = float(user_input)
                    break

                except ValueError:

                    print(f"{Colours.RED}Error: Please provide a valid number.{Colours.END}")

            print(f"{Colours.GREEN}Hub radius of {hub_radius:.3g} m selected!{Colours.END}")

            # get new stagnation pressure loss coefficient from user
            print(
                f"{Colours.RED}Please state the stagnation pressure loss coefficient:{Colours.END}"
            )
            while True:

                user_input = input()
                if user_input == "":

                    Y_p = engine.blade_rows[index - 1].Y_p
                    break

                try:

                    Y_p = float(user_input)
                    break

                except ValueError:

                    print(f"{Colours.RED}Error: Please provide a valid number.{Colours.END}")
            
            print(
                f"{Colours.GREEN}Stagnation pressure loss coefficient "
                f"of {Y_p:.3g} selected!{Colours.END}"
            )

            # update blade row and print new engine summary
            engine.blade_rows[index - 1] = Blade_row(
                blade_speed,
                metal_angle,
                tip_radius,
                hub_radius,
                Y_p
            )
            print(repr(engine))
            print(f"{Colours.RED}Would you like to edit another blade row? [y / n]{Colours.END}")
            while True:

                user_input = input()
                if user_input == "y" or user_input == "n":
                    
                    break
                
                else:

                    print(f"{Colours.RED}Error: Please respond with [y / n].{Colours.END}")

            # exit loop if user has no further blade rows they would like to edit
            if user_input == "n":

                break
    
    print(f"{Colours.GREEN}Blade configurations stored!{Colours.END}")

    # determine a new nominal blade speed
    for blade_row in engine.blade_rows:

        if "R" in blade_row.type:

            Defaults.blade_speed = blade_row.blade_speed
            break

    # determine flow state at engine inlet and store in first stag
    engine.blade_rows[0].inlet = Flow_state(
        Defaults.air_speed,
        Defaults.air_speed * np.tan(Defaults.inlet_swirl),
        p_atm, T_atm
    )

    # solve velocity triangles
    engine.calculate_mass_flow_rate()
    engine.solve_velocity_triangles()
    engine.assign_stages()
    engine.determine_efficiency()
    engine.visualise_velocity_triangles()

    # store nominal flow and stage loading coefficients for each stage
    nominal_flow_coefficients = [
        stage.flow_coefficient for stage in engine.stages
    ]
    nominal_stage_loading_coefficients = [
        stage.stage_loading_coefficient for stage in engine.stages
    ]

    # prepare to sweep over velocities and preallocate storage
    velocities = np.linspace(Defaults.air_speed / 2, 3 * Defaults.air_speed / 2, 5)
    n_stages = len(engine.stages)
    flow_coefficients = np.zeros((len(velocities), n_stages))
    stage_loading_coefficients = np.zeros((len(velocities), n_stages))

    # sweep over velocities and store flow and stage loading coefficients
    for i, v in enumerate(velocities):

        engine.blade_rows[0].inlet = Flow_state(
            v,
            v * np.tan(Defaults.inlet_swirl),
            p_atm, T_atm
        )
        engine.calculate_mass_flow_rate()
        engine.solve_velocity_triangles()
        engine.assign_stages()
        engine.determine_efficiency()

        for j, stage in enumerate(engine.stages):

            flow_coefficients[i, j] = stage.flow_coefficient
            stage_loading_coefficients[i, j] = stage.stage_loading_coefficient

    # plot compressor characteristic
    fig, ax = plt.subplots(figsize=(8, 4))
    for index in range(n_stages):

        ax.plot(
            flow_coefficients[:, index],
            stage_loading_coefficients[:, index],
            label = f"Stage {index + 1}",
            color = engine.stages[index].colour
        )
        
    for index in range(n_stages):

        ax.plot(
            nominal_flow_coefficients[index],
            nominal_stage_loading_coefficients[index],
            color = engine.stages[index].colour,
            linestyle = '',
            marker = '.', markersize = 8
        )

    ax.set_xlabel("Flow coefficient (ϕ)")
    ax.set_ylabel("Stage loading coefficient (ψ)")
    ax.set_title("Compressor stage characteristics")
    ax.grid()
    ax.legend()
    plt.show()

# run main() on running the script

if __name__ == "__main__":
    """Run main() on running the script."""
    main()