# 0.0 import modules

import numpy as np
import matplotlib.pyplot as plt
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
    rotor_blade_angle = -20 * np.pi / 180           # rad
    stator_blade_angle = 10 * np.pi / 180           # rad
    blade_speed = 100                               # m/s
    stagnation_pressure_loss_coefficient = 0.00001
    inlet_tip_radius = 1                            # m
    exit_tip_radius = 1                           # m
    hub_tip_ratio = 0.4

    # values used for determining engine inlet conditions
    inlet_swirl = 0
    air_speed = 40                                  # m/s

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
        gamma * M * np.power(1 + (gamma - 1) * M**2 / 2, -(gamma + 1) / 2 * (gamma - 1))
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
data *= 0.8
data[:, 0] -= 0.2

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
    
    def calculate_mass_flow_rate(self):
        """Calculates mass flow rate given the flow state at engine inlet."""
        inlet = self.blade_rows[0].inlet
        self.m_dot = inlet.rho * inlet.axial_velocity * self.blade_rows[0].area

    def solve_velocity_triangles(self):
        """Solves for axial and tangential velocity at inlet and exit to each blade row."""

        # iterate over all stages and solve for flow
        for index, blade_row in enumerate(self.blade_rows):

            if blade_row.inlet == None:

                blade_row.inlet = self.solve_area_change(*self.blade_rows[index - 1:index + 1])

            print(f"Inlet conditions: {blade_row.inlet}")

            blade_row.exit = self.solve_flow(blade_row)
            
            print(f"Exit conditions: {blade_row.exit}")

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
            self.m_dot * np.sqrt(c_p * T_0_rel)
            / (blade_row.area * np.cos(blade_row.inlet.relative_swirl_angle) * p_0_rel)
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

        # establish velocity bounds to ensure choking does not occur
        v_min = 1e-16
        v_max = np.sqrt((2.0 * c_p * T_0 - v_t**2) / (1.0 + 2.0 * (gamma - 1.0)))

        # establish LHS quantity and solve iteratively
        lhs = self.m_dot * R * T_0 / (blade_row2.area * p_0)
        def residual(vx):
            rhs = vx * np.power(1.0 - (vx**2 + v_t**2) / (2.0 * c_p * T_0), 1.0 / (gamma - 1.0))
            return rhs - lhs
        
        """xx = np.linspace(v_min, v_max, 20)
        yy = residual(xx)
        fig, ax = plt.subplots()
        ax.plot(xx, yy)
        ax.grid()
        plt.show()"""

        print("DEBUG area_change - inputs:")
        print(f" mdot = {self.m_dot}, A = {blade_row2.area}, p0 = {p_0}, T0 = {T_0}")
        print(f" v_t (exit swirl guess) = {v_t}")
        print(f" gamma = {gamma}, R = {R}, cp = {c_p}")
        print(" C =", lhs)

        # quick residual checks
        print(" residual(v_min) =", residual(v_min))
        print(" residual(v_max) =", residual(v_max))
        
        v_x = brentq(residual, v_min, v_max)
        Vr = np.sqrt(v_x**2 + v_t**2)
        p = p_0 * np.power(1.0 - Vr**2 / (2.0 * c_p * T_0), gamma / (gamma - 1.0))
        T = T_0 * (1.0 - Vr**2 / (2.0 * c_p * T_0))
    
        exit_flow = Flow_state(v_x, v_t, p, T)
        return exit_flow
    
    def visualise_velocity_triangles(self):
        """Function to plot the velocity triangles and pressure and temperature distributions."""
        # create matplotlib plot with multiple axes
        fig, ax = plt.subplots()
        ax1 = ax.twinx()
        ax2 = ax.twinx()

        # store pressure and temperature distributions and plot against blade row number
        pressures = [
            p for blade_row in self.blade_rows
            for p in (blade_row.inlet.p, blade_row.exit.p)
        ]
        temperatures = [
            T for blade_row in self.blade_rows
            for T in (blade_row.inlet.T, blade_row.exit.T)
        ]
        xx = np.arange(2 * len(self.blade_rows)) / 2 + 0.75
        ax1.plot(
            xx,
            pressures,
            color = 'C1',
            label = 'Pressure (Pa)',
            alpha = 0.5
        )
        ax2.plot(
            xx,
            temperatures,
            color = 'C2',
            label = 'Temperature (K)',
            alpha = 0.5
        )

        pressures = [p_atm]
        temperatures = [T_atm]

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
            ax.plot(data_rot[:, 0] + index + 1, data_rot[:, 1], color = 'C5')

            # display velocity triangles as arrows
            ax.annotate(
                "",
                xy=(
                    index + 1 + blade_row.exit.axial_velocity / Defaults.blade_speed,
                    blade_row.exit.tangential_velocity / Defaults.blade_speed
                ),
                xytext=(index + 1, 0),
                arrowprops = dict(
                    arrowstyle = "->",
                    color = 'C0',
                    shrinkA = 0,
                    shrinkB = 0,
                    lw = 2
                )
            )

            # display blade row speeds
            ax.annotate(
                "",
                xy = (
                    index + 1,
                    blade_row.blade_speed / Defaults.blade_speed
                ),
                xytext = (index + 1, 0),
                arrowprops = dict(
                    arrowstyle = "->",
                    color = 'C3',
                    shrinkA = 0,
                    shrinkB = 0,
                    lw = 2
                )
            )

            # store pressure and temperature for plotting
            pressures.append(blade_row.exit.p)
            temperatures.append(blade_row.exit.T)

        ax.plot([], [], color = 'C0', label = 'Absolute flow velocity')
        ax.plot([], [], color = 'C3', label = 'Blade speeds')
        ax.set_xlim(0, len(self.blade_rows) + 1)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')

        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax1.get_legend_handles_labels()
        lines_3, labels_3 = ax2.get_legend_handles_labels()

        ax.legend(
            lines_1 + lines_2 + lines_3,
            labels_1 + labels_2 + labels_3,
            loc = 'lower right'
        )

        ax.tick_params(axis = 'y', labelcolor = 'C0')
        ax1.tick_params(axis = 'y', labelcolor = 'C1')
        ax2.tick_params(axis = 'y', labelcolor = 'C2')

        # ensure third y-axis labels appear without overlapping
        ax2.spines["right"].set_position(("axes", 1.25))
        ax2.spines["right"].set_visible(True)
        ax.set_xlabel("Blade row number")
        ax.set_ylabel("Dimensionless tangential velocity", color = 'C0')
        ax1.set_ylabel("Pressure (Pa)", color = 'C1')
        ax2.set_ylabel("Temperature (K)", color = 'C2')
        ax.grid()
        plt.tight_layout()

    def determine_efficiency(self):
        """Determine key performance metrics for the engine system."""
        # solve for pressure ratio and coefficient and temperature ratio
        self.p_r = self.blade_rows[-1].exit.p_0 / self.blade_rows[0].inlet.p_0
        self.C_p = (
            (self.blade_rows[-1].exit.p_0 - self.blade_rows[0].inlet.p_0)
            / (0.5 * self.blade_rows[0].inlet.rho * Defaults.blade_speed**2)
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
        #self.eta_p = np.log(self.T_r) / ((gamma - 1) * np.log(self.p_r) / gamma)

        # print results to console
        print(f"Stagnation pressure ratio: {self.p_r:.3g}")
        print(f"Stagnation pressure coefficient: {self.C_p:.3g}")
        print(f"Stagnation temperature ratio: {self.T_r:.3g}")
        print(f"Isentropic efficiency: {self.eta_s:.3g}")
        print(f"Polytropic efficiency: {self.eta_p:.3g}")

# 1.1 define Blade_row class

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
    def __init__(self, blade_speed, metal_angle, tip_radius, hub_radius, inlet=None, exit=None):
        """Create instance of the Blade_row class."""
        self.blade_speed = blade_speed
        self.metal_angle = metal_angle
        self.tip_radius = tip_radius
        self.hub_radius = hub_radius
        self.inlet = inlet
        self.exit = exit
        self.Y_p = Defaults.stagnation_pressure_loss_coefficient

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

    def __str__(self):
        """Print string representation of blade row."""
        string = f"""{self.row_type}
Blade speed: {Colours.GREEN}{self.blade_speed:.3g}{Colours.END} m/s
Metal angle: {Colours.GREEN}{rad_to_deg(self.metal_angle):.3g}{Colours.END} °
Tip radius: {Colours.GREEN}{self.tip_radius:.3g}{Colours.END} m
Hub radius: {Colours.GREEN}{self.hub_radius:.3g}{Colours.END} m
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
        # input quantities
        self.axial_velocity = axial_velocity
        self.tangential_velocity = tangential_velocity
        self.p = p
        self.T = T
        
        # derived quantities
        self.velocity = np.sqrt(self.axial_velocity**2 + self.tangential_velocity**2)
        self.h = c_p * (self.T - T_atm)
        self.rho = self.p / (R * self.T)

        # stagnation quantities
        self.p_0 = self.p + 0.5 * self.rho * self.velocity**2
        self.T_0 = self.T + 0.5 * self.velocity**2 / c_p
        self.h_0 = c_p * (self.T_0 - T_atm)

        # Mach quantities
        self.a = np.sqrt(gamma * R * self.T)
        self.M = self.velocity / self.a

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
        self.p_0_rel = self.p + 0.5 * self.rho * self.relative_velocity**2
        self.T_0_rel = self.T + self.relative_velocity**2 / (2 * c_p)
        self.M_rel = self.relative_velocity / self.a

    def static_quantities(self, blade_speed, p_0_rel):
        """Computes static quantities given a blade speed and a relative stagnation pressure."""
        pass



# 2.0 main function

def main():
    """Function to run on script execution."""
    # visualise colour options for convenience
    colours = Colours()
    print(colours)
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
            r_t_in * np.power(r_t_out / r_t_in, i / no_of_stages),
            Defaults.hub_tip_ratio * r_t_in
        )
        stator = Blade_row(
            0,
            Defaults.stator_blade_angle,
            r_t_in *np.power(r_t_out / r_t_in, (i + 0.5) / no_of_stages),
            Defaults.hub_tip_ratio * r_t_in
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
                try:
                
                    blade_speed = float(user_input)
                    break
                
                except ValueError:
                
                    print(f"{Colours.RED}Error: Please provide a valid number. ")

            print(f"{Colours.GREEN}Blade speed of {blade_speed:.3g} m/s selected!{Colours.END}")

            # get new metal angle from user
            print(f"{Colours.RED}Please state the new metal angle (°):{Colours.END}")
            while True:

                user_input = input()
                try:

                    metal_angle = float(user_input)
                    break

                except ValueError:

                    print(f"{Colours.RED}Error: Please provide a valid number.")
            
            print(f"{Colours.GREEN}Metal angle of {metal_angle:.3g} ° selected!{Colours.END}")

            # get new tip radius from user
            print(f"{Colours.RED}Please state the new tip radius (m):{Colours.END}")
            while True:

                user_input = input()
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
                try:

                    hub_radius = float(user_input)
                    break

                except ValueError:

                    print(f"{Colours.RED}Error: Please provide a valid number.")
            
            print(f"{Colours.GREEN}Hub radius of {hub_radius:.3g} m selected!{Colours.END}")

            # update blade row and print new engine summary
            engine.blade_rows[index - 1] = Blade_row(
                blade_speed,
                deg_to_rad(metal_angle),
                tip_radius,
                hub_radius
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
            print(
                f"{Colours.GREEN}Nominal blade speed of {Defaults.blade_speed} "
                f"m/s determined!{Colours.END}"
            )
            break

    # determine flow state at engine inlet and store in first stage
    print(f"{Colours.CYAN}Determining flow state at engine inlet...{Colours.END}")
    flow_state = Flow_state(
        Defaults.air_speed,
        Defaults.air_speed * np.tan(Defaults.inlet_swirl),
        p_atm, T_atm
    )
    engine.blade_rows[0].inlet = flow_state
    engine.calculate_mass_flow_rate()
    print(f"{Colours.GREEN}Success!\n{Colours.END}")

    # solve velocity triangles
    print(f"{Colours.CYAN}Analysing velocity triangles...{Colours.END}")
    engine.solve_velocity_triangles()
    print(f"{Colours.GREEN}Success!\n{Colours.END}")

    # visualise results
    print(f"{Colours.CYAN}Visualising velocity triangles...{Colours.END}")
    engine.determine_efficiency()
    engine.visualise_velocity_triangles()
    print(f"{Colours.GREEN}Success!\n{Colours.END}")
    plt.show()


def custom_blades():
    """User has selected to specify blade angles individually."""
    pass
        
        

# run main() on running the script

if __name__ == "__main__":
    """Run main() on running the script."""
    main()