# 0.0 import modules

import numpy as np
import matplotlib.pyplot as plt

# 0.1 global variables

p_atm = 1e5                 # Pa
T_atm = 288                 # K
c_p = 1005                  # J/kgK
gamma = 1.4
R = c_p * (1 - 1 / gamma)   # J/kgK

# 0.2 define Defaults class

class Defaults:
    """Container for default values relating to the engine system."""

    rotor_blade_angle = -30 * np.pi / 180    # rad
    stator_blade_angle = 10 * np.pi / 180   # rad
    inlet_swirl = 0
    flow_coefficient = 0.5
    velocity = 40                           # m/s

    blade_speed = velocity / flow_coefficient

# 0.3 compressible flow lookup tables - or use perfect gas relations?

pass

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

    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __repr__(self):
        """Return string representation of all available colours."""
        string = f"""
        Available colours are as follows:
        {self.PURPLE}PURPLE{self.END}
        {self.BLUE}BLUE{self.END}
        {self.CYAN}CYAN{self.END}
        {self.GREEN}GREEN{self.END}
        {self.YELLOW}YELLOW{self.END}
        {self.RED}RED{self.END}
        {self.BOLD}BOLD{self.END}
        {self.UNDERLINE}UNDERLINE{self.END}
        """
        return string

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
    solve_flow(self, inlet_flow, blade_row): solves for the velocities at exit to a blade row
    """
    def __init__(self):
        """Create instance of the Engine class."""
        self.stages = []

    def solve_velocity_triangles(self):
        """Solves for axial and tangential velocity at inlet and exit to each blade row."""

        # iterate over all stages and solve for flow
        for index, stage in enumerate(self.stages):
            if stage.rotor.inlet == None:
                stage.rotor.inlet = self.stages[index - 1].stator.exit
            stage.rotor.exit = self.solve_flow(stage.rotor.inlet, stage.rotor)
            stage.stator.inlet = stage.rotor.exit
            stage.stator.exit = self.solve_flow(stage.stator.inlet, stage.stator)

    def solve_flow(self, inlet_flow, blade_row):
        """Solves for velocities at exit to a given blade row."""

        # determine tangential velocity in the relative frame and then convert to absolute
        relative_tangential_velocity = inlet_flow.axial_velocity * np.tan(blade_row.metal_angle)
        tangential_velocity = relative_tangential_velocity + blade_row.blade_speed
        velocity = np.sqrt(inlet_flow.axial_velocity**2 + tangential_velocity**2)

        # apply Euler's formula and solve for outlet temperature
        T_02 = (
            inlet_flow.T_0 + blade_row.blade_speed
            * (tangential_velocity - inlet_flow.tangential_velocity) / c_p
        )
        T = T_02 - 0.5 * velocity**2 / c_p

        # apply isentropic relation to solve for outlet pressure
        p_02 = inlet_flow.p_0 * np.power(T_02 / inlet_flow.T_0, gamma / (gamma - 1))
        p = p_02 / (1 + velocity**2 / (2 * R * T))

        # store results as instance of Flow_state class and return
        exit_flow = Flow_state(
            inlet_flow.axial_velocity,
            tangential_velocity,
            p, T
        )
        return exit_flow
    
    def visualise_velocity_triangles(self):
        """Function to plot the velocity triangles and pressure and temperature distributions."""
        fig, ax = plt.subplots()

        ax1 = ax.twinx()
        ax2 = ax.twinx()

        pressures = [p_atm]
        temperatures = [T_atm]

        for index, stage in enumerate(self.stages):
            print(f"Index: {index}")

            # display NACA aerofoils in position

            # construct rotation matrix and carry out matrix multiplication
            R = np.array([
                [np.cos(stage.rotor.metal_angle), -np.sin(stage.rotor.metal_angle)],
                [np.sin(stage.rotor.metal_angle),  np.cos(stage.rotor.metal_angle)]
            ])
            coords = data.copy()
            data_rot = coords @ R.T

            print(data_rot)

            ax.plot(data_rot[:, 0] + 2 * index + 1, data_rot[:, 1], color = 'C5')

            R = np.array([
                [np.cos(stage.stator.metal_angle), -np.sin(stage.stator.metal_angle)],
                [np.sin(stage.stator.metal_angle),  np.cos(stage.stator.metal_angle)]
            ])
            coords = data.copy()
            data_rot = coords @ R.T
            ax.plot(data_rot[:, 0] + 2 * index + 2, data_rot[:, 1], color = 'C5')

            # display velocity triangles for flow
            ax.annotate(
                "",
                xy=(
                    2 * index + 1 + stage.rotor.exit.axial_velocity / Defaults.velocity,
                    stage.rotor.exit.tangential_velocity / Defaults.velocity
                ),
                xytext=(2 * index + 1, 0),
                arrowprops=dict(arrowstyle = "->", color = 'C0')
            )
            ax.annotate(
                "",
                xy=(
                    2 * index + 2 + stage.stator.exit.axial_velocity / Defaults.velocity,
                    stage.stator.exit.tangential_velocity / Defaults.velocity
                ),
                xytext=(2 * index + 2, 0),
                arrowprops=dict(arrowstyle = "->", color = 'C0')
            )

            # display blade row speeds
            ax.annotate(
                "",
                xy = (
                    2 * index + 1,
                    stage.rotor.blade_speed / Defaults.velocity
                ),
                xytext = (2 * index + 1, 0),
                arrowprops = dict(arrowstyle = "->", color = 'C3')
            )
            ax.annotate(
                "",
                xy = (
                    2 * index + 2,
                    stage.stator.blade_speed / Defaults.velocity
                ),
                xytext = (2 * index + 2, 0),
                arrowprops = dict(arrowstyle = "->", color = 'C3')
            )

            # store pressure and temperature for plotting
            pressures.append(stage.rotor.exit.p)
            pressures.append(stage.stator.exit.p)
            temperatures.append(stage.rotor.exit.T)
            temperatures.append(stage.stator.exit.T)

        ax.plot([], [], color = 'C0', label = 'Absolute flow velocity')
        ax.plot([], [], color = 'C3', label = 'Blade speeds')

        ax1.plot(np.arange(2 * len(self.stages) + 1) + 0.5, pressures, color = 'C1', label = 'Pressure (Pa)')
        ax2.plot(np.arange(2 * len(self.stages) + 1) + 0.5, temperatures, color = 'C2', label = 'Temperature (K)')
        ax.set_xlim(0, 2 * len(self.stages) + 1)
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
        plt.show()

# 1.1 define Stage class

class Stage:
    """
    Used to store two consecutive blade rows as a single entity - usually a rotor and a stator.
    Allows for individual stage performance to be investigated separately. Can be subsituted for
    a single blade rows instead in the case of an odd number of blade rows (e.g. R-S-R).
    
    Parameters
    ----------
    asdf

    Public Methods
    --------------
    asdf
    """
    def __init__(self, rotor=None, stator=None):
        """Create instance of the Stage class."""
        self.rotor = rotor
        self.stator = stator


# 1.2 define Blade_row class

class Blade_row:
    """
    Stores a single row of blades and their associated parameters, used to investigate the flow
    across a Blade_row or a Stator. Stator is a special case of the Blade_row class where the blade
    velocity is zero. Every instance of the rotor class will contain and inlet and exit flow
    state where all of the flow properties are stored.
    
    Parameters
    ----------
    asdf

    Public Methods
    --------------
    asdf
    """
    def __init__(self, blade_speed, metal_angle, inlet=None, exit=None):
        """Create instance of the Blade_row class."""
        self.blade_speed = blade_speed
        self.metal_angle = metal_angle
        self.inlet = inlet
        self.exit = exit

# 1.3 define Flow_state class

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
    asdf
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

        # total quantities
        self.p_0 = self.p + 0.5 * self.rho * self.velocity**2
        self.T_0 = self.T + 0.5 * self.velocity**2 / c_p
        self.h_0 = c_p * (self.T_0 - T_atm)


# 2.0 main function

def main():
    """Function to run on script execution."""
    print(f"{Colours.CYAN}Running mean line analysis script...{Colours.END}")

    # determine from user whether or not to use the default blade row configurations
    print(f"{Colours.RED}Would you like to accept the default blade row configurations? [y / n]"
          f"{Colours.END}")
    while True:
        user_input = input()
        if user_input == "y" or user_input == "n":
            break
        else:
            print(f"{Colours.RED}Error: response not understood.{Colours.END}")
    if user_input == "y":
        print(f"{Colours.GREEN}Default configuration accepted!{Colours.END}")
        use_defaults()
    elif user_input == "n":
        custom_blades

def use_defaults():
    """User has selected to accept the default blade angles."""
    print(f"{Colours.RED}Please state the desired number of stages:{Colours.END}")
    while True:
        user_input = input()
        try:
            user_input = int(user_input)
            break
        except ValueError:
            print(f"{Colours.RED}Error: Please provide an integer.{Colours.END}")
    print(f"{Colours.GREEN}{user_input} stages selected!{Colours.END}")
    print(f"{Colours.CYAN}Analysing velocity triangles...{Colours.END}")

    # construct engine class for the appropriate number of stages
    engine = Engine()
    for i in range(user_input):
        rotor = Blade_row(Defaults.blade_speed, Defaults.rotor_blade_angle)
        stator = Blade_row(0, Defaults.stator_blade_angle)
        stage = Stage(rotor, stator)
        engine.stages.append(stage)

    # determine flow state at engine inlet and store in first stage
    flow_state = Flow_state(
        Defaults.velocity,
        Defaults.velocity * np.tan(Defaults.inlet_swirl),
        p_atm, T_atm
    )
    engine.stages[0].rotor.inlet = flow_state

    # solve velocity triangles
    engine.solve_velocity_triangles()

    # visualise results
    engine.visualise_velocity_triangles()


def custom_blades():
    """User has selected to specify blade angles individually."""
    pass
        
        

# run main() on running the script

if __name__ == "__main__":
    """Run main() on running the script."""
    main()