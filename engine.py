# import modules

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.patches import Polygon
import numpy as np
import copy
from scipy.interpolate import make_interp_spline

from stage import Stage
from nozzle import Nozzle
import utils
from flow_state import Flow_state

# 1.0 define Engine class

class Engine:
    """
    Used to store multiple (if applicable) stages and determine the overall engine performance.
    
    Parameters
    ----------
    no_of_stages : int
        Number of rotor-stator compressor stages to add to the engine.
    M_1 : float
        Inlet Mach number used as input to solve through the engine.
    scenario : class
        Instance of the Flight_scenario class for which this engine is designed.
    n : float
        The vortex exponent.
    N : float
        The number of annular streamtubes through which to solve the flow.
    """
    def __init__(self, no_of_stages, M_1, scenario, n, N):
        """Create instance of the Engine class."""
        # store input variables
        self.no_of_stages = no_of_stages
        self.M_1 = M_1
        self.M_flight = scenario.M
        self.C_th_design = scenario.C_th
        self.n = n
        self.N = N

        # create the appropriate number of empty stages and blade rows
        self.stages = []
        self.blade_rows = []
        for i in range(self.no_of_stages):

            self.stages.append(Stage(self.n, self.N, i))
            self.blade_rows.extend(self.stages[-1].blade_rows)

        # create nozzle
        self.nozzle = Nozzle(2 * self.no_of_stages - 0.5, 2 * self.no_of_stages)

        # run engine design subroutine
        self.design()

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

    def __repr__(self):
        """Print a detailed summary of the information contained in the class."""
        string = f"""
{utils.Colours.UNDERLINE}Engine Details{utils.Colours.END}\n
"""
        for index, blade_row in enumerate(self.blade_rows):

            string += f"[{index}] {blade_row}"

        string += f"{self.nozzle}"

        return string

    def design(self):
        """Determines appropriate values for blade metal angles for the given requirements."""
        # iterate over all stages
        for index, stage in enumerate(self.stages):

            # store rotor and stator as variables for convenience
            rotor = stage.blade_rows[0]
            stator = stage.blade_rows[1]

            # handle first stage
            if index == 0:

                # set first stage to default inlet conditions
                rotor.set_inlet_conditions(self.M_1, utils.Defaults.inlet_swirl)

            # handle final stage
            if index == len(self.stages) - 1:

                pass

            # handle all other stages
            else:

                # set stage inlet to previous stage exit conditions
                rotor.inlet = self.stages[index - 1].blade_rows[1].exit

            # define blade geometry for that stage
            rotor.rotor_design(stage.phi, stage.psi)
            stator.inlet = copy.deepcopy(rotor.exit)
            stator.stator_design(
                stage.reaction,
                stage.blade_rows[0].inlet[index].flow_state.T,
                stage.blade_rows[0].exit[index].flow_state.T,
                index == len(self.stages) - 1
            )

        # set nozzle inlet conditions
        self.nozzle.inlet = copy.deepcopy(self.blade_rows[-1].exit)

        # design nozzle to match atmospheric pressure in jet periphery
        p_atm = utils.stagnation_pressure_ratio(self.M_flight)
        self.nozzle.nozzle_design(p_atm)
        self.performance_metrics()

    def analyse(self):
        """Analyses the entire engine system."""
        # analyse all blade_rows
        for index, blade_row in enumerate(self.blade_rows):

            # skip for first blade row
            if index > 0:

                # create a nozzle and solve to account for inter-blade row area change
                nozzle = Nozzle(
                    #self.blade_rows[index - 1].casing_area_ratio,
                    self.blade_rows[index - 1].exit
                )
                nozzle.solve_nozzle()
                blade_row.inlet = nozzle.exit

            # calculate conditions at blade row exit
            blade_row.solve_blade_row()

        # analyse nozzle
        self.nozzle.inlet = self.blade_rows[-1].exit
        self.nozzle.solve_nozzle()

        # calculate engine and individual stage efficiencies
        self.determine_efficiency()

        # assign colours to stages and blade rows
        self.assign_colours()

    def collect_flow_states(self):
        """Recursively collect all instances of a given class stored within the engine."""
        # helper function
        def _collect(obj, seen):

            # prevent infinite recursion from cyclic references
            if id(obj) in seen:

                return []

            # if unseen, add to set
            seen.add(id(obj))
            found = []

            # direct match
            if isinstance(obj, Flow_state):

                found.append(obj)
                return found

            # recurse into class attributes
            if hasattr(obj, "__dict__"):

                for value in vars(obj).values():

                    found.extend(_collect(value, seen))

            # recurse into common containers
            elif isinstance(obj, list):

                for item in obj:

                    found.extend(_collect(item, seen))

            return found

        self.flow_states = _collect(self, seen=set())

    def assign_colours(self):
        """"Assigns each stage a unique colour."""
        # iterate over all stages
        for index, stage in enumerate(self.stages):

            # handle case where engine has only one stage
            if len(self.stages) == 1:
                
                stage.colour = utils.Colours.rainbow_colour_map(0)

            # for multi-stage engines
            else:

                stage.colour = utils.Colours.rainbow_colour_map(index / (len(self.stages) - 1))

            # assign stage number and colour to blade row
            for blade_row in stage.blade_rows:

                blade_row.colour = stage.colour
    
    def visualise_velocity_triangles(self):
        """Function to plot the velocity triangles and pressure and temperature distributions."""
        # create plot for displaying velocity triangles
        fig, ax = plt.subplots(figsize = (10, 6))

        # set title
        ax.title.set_text(
            f"Flight Mach number: {self.M_flight:.3g}\n"
            f"Thrust coefficient: {self.C_th:.3g}\n"
            f"Jet velocity ratio: {self.jet_velocity_ratio:.3g}\n"
            f"Nozzle area ratio: {self.nozzle.area_ratio:.3g}"
        )

        spanwise_positions = [0]

        if self.N > 1:

            spanwise_positions.append(-1)

        if self.N > 2:

            spanwise_positions.append(int(np.floor(self.N / 2)))

        # iterate over all blade rows
        for index, blade_row in enumerate(self.blade_rows):

            # iterate over all inlet and exit streamtubes chosen for plotting
            for j, k in enumerate(spanwise_positions):

                # generate blade shape data in blade row
                blade_row.draw_blades()

                # draw blade row
                ax.plot(blade_row.xx[k] + index, blade_row.yy[k] + j, color = blade_row.colour)

        # iterate over all blade rows again
        for index, blade_row in enumerate(self.blade_rows):

            # iterate over all inlet and exit streamtubes chosen for plotting
            for j, k in enumerate(spanwise_positions):

                # plot blade row at chosen spanwise positions
                self.plot_blade_row(blade_row, ax, index, j, k)

        ax.set_aspect('equal')
        return

        # iterate over all blade rows again
        """for index, blade_row in enumerate(self.blade_rows):

            # iterate over all inlet and exit streamtubes:
            for j, (inlet, exit) in enumerate(zip(blade_row.inlet, blade_row.exit)):

                x_te, y_te = blade_row.xx[j][0], blade_row.yy[j][0]
                if "Rotor" in blade_row.label:
                        
                    # display relative velocity vector at blade row inlet
                    ax.annotate(
                        "",
                        xy = (
                            index + inlet.flow_state.M_rel * np.cos(inlet.flow_state.beta),
                            j + inlet.flow_state.M_rel * np.sin(inlet.flow_state.beta)
                        ),
                        xytext = (index, j),
                        arrowprops = dict(
                            arrowstyle = "->", color = 'C4',
                            shrinkA = 0, shrinkB = 0, lw = 1.5
                        )
                    )

                    # display relative velocity vector at blade row exit
                    ax.annotate(
                        "",
                        xy = (
                            x_te + index + exit.flow_state.M_rel * np.cos(exit.flow_state.beta),
                            y_te + j + exit.flow_state.M_rel * np.sin(exit.flow_state.beta)
                        ),
                        xytext = (x_te + index, y_te + j),
                        arrowprops = dict(
                            arrowstyle = "->", color = 'C4',
                            shrinkA = 0, shrinkB = 0, lw = 1.5
                        )
                    )

                # display absolute velocity vector at blade row inlet
                ax.annotate(
                    "",
                    xy = (
                        index + inlet.flow_state.M * np.cos(inlet.flow_state.alpha),
                        j + inlet.flow_state.M * np.sin(inlet.flow_state.alpha)
                    ),
                    xytext = (index, j),
                    arrowprops = dict(
                        arrowstyle = "->", color = 'C0',
                        shrinkA = 0, shrinkB = 0, lw = 1.5
                    )
                )

                # display absolute velocity vector at blade row exit
                ax.annotate(
                    "",
                    xy = (
                        x_te + index + exit.flow_state.M * np.cos(exit.flow_state.alpha),
                        y_te + j + exit.flow_state.M * np.sin(exit.flow_state.alpha)
                    ),
                    xytext = (x_te + index, y_te + j),
                    arrowprops = dict(
                        arrowstyle = "->", color = 'C0',
                        shrinkA = 0, shrinkB = 0, lw = 1.5
                    )
                )

                if "Rotor" in blade_row.label:

                    # display blade row speed vector at blade row inlet
                    ax.annotate(
                        "",
                        xy = (
                            index,
                            j + inlet.M_blade
                        ),
                        xytext = (index, j),
                        arrowprops = dict(
                            arrowstyle = "->", color = 'C3',
                            shrinkA = 0, shrinkB = 0, lw = 1.5
                        )
                    )

                    # display blade row speed vector at blade row exit
                    ax.annotate(
                        "",
                        xy = (
                            x_te + index,
                            y_te + j + exit.M_blade
                        ),
                        xytext = (x_te + index, y_te + j),
                        arrowprops = dict(
                            arrowstyle = "->", color = 'C3',
                            shrinkA = 0, shrinkB = 0, lw = 1.5
                        )
                    )"""

        # old code
        fig, ax_upper = plt.subplots(figsize = (10, 6))
        fig, ax_lower = plt.subplots(figsize = (10, 6))
        ax_lower1 = ax_lower.twinx()
        ax_lower2 = ax_lower.twinx()

        # set title and subtitle
        ax_upper.title.set_text(
            f"Flight Mach number: {self.M_flight:.3g}\n"
            f"Thrust coefficient: {self.C_th:.3g}\n"
            f"Jet velocity ratio: {self.jet_velocity_ratio:.3g}\n"
            f"Nozzle area ratio: {self.nozzle.area_ratio:.3g}"
        )
        ax_upper.title.set_fontsize(10)
        ax_lower.title.set_text(
            f"Stagnation pressure ratio = {self.overall_p_0:.3g}\n"
            f"Stagnation temperature ratio = {self.overall_T_0:.3g}\n"
            f"Isentropic efficiency = {self.eta_s:.3g}\n"
            f"Polytropic efficiency = {self.eta_p:.3g}"
        )
        ax_lower.title.set_fontsize(10)

        # plot quantities against blade row number
        xx = np.arange(len(self.flow_states)) / 2 + 0.75
        alpha = 0.5
        ax_lower.plot(
            xx, [flow_state.p_0 * utils.stagnation_pressure_ratio(flow_state.M)
                 for flow_state in self.flow_states],
            color = 'C1', alpha = alpha
        )
        ax_lower1.plot(
            xx, [flow_state.T_0 * utils.stagnation_temperature_ratio(flow_state.M)
                 for flow_state in self.flow_states],
            color = 'C2', alpha = alpha
        )
        ax_lower.plot(
            xx, [flow_state.p_0 for flow_state in self.flow_states],
            color = 'C1', label = 'Pressure'
        )
        ax_lower1.plot(
            xx, [flow_state.T_0 for flow_state in self.flow_states],
            color = 'C2', label = 'Temperature'
        )
        ax_lower2.plot(
            xx, [flow_state.M for flow_state in self.flow_states],
            color = 'k', label = 'Mach number'
        )

        def display_blade_row(blade_row, index):
            """Helper function to plot a given blade row and corresponding velocity triangle."""
            # construct camber line as a parabolic profile
            b = np.tan(blade_row.inlet_blade_angle)
            a = (np.tan(blade_row.exit_blade_angle) - b) / 2
            def camber(x):
                """Return the y-component of the parabolic camber line given the x-value."""
                return a * x**2 + b * x

            # determine points on the camberline to compute the aerofoil shape from
            xx_0 = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, 100)))
            yy_0 = [camber(x) for x in xx_0]
            dy_0 = np.gradient(yy_0, xx_0)

            # determine cumulative length of chord line
            ll_0 = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(xx_0)**2 + np.diff(yy_0)**2))])
            camber_length = ll_0[-1]
            ll_0 = ll_0 / camber_length

            # normalise camber line to unit camber length
            xx_0 = xx_0 / camber_length
            yy_0 = yy_0 / camber_length

            # get the upper surface from the raw imported aerofoil data
            zz_0 = np.transpose([x for x in utils.data if x[1] >= 0])
            zz_0 = zz_0[:, zz_0[0].argsort()]

            # initialise empty arrays for the upper and lower surfaces
            xx_upper = np.zeros(xx_0.shape, dtype = "float")
            xx_lower = np.zeros(xx_0.shape, dtype = "float")
            yy_upper = np.zeros(xx_0.shape, dtype = "float")
            yy_lower = np.zeros(xx_0.shape, dtype = "float")

            # iterate over each datapoint
            for i, (x, y, dy_dx, l) in enumerate(zip(xx_0, yy_0, dy_0, ll_0)):

                # add thickness to the camberline
                norm = np.sqrt(1 + dy_dx**2)
                xx_upper[i] = x - np.interp(l, *zz_0) * dy_dx / norm
                xx_lower[i] = x + np.interp(l, *zz_0) * dy_dx / norm
                yy_upper[i] = y + np.interp(l, *zz_0) / norm
                yy_lower[i] = y - np.interp(l, *zz_0) / norm

            # get trailing edge coordinates
            x_te = xx_upper[-1]
            y_te = yy_upper[-1]

            # reverse upper surface
            xx_upper = xx_upper[::-1]
            yy_upper = yy_upper[::-1]

            # combine upper and lower surfaces
            xx = np.concatenate([xx_upper, xx_lower])
            yy = np.concatenate([yy_upper, yy_lower])

            # display aerofoil
            ax_upper.plot(xx + index, yy, color = blade_row.colour)

            # debugging
            """fig, ax = plt.subplots()
            ax.plot(xx, yy)
            ax.plot(xx_0, yy_0)
            plt.show()"""

            # display relative velocity vector at blade row inlet
            ax_upper.annotate(
                "",
                xy = (
                    index + blade_row.inlet.M * np.cos(blade_row.inlet.alpha),
                    blade_row.inlet.M * np.sin(blade_row.inlet.alpha)
                    - blade_row.inlet_blade_Mach_number
                ),
                xytext = (index, 0),
                arrowprops = dict(
                    arrowstyle = "->", color = 'C4',
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )

            # display relative velocity vector at blade row exit
            ax_upper.annotate(
                "",
                xy = (
                    x_te + index + blade_row.exit.M * np.cos(blade_row.exit.alpha),
                    y_te + blade_row.exit.M * np.sin(blade_row.exit.alpha) - blade_row.blade_Mach_number
                ),
                xytext = (x_te + index, y_te),
                arrowprops = dict(
                    arrowstyle = "->", color = 'C4',
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )

            # display absolute velocity vector at blade row inlet
            ax_upper.annotate(
                "",
                xy = (
                    index + blade_row.inlet.M * np.cos(blade_row.inlet.alpha),
                    blade_row.inlet.M * np.sin(blade_row.inlet.alpha)
                ),
                xytext = (index, 0),
                arrowprops = dict(
                    arrowstyle = "->", color = 'C0',
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )

            # display absolute velocity vector at blade row exit
            ax_upper.annotate(
                "",
                xy = (
                    x_te + index + blade_row.exit.M * np.cos(blade_row.exit.alpha),
                    y_te + blade_row.exit.M * np.sin(blade_row.exit.alpha)
                ),
                xytext = (x_te + index, y_te),
                arrowprops = dict(
                    arrowstyle = "->", color = 'C0',
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )

            # display blade row speed vector at blade row inlet
            ax_upper.annotate(
                "",
                xy = (
                    index,
                    blade_row.inlet_blade_Mach_number
                ),
                xytext = (index, 0),
                arrowprops = dict(
                    arrowstyle = "->", color = 'C3',
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )

            # display blade row speed vector at blade row exit
            ax_upper.annotate(
                "",
                xy = (
                    x_te + index,
                    y_te + blade_row.blade_Mach_number
                ),
                xytext = (x_te + index, y_te),
                arrowprops = dict(
                    arrowstyle = "->", color = 'C3',
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )

        # display all blade rows
        for index, blade_row in enumerate(self.blade_rows):

            display_blade_row(blade_row, index + 1)

        # compute cumulative area ratios for plotting annulus area change
        area_ratios = [blade_row.casing_area_ratio for blade_row in self.blade_rows]
        area_ratios.append(self.nozzle.area_ratio)
        areas = np.cumprod(area_ratios)

        ax_upper.plot(np.arange(len(areas)) + 1, areas, color = 'k')
        ax_upper.plot(np.arange(len(areas)) + 1, -np.array(areas), color = 'k')

        # label velocity arrows in legend
        ax_upper.plot([], [], color = 'C0', label = 'Absolute Mach number')
        ax_upper.plot([], [], color = 'C4', label = 'Relative Mach number')
        ax_upper.plot([], [], color = 'C3', label = 'Blade Mach number')

        # set axis limits and aspect ratio
        ax_upper.set_aspect('equal', adjustable='datalim')

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
        )

        # colour plot ticks
        ax_upper.tick_params(axis = 'y', labelcolor = 'C0')
        ax_lower.tick_params(axis = 'y', labelcolor = 'C1')
        ax_lower1.tick_params(axis = 'y', labelcolor = 'C2')
        ax_lower2.tick_params(axis = 'y', labelcolor = 'k')

        # ensure third y-axis labels appear without overlapping
        ax_lower1.spines["right"].set_position(("axes", 1.02))
        ax_lower1.spines["right"].set_visible(True)
        ax_lower2.spines["right"].set_position(("axes", 1.15))
        ax_lower2.spines["right"].set_visible(True)

        # set axis labels
        ax_upper.set_xlabel("Blade row number")
        ax_upper.set_ylabel("Mach number", color = 'C0')
        
        ax_lower.set_xlabel("Blade row number")
        ax_lower.set_ylabel("Pressure ratio", color = 'C1')
        ax_lower1.set_ylabel("Temperature ratio", color = 'C2')
        ax_lower2.set_ylabel("Mach number", color = 'k')

        # set grids on both plots
        ax_upper.grid()
        ax_lower.grid()

        plt.tight_layout()

    def plot_blade_row(self, blade_row, ax, index, j, k):
        """Plots a blade row onto a given axes at a specified spanwise position."""
        # store inlet and exit for convenience
        inlet = blade_row.inlet[k]
        exit = blade_row.exit[k]

        # get trailing edge coordinates
        x_te, y_te = blade_row.xx[k][0], blade_row.yy[k][0]

        # only plot relative quantities if blade is a rotor
        if "Rotor" in blade_row.label:
                
            # display relative velocity vector at blade row inlet
            ax.annotate(
                "",
                xy = (
                    index + inlet.flow_state.M_rel * np.cos(inlet.flow_state.beta),
                    j + inlet.flow_state.M_rel * np.sin(inlet.flow_state.beta)
                ),
                xytext = (index, j),
                arrowprops = dict(
                    arrowstyle = "->", color = 'C4',
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )

            # display relative velocity vector at blade row exit
            ax.annotate(
                "",
                xy = (
                    x_te + index + exit.flow_state.M_rel * np.cos(exit.flow_state.beta),
                    y_te + j + exit.flow_state.M_rel * np.sin(exit.flow_state.beta)
                ),
                xytext = (x_te + index, y_te + j),
                arrowprops = dict(
                    arrowstyle = "->", color = 'C4',
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )

        # display absolute velocity vector at blade row inlet
        ax.annotate(
            "",
            xy = (
                index + inlet.flow_state.M * np.cos(inlet.flow_state.alpha),
                j + inlet.flow_state.M * np.sin(inlet.flow_state.alpha)
            ),
            xytext = (index, j),
            arrowprops = dict(
                arrowstyle = "->", color = 'C0',
                shrinkA = 0, shrinkB = 0, lw = 1.5
            )
        )

        # display absolute velocity vector at blade row exit
        ax.annotate(
            "",
            xy = (
                x_te + index + exit.flow_state.M * np.cos(exit.flow_state.alpha),
                y_te + j + exit.flow_state.M * np.sin(exit.flow_state.alpha)
            ),
            xytext = (x_te + index, y_te + j),
            arrowprops = dict(
                arrowstyle = "->", color = 'C0',
                shrinkA = 0, shrinkB = 0, lw = 1.5
            )
        )

        if "Rotor" in blade_row.label:

            # display blade row speed vector at blade row inlet
            ax.annotate(
                "",
                xy = (
                    index,
                    j + inlet.M_blade
                ),
                xytext = (index, j),
                arrowprops = dict(
                    arrowstyle = "->", color = 'C3',
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )

            # display blade row speed vector at blade row exit
            ax.annotate(
                "",
                xy = (
                    x_te + index,
                    y_te + j + exit.M_blade
                ),
                xytext = (x_te + index, y_te + j),
                arrowprops = dict(
                    arrowstyle = "->", color = 'C3',
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )

    def performance_metrics(self):
        """Determine key performance metrics for the engine system."""
        # find thrust coefficient
        self.C_th = (
            utils.mass_flow_function(self.M_1) * np.sqrt(utils.gamma - 1) * (
                np.sum([
                    exit.flow_state.M * np.cos(exit.flow_state.alpha)
                    * np.sqrt(exit.flow_state.T)
                    - self.M_flight * np.sqrt(utils.stagnation_temperature_ratio(self.M_flight))
                    for exit in self.nozzle.exit
                ])
            )
        )

        # find propulsive efficiency
        self.eta_prop = (
            2 * self.C_th / utils.mass_flow_function(self.M_1) * np.sqrt(
                utils.stagnation_temperature_ratio(self.M_flight) / (utils.gamma - 1)
            ) / np.sum([
                exit.flow_state.M**2 * exit.flow_state.T
                - self.M_flight**2 * utils.stagnation_temperature_ratio(self.M_flight)
                for exit in self.nozzle.exit
            ])
        )

        # find nozzle efficiency
        self.eta_nozz = (
            (utils.gamma - 1) / 2 * np.sum([
                (
                    exit.flow_state.M**2 * exit.flow_state.T
                    - self.M_flight**2 * utils.stagnation_temperature_ratio(self.M_flight)
                ) / (
                    np.power(exit.flow_state.p_0, 1 - 1 / utils.gamma) - 1
                )
                for exit in self.nozzle.exit
            ])
        )

        # find compressor efficiency
        self.eta_comp = (
            np.sum([
                (np.power(exit.flow_state.p_0, 1 - 1 / utils.gamma) - 1)
                / (exit.flow_state.T_0 - 1)
                for exit in self.nozzle.exit
            ])
        )

        self.jet_velocity_ratio = 1

    def plot_contours(self):
        """Creates a plot of a section view of the engine with contours of a specified quantity."""
        # plotting parameters
        alpha = 0.5
        rotor_tip_span = 0.98

        # create a plot
        fig, ax = plt.subplots(figsize = (10, 6))

        # initialise fine array of x-values
        xx = np.linspace(0, len(self.blade_rows) - 0.5, 200)
        yy = np.linspace(xx[-1], xx[-1] + 0.5, 50)

        # draw centreline
        ax.plot(np.linspace(xx[0], yy[-1], 100), np.full(100, 0), linestyle = '--', color = 'k')

        # create spline for upper boundary of outer streamtube
        spline = make_interp_spline(
            [-1e-3] + [
                x for blade_row in self.blade_rows 
                for x in (blade_row.x_inlet, blade_row.x_exit)
            ] + [xx[-1] + 1e-3],
            [self.blade_rows[0].inlet[-1].r + self.blade_rows[0].inlet[-1].dr] + [
                r for blade_row in self.blade_rows for r in (
                    blade_row.inlet[-1].r + blade_row.inlet[-1].dr,
                    blade_row.exit[-1].r + blade_row.exit[-1].dr
                )
            ] + [self.blade_rows[-1].exit[-1].r + self.blade_rows[-1].exit[-1].dr],
            k = min(3, len(self.blade_rows))
        )

        # iterate over all stages
        for stage in self.stages:

            # store rotor and stator locally for convenience
            rotor = stage.blade_rows[0]
            stator = stage.blade_rows[1]

            # determine array of rotor x-values to plot
            x = np.linspace(
                rotor.x_inlet + (rotor.x_exit - rotor.x_inlet) * 0.05,
                rotor.x_exit - (rotor.x_exit - rotor.x_inlet) * 0.05, 100
            )

            # create array of rotor vertex positions
            vertices = np.column_stack([
                np.concatenate([x, x[::-1]]),
                np.concatenate([rotor_tip_span * spline(x), np.full_like(x, rotor.r_hub)])
            ])

            # create rotor polygon and add
            poly = Polygon(
                vertices, closed = True,
                facecolor = (0.8392, 0.1529, 0.1569, alpha),
                edgecolor = (0.8392, 0.1529, 0.1569),
                linewidth = 2
            )
            ax.add_patch(poly)

            # draw x-shape over rotor
            ax.plot([x[0], x[-1]], [rotor.r_hub, rotor_tip_span * spline(x[-1])], color = 'C3')
            ax.plot([x[0], x[-1]], [rotor_tip_span * spline(x[0]), rotor.r_hub], color = 'C3')

            # determine array of stator x-values to plot
            x = np.linspace(
                stator.x_inlet + (stator.x_exit - stator.x_inlet) * 0.05,
                stator.x_exit - (stator.x_exit - stator.x_inlet) * 0.05, 100
            )

            # create array of stator vertex positions
            vertices = np.column_stack([
                np.concatenate([x, x[::-1]]),
                np.concatenate([spline(x), np.full_like(x, stator.r_hub)])
            ])

            # create stator polygon and add
            poly = Polygon(
                vertices, closed = True,
                facecolor = (0.1216, 0.4667, 0.7059, alpha),
                edgecolor = (0.1216, 0.4667, 0.7059),
                linewidth = 2
            )
            ax.add_patch(poly)

            # draw x-shape over stator
            ax.plot([x[0], x[-1]], [stator.r_hub, spline(x[-1])], color = 'C0')
            ax.plot([x[0], x[-1]], [spline(x[0]), stator.r_hub], color = 'C0')

            # annotate rotor rows
            ax.annotate(
                "ROTOR",
                xy = (0, 0),
                xytext = ((rotor.x_inlet + rotor.x_exit) / 2, 3 * utils.Defaults.hub_tip_ratio / 4),
                color = 'C3', fontsize = 12
            )

            # annotate stator rows
            ax.annotate(
                "STATOR",
                xy = (0, 0),
                xytext = ((stator.x_inlet + stator.x_exit) / 2, 3 * utils.Defaults.hub_tip_ratio / 4),
                color = 'C0', fontsize = 12
            )

        # plot spline
        ax.plot(xx, spline(xx), color = 'k')

        # plot upper bound datapoints as dots
        ax.plot(
            [
                x for blade_row in self.blade_rows 
                for x in (blade_row.x_inlet, blade_row.x_exit)
            ] + [self.nozzle.x_exit],
            [
                r for blade_row in self.blade_rows for r in (
                    blade_row.inlet[-1].r + blade_row.inlet[-1].dr,
                    blade_row.exit[-1].r + blade_row.exit[-1].dr
                )
            ] + [self.nozzle.exit[-1].r + self.nozzle.exit[-1].dr],
            linestyle = '', marker = '.', color = 'k'
        )

        # create spline for upper bound of nozzle
        spline = make_interp_spline(
            [
                xx[-1] - 1e-3, self.nozzle.x_inlet,
                self.nozzle.x_exit, self.nozzle.x_exit + 1e-3
            ],
            [
                self.nozzle.inlet[-1].r + self.nozzle.inlet[-1].dr,
                self.nozzle.inlet[-1].r + self.nozzle.inlet[-1].dr,
                self.nozzle.exit[-1].r + self.nozzle.exit[-1].dr,
                self.nozzle.exit[-1].r + self.nozzle.exit[-1].dr
            ],
            k = 3
        )

        # plot spline
        ax.plot(yy, spline(yy), color = 'k')
        
        # iterate over all radial positions, apart from outer streamtube
        for index in range(self.N):

            # create spline for lower bound of streamtube
            spline = make_interp_spline(
                [-1e-3] + [
                    x for blade_row in self.blade_rows 
                    for x in (blade_row.x_inlet, blade_row.x_exit)
                ] + [xx[-1] + 1e-3],
                [self.blade_rows[0].inlet[index].r - self.blade_rows[0].inlet[index].dr] + [
                    r for blade_row in self.blade_rows for r in (
                        blade_row.inlet[index].r - blade_row.inlet[index].dr,
                        blade_row.exit[index].r - blade_row.exit[index].dr
                    )
                ] + [self.blade_rows[-1].exit[index].r - self.blade_rows[-1].exit[index].dr],
                k = min(3, len(self.blade_rows))
            )

            # plot spline
            ax.plot(xx, spline(xx), color = 'k')

            # plot lower bound datapoints as dots
            ax.plot(
                [
                    x for blade_row in self.blade_rows 
                    for x in (blade_row.x_inlet, blade_row.x_exit)
                ] + [self.nozzle.x_exit],
                [
                    r for blade_row in self.blade_rows for r in (
                        blade_row.inlet[index].r - blade_row.inlet[index].dr,
                        blade_row.exit[index].r - blade_row.exit[index].dr
                    )
                ] + [self.nozzle.exit[index].r - self.nozzle.exit[index].dr],
                linestyle = '', marker = '.', color = 'k'
            )

            # create spline for lower bound of nozzle
            spline = make_interp_spline(
                [
                    xx[-1] - 1e-3, self.nozzle.x_inlet,
                    self.nozzle.x_exit, self.nozzle.x_exit + 1e-3
                ],
                [
                    self.nozzle.inlet[index].r - self.nozzle.inlet[index].dr,
                    self.nozzle.inlet[index].r - self.nozzle.inlet[index].dr,
                    self.nozzle.exit[index].r - self.nozzle.exit[index].dr,
                    self.nozzle.exit[index].r - self.nozzle.exit[index].dr
                ],
                k = 3
            )

            # plot spline
            ax.plot(yy, spline(yy), color = 'k')

    def plot_spanwise_variations(self, q, label, angle = False):
        """Creates a plot of the spanwise variations of a specified quantity for each blade row."""
        # helper function to convert angles if appropriate
        def convert_angle(x):

            if angle:

                return utils.rad_to_deg(x)
            
            else:

                return x

        # create plot with an axis for each blade row inlet and exit and reshape
        fig, axes = plt.subplots(ncols = 2 * len(self.blade_rows), figsize = (10, 6))
        axes = np.reshape(axes, (len(self.blade_rows), 2))

        # initialise array of r values for convenience
        rr = np.linspace(utils.Defaults.hub_tip_ratio, 1, 100)

        # assign values for capturing appropriate axis limits
        x_min = 1e12
        x_max = -1e12

        # iterate over all axes:
        for ax, blade_row in zip(axes, self.blade_rows):

            # handle case where only one inlet/exit datapoint is available
            if len(blade_row.inlet) == 1:

                # plot a constant value everywhere
                yy_0 = np.full_like(rr, [getattr(inlet.flow_state, q) for inlet in blade_row.inlet])
                yy_1 = np.full_like(rr, [getattr(exit.flow_state, q) for exit in blade_row.exit])

            # if more than one inlet/exit datapoint exists, fit spline
            else:

                # fit spline and store corresponding outputs
                spline = make_interp_spline(
                    [inlet.r for inlet in blade_row.inlet],
                    [getattr(inlet.flow_state, q) for inlet in blade_row.inlet],
                    k = min(2, len(blade_row.inlet) - 1)
                )
                yy_0 = spline(rr)
                spline = make_interp_spline(
                    [exit.r for exit in blade_row.exit],
                    [getattr(exit.flow_state, q) for exit in blade_row.exit],
                    k = min(2, len(blade_row.exit) - 1)
                )
                yy_1 = spline(rr)

            # plot spline
            ax[0].plot(convert_angle(yy_0), rr, alpha = 0.5)
            ax[1].plot(convert_angle(yy_1), rr, alpha = 0.5)

            # plot inlet conditions
            ax[0].plot(
                [convert_angle(getattr(inlet.flow_state, q)) for inlet in blade_row.inlet],
                [inlet.r for inlet in blade_row.inlet],
                linestyle = '', marker = '.'
            )

            # plot exit conditions
            ax[1].plot(
                [convert_angle(getattr(exit.flow_state, q)) for exit in blade_row.exit],
                [exit.r for exit in blade_row.exit],
                linestyle = '', marker = '.'
            )

            # update x-axis limits
            x_min = min(
                x_min, *[convert_angle(getattr(inlet.flow_state, q)) for inlet in blade_row.inlet],
                *[convert_angle(getattr(exit.flow_state, q)) for exit in blade_row.exit]
            )
            x_max = max(
                x_max, *[convert_angle(getattr(inlet.flow_state, q)) for inlet in blade_row.inlet],
                *[convert_angle(getattr(exit.flow_state, q)) for exit in blade_row.exit]
            )

        # flatten axes and iterate
        axes = axes.ravel()
        for ax in axes:

            # set axis x- and y-limits
            ax.set_xlim(x_min - (x_max - x_min) / 10, x_max + (x_max - x_min) / 10)
            ax.set_ylim(utils.Defaults.hub_tip_ratio, 1)

            # set grid and maximum number pf x-ticks
            ax.grid()
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins = 2))

        # set y-label and tight layout
        axes[0].set_ylabel('Dimensionless radius')
        plt.tight_layout()

        # set title
        plt.subplots_adjust(top = 0.9)
        fig.text(
            0.5, 0.95, f"Spanwise variation of {label}",
            ha = 'center', va = 'center', fontsize = 12
        )

        # set x-label
        plt.subplots_adjust(bottom = 0.1)
        fig.text(0.5, 0.03, label, ha = 'center', va = 'center')

# IGNORE EVERYTHING FROM HERE ----------------------------------------------------------------

    def engine_analysis2(self):
        """Determine key performance metrics for the engine system."""
        # initialise arrays for storing key engine metrics
        self.eta_p = np.zeros((len(self.blade_rows[0].inlet) + 1,))
        self.C_th = np.zeros((len(self.blade_rows[0].inlet) + 1,))
        self.eta_prop = np.zeros((len(self.blade_rows[0].inlet) + 1,))
        self.eta_nozz = np.zeros((len(self.blade_rows[0].inlet) + 1,))
        self.eta_comp = np.zeros((len(self.blade_rows[0].inlet) + 1,))
        self.eta_elec = np.zeros((len(self.blade_rows[0].inlet) + 1,))
        for index, (inlet, exit) in enumerate(zip(self.blade_rows[0].inlet, self.nozzle.exit)):

            # calculate polytropic efficiency
            self.eta_p[index] = (
                (utils.gamma - 1) * np.log(exit.flow_state.p_0 / inlet.flow_state.p_0)
                / (utils.gamma * np.log(exit.flow_state.T_0 / inlet.flow_state.T_0))
            )

            # calculate thrust coefficient
            self.C_th[index] = (
                utils.mass_flow_function(self.blade_rows[0].inlet[index].flow_state.M)
                * np.sqrt(utils.gamma - 1)
                * (
                    self.nozzle.exit[index].flow_state.M * np.sqrt(
                    self.nozzle.exit[index].flow_state.T
                    / self.blade_rows[0].inlet[index].flow_state.T_0
                )
                - self.M_flight * utils.stagnation_temperature_ratio(self.M_flight)
                )
            )

            # calculate propulsive efficiency
            self.eta_prop[index] = (
                2 * self.C_th[index]
                / utils.mass_flow_function(self.blade_rows[0].inlet[index].flow_state.M)
                * np.sqrt(utils.stagnation_temperature_ratio(self.M_flight) / (utils.gamma - 1))
                / (
                    self.nozzle.exit[index].flow_state.M**2 * self.nozzle.exit[index].flow_state.T
                    - self.M_flight**2 * utils.stagnation_temperature_ratio(self.M_flight)
                )
            )

            # calculate nozzle efficiency
            self.eta_nozz[index] = (
                (utils.gamma - 1) / 2 * (
                    self.nozzle.exit[index].flow_state.M**2 * self.nozzle.exit[index].flow_state.T
                    - self.M_flight**2 * utils.stagnation_temperature_ratio(self.M_flight)
                ) / (
                    np.power(self.nozzle.exit[index].flow_state.p_0, 1 - 1 / utils.gamma) - 1
                )
            )

            # calculate compressor isentropic compressor
            self.eta_comp[index] = (
                (np.power(self.nozzle.exit[index].flow_state.p_0, 1 - 1 / utils.gamma) - 1)
                / (self.nozzle.exit[index].flow_state.T_0 - 1)
            )

            # calculate electrical efficiency - for now set to 1
            self.eta_elec[index] = 1

        self.eta_p[-1] = np.mean(self.eta_p[:-1])
        self.C_th[-1] = np.sum(self.C_th[:-1])
        self.eta_prop[-1] = np.mean(self.eta_prop[:-1])
        self.eta_nozz[-1] = np.mean(self.eta_nozz[:-1])
        self.eta_comp[-1] = np.mean(self.eta_comp[:-1])
        self.eta_elec[-1] = np.mean(self.eta_elec[:-1])

        self.eta = self.eta_prop * self.eta_nozz * self.eta_comp * self.eta_elec

        print(f"self.eta_p: {self.eta_p}")
        print(f"self.C_th: {self.C_th}")
        print(f"self.eta_prop: {self.eta_prop}")
        print(f"self.eta_nozz: {self.eta_nozz}")
        print(f"self.eta_comp: {self.eta_comp}")

    def determine_efficiency(self):
        """Determine key performance metrics for the engine system and individual stages."""
        # solve for pressure ratio and coefficient and temperature ratio
        self.overall_p_0 = self.nozzle.exit.p_0
        self.overall_T_0 = self.nozzle.exit.T_0
        self.C_p = (
            (self.nozzle.exit.p_0 - 1)
            / (1 - utils.stagnation_pressure_ratio(self.M_flight))
        )

        # solve for isentropic and polytropic efficiency
        self.eta_s = (
            (np.power(self.overall_p_0, (utils.gamma - 1) / utils.gamma) - 1)
            / (self.overall_T_0 - 1)
        )
        self.eta_p = (
            (utils.gamma - 1) * np.log(self.overall_p_0)
            / (utils.gamma * np.log(self.overall_T_0))
        )

        # for each stage, determine flow and stage loading coefficients and reaction
        for stage in self.stages:

            stage.determine_efficiency()

        # determine engine jet velocity ratio
        self.jet_velocity_ratio = (
            self.M_flight / self.nozzle.exit.M * np.sqrt(
                utils.stagnation_temperature_ratio(self.M_flight)
                / utils.stagnation_temperature_ratio(self.nozzle.exit.M)
                / self.nozzle.exit.T_0
            )
        )

        # find the thrust coefficient
        self.C_th = (
            self.nozzle.area_ratio * self.nozzle.inlet.p_0 * (
                utils.impulse_function(self.nozzle.exit.M)
                - utils.stagnation_pressure_ratio(self.M_flight) / self.nozzle.inlet.p_0
                + 2 * utils.dynamic_pressure_function(self.nozzle.exit.M) * self.jet_velocity_ratio
            )
        )

        # find the nozzle static pressure ratio compared to atmospheric
        self.nozzle_p_r = (
            utils.stagnation_pressure_ratio(self.nozzle.exit.M)
            / utils.stagnation_pressure_ratio(self.M_flight)
            * self.nozzle.exit.p_0
        )