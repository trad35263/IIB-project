# import modules

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from scipy.optimize import brentq

from stage_2025_10_11 import Stage
from blade_row_2025_10_11 import Blade_row
from nozzle_2025_10_11 import Nozzle
import utils_2025_10_11 as utils

# temporary
from flow_state_2025_10_11 import Flow_state

# 1.0 define Engine class

class Engine:
    """
    Used to store multiple (if applicable) stages and determine the overall engine performance.
    
    Parameters
    ----------
    no_of_stages : int
        Number of rotor-stator compressor stages to add to the engine.
    """
    def __init__(self, no_of_stages):
        """Create instance of the Engine class."""
        # create intake
        self.intake = Nozzle(
            utils.Defaults.intake_area_ratio
        )

        # create inlet guide vanes
        self.blade_rows = []
        self.inlet_guide_vanes = Blade_row(
            0,
            utils.Defaults.inlet_guide_vanes_blade_angle,
            1,
            1,
            utils.Defaults.stagnation_pressure_loss_coefficient
        )
        self.blade_rows.append(self.inlet_guide_vanes)

        # create stages
        self.stages = []
        for i in range(no_of_stages):

            self.stages.append(Stage())
            self.blade_rows.extend(self.stages[-1].blade_rows)

        # create outlet guide vanes
        self.outlet_guide_vanes = Blade_row(
            0,
            utils.Defaults.outlet_guide_vanes_blade_angle,
            1,
            1,
            utils.Defaults.stagnation_pressure_loss_coefficient
        )
        self.blade_rows.append(self.outlet_guide_vanes)

        # create exit nozzle
        self.nozzle = Nozzle(
            utils.Defaults.nozzle_area_ratio
        )

    def __str__(self):
        """Print a simplifed summary of the information contained in the class."""
        # print header and state number of blade rows
        string = f"""
{utils.Colours.UNDERLINE}Engine Summary{utils.Colours.END}

Number of stages: {utils.Colours.GREEN}{len(self.stages)}{utils.Colours.END}
Blade row configuration:
"""

        # print abbreviated form of each type of blade row and return final string
        string += f"{self.inlet_guide_vanes.short_label}-"
        for stage in self.stages:

            for blade_row in stage.blade_rows:

                string += f"{blade_row.short_label}-"

        string += f"{self.outlet_guide_vanes.short_label}-"
        string = string[:-1]
        return string

    def __repr__(self):
        """Print a detailed summary of the information contained in the class."""
        string = f"""
{utils.Colours.UNDERLINE}Engine Details{utils.Colours.END}\n
"""
        string += f"{self.intake}"
        for index, blade_row in enumerate(self.blade_rows):

            string += f"[{index}] {blade_row}"

        string += f"{self.nozzle}"

        return string

    def analyse(self):
        """Analyses the entire engine system."""
        # analyse intake
        self.intake.solve_nozzle()
        self.inlet_guide_vanes.inlet = self.intake.exit

        # analyse all blade_rows
        for index, blade_row in enumerate(self.blade_rows):

            # skip for first blade row
            if index > 0:

                # create a nozzle and solve to account for inter-blade row area change
                nozzle = Nozzle(
                    self.blade_rows[index - 1].casing_area_ratio,
                    self.blade_rows[index - 1].exit
                )
                nozzle.solve_nozzle()
                blade_row.inlet = nozzle.exit

            # calculate conditions at blade row exit
            blade_row.solve_blade_row()

        # analyse nozzle
        self.nozzle.inlet = self.outlet_guide_vanes.exit
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
        height_ratio = 8 / len(self.stages)
        fig, (ax_upper, ax_lower) = plt.subplots(
            2, 1, figsize = (12, 7),
            gridspec_kw={'height_ratios': [height_ratio, 1]},
            sharex = True
        )
        ax_lower1 = ax_lower.twinx()
        ax_lower2 = ax_lower.twinx()

        # set title and subtitle
        ax_upper.title.set_text(
            f"Inlet flow coefficient: {utils.Defaults.inlet_flow_coefficient:.3g}\n"
            f"Inlet Mach number: {utils.Defaults.inlet_Mach_number:.3g}\n"
            f"Jet velocity ratio: {self.jet_velocity_ratio:.3g}"
        )
        ax_upper.title.set_fontsize(12)
        ax_lower.title.set_text(
            f"Stagnation pressure ratio = {self.overall_p_0_ratio:.3g}\n"
            f"Stagnation temperature ratio = {self.overall_T_0_ratio:.3g}\n"
            f"Isentropic efficiency = {self.eta_s:.3g}\n"
            f"Polytropic efficiency = {self.eta_p:.3g}"
        )
        ax_lower.title.set_fontsize(10)

        # plot  quantities against blade row number
        xx = np.arange(len(self.flow_states)) / 2 - 0.75
        ax_lower.plot(
            xx, [flow_state.p_0_ratio for flow_state in self.flow_states],
            color = 'C1', label = 'Pressure'
        )
        ax_lower1.plot(
            xx, [flow_state.T_0_ratio for flow_state in self.flow_states],
            color = 'C2', label = 'Temperature'
        )
        ax_lower2.plot(
            xx, [flow_state.M for flow_state in self.flow_states],
            color = 'k', label = 'Mach number'
        )

        def display_blade_row(blade_row, index):
            """Helper function to plot a given blade row and corresponding velocity triangle."""
            # construct rotation matrix and carry out matrix multiplication
            R = np.array([
                [np.cos(blade_row.blade_angle), -np.sin(blade_row.blade_angle)],
                [np.sin(blade_row.blade_angle),  np.cos(blade_row.blade_angle)]
            ])
            coords = utils.data.copy()
            data_rot = coords @ R.T

            # display rotated NACA aerofoils in position
            ax_upper.plot(data_rot[:, 0] + index, data_rot[:, 1], color = blade_row.colour)

            # display relative velocities
            ax_upper.annotate(
                "",
                xy = (
                    index + blade_row.exit.M * np.cos(blade_row.exit.alpha),
                    blade_row.exit.M * np.sin(blade_row.exit.alpha) - (
                        blade_row.exit.M * np.cos(blade_row.exit.alpha) / blade_row.exit.phi
                        * np.sign(blade_row.blade_speed_ratio)
                    )
                ),
                xytext = (index, 0),
                arrowprops = dict(
                    arrowstyle = "->", color = 'C4',
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )

            # display absolute velocity vectors as arrows
            ax_upper.annotate(
                "",
                xy = (
                    index + blade_row.exit.M * np.cos(blade_row.exit.alpha),
                    blade_row.exit.M * np.sin(blade_row.exit.alpha)
                ),
                xytext = (index, 0),
                arrowprops = dict(
                    arrowstyle = "->", color = 'C0',
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )

            # display blade row speeds as arrows
            ax_upper.annotate(
                "",
                xy = (
                    index,
                    (
                        blade_row.exit.M * np.cos(blade_row.exit.alpha) / blade_row.exit.phi
                        * np.sign(blade_row.blade_speed_ratio)
                    )
                ),
                xytext = (index, 0),
                arrowprops = dict(
                    arrowstyle = "->", color = 'C3',
                    shrinkA = 0, shrinkB = 0, lw = 1.5
                )
            )

        # display all blade rows
        for index, blade_row in enumerate(self.blade_rows):

            display_blade_row(blade_row, index)

        #area_ratios = [self.intake.area_ratio]
        area_ratios = [blade_row.casing_area_ratio for blade_row in self.blade_rows]
        areas = self.intake.area_ratio * np.cumprod(area_ratios)
        #for blade_row in self.blade_rows:
            
            #area_ratios.append(blade_row.casing_area_ratio)

        #area_ratios.append(self.nozzle.area_ratio)

        ax_upper.plot(np.arange(len(areas)), areas, color = 'k')
        ax_upper.plot(np.arange(len(areas)), -np.array(areas), color = 'k')

        def display_nozzle(nozzle, index, scaling=1):
            """Helper function to visualise a given nozzle's area change."""
            ax_upper.plot([index, index + 1], scaling * np.array([1, nozzle.area_ratio]), color = nozzle.colour)
            ax_upper.plot([index, index + 1], scaling * np.array([-1, -nozzle.area_ratio]), color = nozzle.colour)

        # display intake and outlet nozzle
        display_nozzle(self.intake, -1)
        display_nozzle(self.nozzle, 2 * len(self.stages) + 1, areas[-1])

        # label velocity arrows in legend
        ax_upper.plot([], [], color = 'C0', label = 'Absolute flow velocity')
        ax_upper.plot([], [], color = 'C3', label = 'Blade speeds')

        # set axis limits and aspect ratio
        #ax_upper.set_xlim(1e-6, len(self.stages) + 1 - 1e-6)
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
        ax_lower2.spines["right"].set_position(("axes", 1.1))
        ax_lower2.spines["right"].set_visible(True)

        # set axis labels
        ax_lower.set_xlabel("Blade row number")
        ax_upper.set_ylabel("Mach number", color = 'C0')
        ax_lower.set_ylabel("Stagnation pressure ratio", color = 'C1')
        ax_lower1.set_ylabel("Stagnation temperature ratio", color = 'C2')
        ax_lower2.set_ylabel("Mach number", color = 'k')

        # set grids on both plots
        ax_upper.grid()
        ax_lower.grid()

        ax_upper.set_ylim(-1.2, 1.2)

        plt.tight_layout()

    def determine_efficiency(self):
        """Determine key performance metrics for the engine system and individual stages."""
        # solve for pressure ratio and coefficient and temperature ratio
        self.overall_p_0_ratio = self.nozzle.exit.p_0_ratio
        self.overall_T_0_ratio = self.nozzle.exit.T_0_ratio
        self.C_p = (
            (self.nozzle.exit.p_0_ratio - 1)
            / (1 - utils.stagnation_pressure_ratio(self.intake.inlet.M))
        )

        # solve for isentropic and polytropic efficiency
        self.eta_s = (
            (np.power(self.overall_p_0_ratio, (utils.gamma - 1) / utils.gamma) - 1)
            / (self.overall_T_0_ratio - 1)
        )
        self.eta_p = (
            (utils.gamma - 1) * np.log(self.overall_p_0_ratio)
            / (utils.gamma * np.log(self.overall_T_0_ratio))
        )

        # for each stage, determine flow and stage loading coefficients and reaction
        for stage in self.stages:

            # SHOULD THIS BE WITHIN Stage() ?

            stage.phi = stage.blade_rows[0].inlet.phi
            stage.psi = (
                np.power(utils.velocity_function(stage.blade_rows[0].inlet.M), -2)
                * stage.phi**2 * np.power(np.cos(stage.blade_rows[0].inlet.alpha), -2)
                * (stage.blade_rows[-1].exit.T_0_ratio / stage.blade_rows[0].inlet.T_0_ratio - 1)
            )
            stage.reaction = (
                (
                    utils.stagnation_temperature_ratio(stage.blade_rows[0].exit.M)
                    / utils.stagnation_temperature_ratio(stage.blade_rows[0].exit.M)
                    * stage.blade_rows[0].exit.T_0_ratio / stage.blade_rows[0].inlet.T_0_ratio - 1
                ) / (
                    utils.stagnation_temperature_ratio(stage.blade_rows[-1].exit.M)
                    / utils.stagnation_temperature_ratio(stage.blade_rows[0].exit.M)
                    * stage.blade_rows[-1].exit.T_0_ratio / stage.blade_rows[0].inlet.T_0_ratio - 1
                )
            )

        self.jet_velocity_ratio = (
            self.intake.inlet.M / self.nozzle.exit.M * np.sqrt(
                utils.stagnation_temperature_ratio(self.intake.inlet.M)
                / utils.stagnation_temperature_ratio(self.nozzle.exit.M)
                / self.nozzle.exit.T_0_ratio
            )
        )

        self.C_T = (
            utils.mass_flow_function(self.intake.inlet.M)
            * (utils.velocity_function(self.nozzle.exit.M) - utils.velocity_function(self.intake.inlet.M))
        )