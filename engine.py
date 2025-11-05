# import modules

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from scipy.optimize import brentq
from scipy.optimize import root_scalar

from stage import Stage
from blade_row import Blade_row
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
    """
    def __init__(self, no_of_stages, M_1, scenario):
        """Create instance of the Engine class."""
        # design engine for given default values of thrust etc.
        self.no_of_stages = no_of_stages
        self.M_1 = M_1
        self.M_flight = scenario.M
        self.C_th_design = scenario.C_th
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
        # create the appropriate number of empty stages and blade rows
        self.stages = []
        self.blade_rows = []
        for i in range(self.no_of_stages):

            self.stages.append(Stage())
            self.blade_rows.extend(self.stages[-1].blade_rows)

        # iterate over all stages
        for index, stage in enumerate(self.stages):

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
            #print(f"{utils.Colours.GREEN}Designing rotor...{utils.Colours.END}")
            rotor.rotor_design(stage.phi, stage.psi)
            stator.inlet = rotor.exit
            #print(f"{utils.Colours.GREEN}Designing stator...{utils.Colours.END}")
            stator.stator_design(
                stage.reaction,
                stage.blade_rows[0].inlet[index].flow_state.T,
                stage.blade_rows[0].exit[index].flow_state.T,
                index == len(self.stages) - 1
            )

        # create nozzle and set inlet conditions
        self.nozzle = Nozzle()
        self.nozzle.inlet = self.blade_rows[-1].exit

        # residual function - think this is the wrong approach
        def residual(M_guess):
            #print(f"\nM_guess: {M_guess}")
            """Residual function to find root of."""
            # for a guessed value of nozzle exit Mach number, determine the jet velocity ratio
            jet_velocity_ratio = (
                self.M_flight / M_guess * np.sqrt(
                    utils.stagnation_temperature_ratio(self.M_flight)
                    / utils.stagnation_temperature_ratio(M_guess)
                    / self.nozzle.inlet.T_0
                )
            )

            #print(f"jet_velocity_ratio: {jet_velocity_ratio}")

            # find the corresponding nozzle area ratio
            area_ratio = (
                utils.stagnation_density_ratio(self.M_1)
                / utils.stagnation_density_ratio(M_guess)
                * self.M_1 / M_guess * np.sqrt(
                    utils.stagnation_temperature_ratio(self.M_1)
                    / utils.stagnation_temperature_ratio(M_guess)
                    * self.nozzle.inlet.T_0
                )
                / self.nozzle.inlet.p_0
            )

            #print(f"area_ratio: {area_ratio}")

            # find the corresponding thrust coefficient
            C_th_guess = (
                area_ratio * self.nozzle.inlet.p_0 * (
                    utils.impulse_function(M_guess)
                    - utils.stagnation_pressure_ratio(self.M_flight)
                    / self.nozzle.inlet.p_0
                    + 2 * utils.dynamic_pressure_function(M_guess) * jet_velocity_ratio
                )
            )
            #print(f"C_th_guess: {C_th_guess}")
            """C_th_guess = (
                utils.impulse_function(M_guess)
                - utils.stagnation_pressure_ratio(self.M_flight) / self.blade_rows[-1].exit.p_0
                - 2 * utils.dynamic_pressure_function(M_guess) * jet_velocity_ratio
            )"""

            # return difference to design thrust coefficient
            return C_th_guess - self.C_th_design

        # uncomment for debugging
        """xx = np.linspace(self.M_flight, 1 - 1e-3, 20)
        yy = [residual(x) + self.C_th_design for x in xx]
        zz = [utils.stagnation_pressure_ratio(x) * self.nozzle.inlet.p_0 for x in xx]
        pp = [utils.stagnation_pressure_ratio(self.nozzle.inlet.M) * self.nozzle.inlet.p_0 for x in xx]
        pp_0 = [utils.stagnation_pressure_ratio(self.M_flight) for x in xx]
        fig, ax = plt.subplots()
        ax.plot(xx, yy, label = "Thrust coefficient")
        ax.plot(xx, zz, label = "Nozzle pressure ratio")
        ax.plot(xx, pp, label = "Nozzle static pressure")
        ax.plot(xx, pp_0, label = "Atmospheric static pressure")
        ax.grid()
        ax.legend()
        ax.set_xlabel("Nozzle exit Mach number")
        plt.show()"""

        """try:

            solution = root_scalar(residual, bracket = [self.M_flight, 1], method = "brentq")
            M_j = solution.root

        except:

            print(
                f"{utils.Colours.RED}Error occured solving for thrust coefficient! "
                f"Setting nozzle to choke instead.{utils.Colours.END}"
            )
            M_j = 1"""

        # design nozzle to match atmospheric pressure in jet periphery
        #print(f"{utils.Colours.GREEN}Designing nozzle...{utils.Colours.END}")
        p_atm = utils.stagnation_pressure_ratio(self.M_flight)
        self.nozzle.nozzle_design(p_atm)
        self.engine_analysis()

        # determine Mach number for which nozzle exit static pressure is equal to atmospheric
        """p_r = utils.stagnation_pressure_ratio(self.M_flight) / self.nozzle.inlet.p_0
        M_j = utils.invert(utils.stagnation_pressure_ratio, p_r)
        if M_j == None:

            M_j = 1

        self.nozzle.define_nozzle_geometry(M_j)"""
        #self.determine_efficiency()

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
        # create matplotlib plot with multiple axes
        # could not get this to work well!
        """height_ratio = 24 / len(self.blade_rows)
        fig, (ax_upper, ax_lower) = plt.subplots(
            2, 1, figsize = (9, 7),
            gridspec_kw={'height_ratios': [height_ratio, 1]},
            sharex = True
        )"""
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

    def engine_analysis(self):
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

        print(f"self.C_th: {self.C_th}")
        print(f"self.eta_prop: {self.eta_prop}")
        print(f"self.eta_nozz: {self.eta_nozz}")
        print(f"self.eta_comp: {self.eta_comp}")

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