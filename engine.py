# import modules

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import copy
from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar
from time import perf_counter as timer
import itertools

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
        Vortex exponent.
    N : float
        Number of annular streamtubes through which to solve the flow.
    """
    def __init__(self, scenario):
        """Create instance of the Engine class."""
        # store input variables
        self.M_flight = scenario.M
        self.C_th_design = scenario.C_th

        # store default engine parameters
        self.no_of_stages = utils.Defaults.no_of_stages
        self.n = utils.Defaults.vortex_exponent
        self.N = utils.Defaults.no_of_annuli

        # preallocate variables as None
        self.C_th = None

        # initial guess for inlet Mach number
        self.M_1 = 0.1

        # create the appropriate number of empty stages and blade rows
        self.stages = []
        self.blade_rows = []
        for i in range(self.no_of_stages):

            # create stage and blade rows and store in lists
            self.stages.append(Stage(self.n, i))
            self.blade_rows.extend(self.stages[-1].blade_rows)

        # create nozzle
        self.nozzle = Nozzle(2 * self.no_of_stages - 0.5, 2 * self.no_of_stages + 0.5)

        # set up root-solving function for thrust coefficient
        def solve_thrust(var):
            """Iterates through engine designs until a given thrust is achieved."""
            # unpack var and use as inlet Mach number, eliminating negative solutions
            self.M_1 = np.abs(float(np.squeeze(var)))

            # draw loading bar
            if utils.Defaults.loading_bar and self.benchmark == None:

                print("|" + "-" * 100 + "|")

            # design engine, store inside scenario and calculate residual
            self.design()
            scenario.engine = self
            residual = self.C_th - self.C_th_design

            # append x and y values for visualising iteration process
            scenario.x.append(var)
            scenario.y.append(residual)

            if utils.Defaults.loading_bar:

                # set benchmark for loading bar
                if self.benchmark == None:

                    # benchmark for loading bar is the first residual
                    self.benchmark = self.C_th - self.C_th_design

                # calculate progress and print loading bar
                new_progress = max(0.0, 100 * min(1.0, np.power(1 - residual / self.benchmark, 4)))
                self.progress = max(self.progress, new_progress)
                bar = f"{utils.Colours.GREEN}|" + "-" * int(self.progress) + f"|{utils.Colours.END}"
                print("\r" + bar, end = "", flush = True)

            # return thrust coefficient residual
            return residual

        fig, ax = plt.subplots()

        # increment number of annuli for analysis, starting with mean-line only
        t1 = timer()

        NN = [1]
        if utils.Defaults.no_of_annuli > 1:
            NN.append(2)
        if utils.Defaults.no_of_annuli > 2:
            NN.append(3)
        if utils.Defaults.no_of_annuli > 3:
            NN.append(utils.Defaults.no_of_annuli)

        for N in NN:

            # initialise variables for storing loading bar progress
            self.benchmark = None
            self.progress = 0

            scenario.x = []
            scenario.y = []
            self.N = N
            print(
                f"{utils.Colours.CYAN}Performing analysis with {self.N} streamtubes..."
                f"{utils.Colours.END}"
            )
            x0 = self.M_1
            x1 = 0.9 * self.M_1
            if self.N == utils.Defaults.no_of_annuli:

                sol = root_scalar(solve_thrust, x0 = x0, x1 = x1, method = "secant")

            else:

                sol = root_scalar(solve_thrust, x0 = x0, x1 = x1, method = "secant", rtol = 2e-1)

            if utils.Defaults.loading_bar:

                print("\r" + "|" + "-" * 100 + "|\n", end = "", flush=True)

            ax.plot(scenario.x, scenario.y, label = f"{self.N}", linestyle = '', marker = '.')

        t2 = timer()
        print(
            f"Engine design completed after {utils.Colours.GREEN}{t2 - t1:.3g}s:"
            f"{utils.Colours.END}"
        )
        print(self)

        # loop over all blade rows
        for blade_row in self.blade_rows:

            # run subroutine to determine blade metal angles and pitch-to-chord
            blade_row.empirical_design()

        ax.grid()
        ax.legend()

        # create cycle of colours
        self.colour_cycle = itertools.cycle(plt.cm.tab10.colors)

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

# design functions --------------------------------------------------------------------------------

    def design(self):
        """Determines appropriate values for blade metal angles for the given requirements."""
        # iterate over all stages
        utils.debug(f"\nself.M_1: {self.M_1}")
        for index, stage in enumerate(self.stages):

            # store rotor and stator as variables for convenience
            rotor = stage.blade_rows[0]
            stator = stage.blade_rows[1]

            # handle first stage
            if index == 0:

                # set first stage to default inlet conditions
                rotor.set_inlet_conditions(self.M_1, utils.Defaults.inlet_swirl, self.N)

            # handle all other stages
            else:

                # set stage inlet to previous stage exit conditions
                rotor.inlet = copy.deepcopy(self.stages[index - 1].blade_rows[1].exit)

            # define blade geometry for that stage
            t1 = timer()
            rotor.rotor_design(stage.phi, stage.psi)
            t2 = timer()
            utils.debug(f"Rotor design duration: {utils.Colours.GREEN}{t2 - t1:.3g}s{utils.Colours.END}")
            stator.inlet = copy.deepcopy(rotor.exit)
            t1 = timer()
            stator.stator_design(index == len(self.stages) - 1)
            t2 = timer()
            utils.debug(f"Stator design duration: {utils.Colours.GREEN}{t2 - t1:.3g}s{utils.Colours.END}")

            # store rotor and stator as such in stage for convenience
            stage.rotor = rotor
            stage.stator = stator

        # set nozzle inlet conditions
        self.nozzle.inlet = copy.deepcopy(self.blade_rows[-1].exit)

        # design nozzle to match atmospheric pressure in jet periphery
        p_atm = utils.stagnation_pressure_ratio(self.M_flight)
        self.nozzle.nozzle_design(p_atm)
        self.evaluate()

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

    def evaluate(self):
        """Determine key performance metrics for the engine system."""
        # investigate nozzle performance
        self.nozzle.evaluate(self.M_1, self.M_flight)

        # iterate over all stages
        for stage in self.stages:

            # investigate stage performance
            stage.evaluate()

        # sum thrust coefficients
        self.C_th = np.sum([exit.C_th for exit in self.nozzle.exit])

        # find mass-averaged propulsive efficiency
        self.eta_prop = np.mean([exit.eta_prop for exit in self.nozzle.exit])

        # find mass-averaged nozzle efficiency
        self.eta_nozz = np.mean([exit.eta_nozz for exit in self.nozzle.exit])

        # find mass-averaged compressor (isentropic) efficiency
        self.eta_comp = np.mean([exit.eta_comp for exit in self.nozzle.exit])

        # take electric efficiency as 1 for now
        self.eta_elec = 1
        
        # find mass-averaged jet velocity ratio
        self.jet_velocity_ratio = np.mean([exit.jet_velocity_ratio for exit in self.nozzle.exit])

        # store nozzle area ratio
        self.nozzle_area_ratio = self.nozzle.A_exit

# plotting functions ------------------------------------------------------------------------------
    
    def plot_velocity_triangles(self):
        """Function to plot the velocity triangles and pressure and temperature distributions."""
        # create plot for displaying velocity triangles
        fig, ax = plt.subplots(figsize = (10, 6))

        # determine factor to scale Mach triangles by
        scaling = 1 / (4 * self.M_1)

        # initialise array of spanwise position indices to plot
        spanwise_positions = [0]

        # if more than 2 streamtubes exist
        if self.N > 2:

            # plot central streamtube
            spanwise_positions.append(int(np.floor(self.N / 2)))

        # if more than 1 streamtubes exist
        if self.N > 1:

            # plot outer-most streamtube
            spanwise_positions.append(-1)

        # iterate over all blade rows
        for index, blade_row in enumerate(self.blade_rows):

            # iterate over all inlet and exit streamtubes chosen for plotting
            for j, k in enumerate(spanwise_positions):

                # plot blade row at chosen spanwise positions
                blade_row.plot_blade_row(ax, index, j, k, scaling)

        # set legend labels
        ax.plot([], [], color = 'C0', label = 'Absolute Mach number')
        ax.plot([], [], color = 'C4', label = 'Relative Mach number')
        ax.plot([], [], color = 'C3', label = 'Blade Mach number')

        # configure plot
        ax.set_aspect('equal')
        ax.grid()
        ax.legend()
        plt.tight_layout()

    def plot_contours(self):
        """Creates a plot of a section view of the engine with contours of a specified quantity."""
        # plotting parameters
        alpha = 0.5
        rotor_tip_span = 0.98

        # create a plot
        fig, ax = plt.subplots(figsize = (10, 6))

        # initialise fine array of x-values
        xx = np.linspace(0, len(self.blade_rows) - 0.5, 200)
        yy = np.linspace(xx[-1], xx[-1] + 1, 50)

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
                color = 'C3', fontsize = 12, ha = 'center', va = 'center'
            )

            # annotate stator rows
            ax.annotate(
                "STATOR",
                xy = (0, 0),
                xytext = ((stator.x_inlet + stator.x_exit) / 2, 3 * utils.Defaults.hub_tip_ratio / 4),
                color = 'C0', fontsize = 12, ha = 'center', va = 'center'
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

        # add grid
        ax.grid()

    def plot_spanwise_variations(self, quantities):
        """Creates a plot of the spanwise variations of a specified quantity for each blade row."""
        # separate input list into pairs of values
        q_label_bools = [quantities[i:i + 3] for i in range(0, len(quantities), 3)]

        # helper function to retrieve values from appropriate place and with correct units
        def get_attribute(in_out, q, label, bool):
            """"""
            # when value to be retrieved is a flow value
            if bool:

                # return False if value does not exist
                if not hasattr(in_out.flow_state, q):

                    return False

                # read attribute
                x = getattr(in_out.flow_state, q)

            # when value to be retrieved is not a flow value
            else:

                # return False if value does not exist
                if not hasattr(in_out, q):

                    return False

                # read attribute
                x = getattr(in_out, q)

            # if attribute is an angle
            if "angle" in label or "Angle" in label:

                # convert to degrees and return
                return utils.rad_to_deg(x)
            
            # for all other attributes
            else:

                # return as is
                return x

        # create plot with an axis for each blade row inlet and exit and reshape axes
        fig, axes = plt.subplots(ncols = 2 * len(self.blade_rows) + 2, figsize = (15, 7))
        axes = np.reshape(axes, (len(self.blade_rows) + 1, 2))

        # assign values for capturing appropriate axis limits
        x_min = 1e12
        x_max = -1e12
        
        # iterate over all pairs of quantity and label
        for i, q_label_bool in enumerate(q_label_bools):

            # separate into quantity, label and boolean value for convenience
            q = q_label_bool[0]
            label = q_label_bool[1]
            bool = q_label_bool[2]

            # set colour
            colour = next(self.colour_cycle)

            # set legend entry in final axis
            axes[-1][1].plot([], [], color = colour, label = label)

            # iterate over all axes:
            for ax, blade_row in zip(axes, self.blade_rows + [self.nozzle]):

                # proceed with inlet and outlet successively
                for index, inlet_outlet in enumerate([blade_row.inlet, blade_row.exit]):

                    # create array of radius values
                    rr = np.linspace(
                        inlet_outlet[0].r - inlet_outlet[0].dr, 
                        inlet_outlet[-1].r + inlet_outlet[-1].dr,
                        100
                    )

                    # store array of attributes separately for convenience
                    qq = [get_attribute(in_out, q, label, bool) for in_out in inlet_outlet]

                    # if attribute is None, skip block of code
                    if False in qq or None in qq:

                        continue

                    # handle case where only one inlet/exit datapoint is available
                    if len(blade_row.inlet) == 1:

                        # plot a constant value everywhere
                        ax[index].plot(rr, np.full_like(rr, [get_attribute(in_out, q, label, bool) for in_out in inlet_outlet]))

                    # if more than one inlet/exit datapoint exists, fit spline
                    else:

                        # fit spline and store corresponding outputs
                        spline = make_interp_spline(
                            [in_out.r for in_out in inlet_outlet],
                            [get_attribute(in_out, q, label, bool) for in_out in inlet_outlet],
                            k = min(2, len(blade_row.inlet) - 1)
                        )

                        # plot spline
                        ax[index].plot(spline(rr), rr, alpha = 0.5, linewidth = 3, color = colour)

                        # plot inlet conditions
                        ax[index].plot(
                            [get_attribute(in_out, q, label, bool) for in_out in inlet_outlet],
                            [in_out.r for in_out in inlet_outlet],
                            linestyle = '', marker = '.', markersize = 8, color = colour
                        )

                    # for first quantity only
                    if i == 0:

                        # shade areas corresponding to hub and casing
                        ax[index].axhspan(0, rr[0], alpha=0.3, color='gray')
                        ax[index].axhspan(rr[-1], 1, alpha=0.3, color='gray')

                    # update x-axis limits
                    x_min = min(
                        x_min, *[get_attribute(in_out, q, label, bool) for in_out in inlet_outlet]
                    )
                    x_max = max(
                        x_max, *[get_attribute(in_out, q, label, bool) for in_out in inlet_outlet]
                    )

        # flatten axes and iterate
        axes = axes.ravel()
        for ax in axes:

            # set axis x- and y-limits
            ax.set_xlim(x_min - (x_max - x_min) / 10, x_max + (x_max - x_min) / 10)
            ax.set_ylim(0, 1)

            # set grid and maximum number pf x-ticks
            ax.grid()
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins = 2))

        # set y-label and tight layout
        axes[0].set_ylabel('Dimensionless radius')
        plt.tight_layout()

        # set x-label
        plt.subplots_adjust(bottom = 0.16)

        # create legend
        ax.legend(loc='center', bbox_to_anchor=(0.5, 0.05), bbox_transform=fig.transFigure)
