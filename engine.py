# import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path
import matplotlib.patches as patches

from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar
from scipy.io import savemat

from time import perf_counter as timer
import itertools
from scipy.interpolate import interp1d
from datetime import datetime
import json

#import matlab.engine
from pathlib import Path as FilePath

# import system modules
import copy
import inspect
import os
import io

# import custom classes
from stage import Stage
from rotor import Rotor
from nozzle import Nozzle
from flow_state import Flow_state
from annulus import Annulus
import utils

# define Engine class
class Engine:
    """
    Used to store multiple (if applicable) stages and determine the overall engine performance.
    
    Parameters
    ----------
    scenario : class
        Instance of the Flight_scenario class for which this engine is designed.
    no_of_stages : int
        Number of stages in the compressor.
    vortex_exponent : float
        Vortex exponent describing the distribution of stage loading across a rotor.
    solver_order : int
        Order of polynomial fits to be considered when solving the flowfield.
    phi : float
        Flow coefficient.
    psi : float
        Stage loading coefficient.
    """
    def __init__(
            self,
            scenario,
            no_of_stages = utils.Defaults.no_of_stages,
            vortex_exponent = utils.Defaults.vortex_exponent,
            solver_order = utils.Defaults.solver_order,
            Y_p = utils.Defaults.Y_p,
            phi = utils.Defaults.phi,
            psi = utils.Defaults.psi
        ):
        """Creates an instance of the Engine class."""
        # store variables from input scenario
        self.M_flight = scenario.M
        self.C_th_design = scenario.C_th
        self.hub_tip_ratio = scenario.hub_tip_ratio
        self.diameter = scenario.diameter
        self.p_atm = scenario.p / scenario.p_0

        # store flight scenario (deepcopy to avoid circular reference)
        self.scenario = copy.deepcopy(scenario)

        # store input variables
        self.no_of_stages = int(no_of_stages)
        self.vortex_exponent = vortex_exponent
        self.solver_order = int(solver_order)
        self.Y_p = Y_p
        self.phi = phi
        self.psi = psi

        # create the appropriate number of empty stages and blade rows
        self.stages = []
        self.blade_rows = []
        for index in range(self.no_of_stages):

            # create stage and blade rows and store in lists
            stage = Stage(self.phi, self.psi, self.vortex_exponent, self.Y_p, index)
            self.stages.append(stage)
            self.blade_rows.extend(self.stages[-1].blade_rows)

            # handle first stage
            if index == 0:

                # set first stage to default inlet conditions
                stage.rotor.inlet = Annulus()

            # for all other stages
            else:

                # set rotor exit conditions to previous stage stator exit conditions
                stage.rotor.inlet = self.stages[index - 1].stator.exit

            stage.rotor.exit = Annulus()
            stage.stator.exit = Annulus()

            # set stator exit conditions to rotor exit conditions
            stage.stator.inlet = stage.rotor.exit

        # create nozzle and set inlet conditions to final stage stator exit conditions
        self.nozzle = Nozzle()
        self.nozzle.inlet = self.stages[-1].stator.exit
        self.nozzle.exit = Annulus()

        # start timer
        t1 = timer()

        # design engine
        self.design()

        # loop over all blade rows
        for blade_row in self.blade_rows:

            # sever shared references between inlet and exit annuli
            blade_row.exit = copy.deepcopy(blade_row.exit)

        # end timer and print feedback
        t2 = timer()
        print(
            f"Engine design completed after {utils.Colours.GREEN}{t2 - t1:.3g}s:"
            f"{utils.Colours.END}"
        )
        print(self)

        # create cycle of colours
        self.colour_cycle = itertools.cycle(plt.cm.tab10.colors)

        # initialise empty list of geometry and off-design dictionaries
        self.geometries = []
        self.off_designs = []

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
        """Designs the engine system for the given flight scenario and inputs."""
        # create empty arrays to store guesses
        self.M_1_guesses = []
        self.C_th_guesses = []

        # root-finding function
        def solve_thrust(M_1):
            """Returns the thrust residual of the engine for a given inlet Mach number."""
            # store inlet Mach number and set at rotor inlet
            self.M_1_guesses.append(M_1)
            self.M_1 = M_1
            self.blade_rows[0].set_inlet_conditions(self.M_1, self.hub_tip_ratio)
            self.m_dot = utils.mass_flow_function(self.M_1)

            def solve_blade_row(v_x_hub, blade_row):
                """Returns the relevant residual for a blade row for a given hub axial velocity."""
                # design blade row and return residual
                blade_row.design(v_x_hub, self.hub_tip_ratio)
                residual = blade_row.exit.rr[-1]**2 - 1
                blade_row.v_x_guesses.append(v_x_hub)
                blade_row.residual_guesses.append(residual)
                return residual

            def solve_nozzle(v_x_hub):
                """Returns the residual for a nozzle for a given centreline axial velocity."""
                # design nozzle and return residual
                self.nozzle.design(v_x_hub, self.hub_tip_ratio)
                residual = self.nozzle.exit.p[-1] - self.p_atm
                self.nozzle.v_x_guesses.append(v_x_hub)
                self.nozzle.residual_guesses.append(residual)
                return residual

            # loop over all blade rows
            for blade_row in self.blade_rows:

                # empty lists of blade row guesses
                blade_row.v_x_guesses = []
                blade_row.residual_guesses = []

                # set initial guess
                x0 = blade_row.inlet.v_x[0]
                x1 = 0.9 * x0

                # solve for blade row exit conditions
                sol = root_scalar(
                    solve_blade_row, x0 = x0, x1 = x1, method = 'secant', maxiter = 20,
                    args = (blade_row,)
                )

            # wrap in try-except loop to discard nozzle design failures
            try:

                # empty lists of blade row guesses
                self.nozzle.v_x_guesses = []
                self.nozzle.residual_guesses = []

                # set initial guess
                x0 = self.nozzle.inlet.v_x[0]
                x1 = 0.9 * x0

                # solve for nozzle exit conditions
                sol = root_scalar(
                    solve_nozzle, x0 = x0, x1 = x1, method = 'secant', maxiter = 20
                )

            # catch exceptions
            except ValueError as error:

                # return arbitrarily large residual
                print(error)
                return 1e9

            # evaluate engine performance
            self.evaluate()
            self.C_th_guesses.append(self.C_th)

            # return difference between actual and design thrust
            return self.C_th - self.C_th_design

        # calculate initial guess
        self.initial_guess()
        x0 = self.M_1
        x1 = 0.99 * x0

        # solve iteratively
        sol = root_scalar(
            solve_thrust, x0 = x0, x1 = x1, method = 'secant', maxiter = utils.Defaults.maxiter
        )
        print(f"sol: {sol}")

    def initial_guess(self):
        """Calculates an initial inlet Mach number guess for the solver."""
        # start timer
        t1 = timer()

        # set up grid of inlet Mach number and span
        self.M_1_initial = np.linspace(0, 1, utils.Defaults.solver_grid)
        rr = np.linspace(self.hub_tip_ratio, 1, utils.Defaults.solver_grid)

        M_1_grid, rr_grid = np.meshgrid(self.M_1_initial, rr)
        velocity_function_1 = utils.velocity_function(M_1_grid)

        # calculate geometric mean radius
        r_mean = np.sqrt(0.5 * (1 + self.hub_tip_ratio**2))

        # calculate compressor stagnation temperature ratio
        T_03_T_01 = (
            1 + self.psi * velocity_function_1**2 * np.power(
                r_mean / rr_grid, self.vortex_exponent - 1
            ) / self.phi**2
        )

        # find static temperature ratio required to satisfy jet boundary condition
        T_j_T_0j = utils.stagnation_temperature_ratio(self.scenario.M) / T_03_T_01

        # get list of stagnation temperature ratios to interpolate
        T_T_0 = utils.stagnation_temperature_ratio(self.M_1_initial)

        # calculate jet Mach number required to satisfy jet boundary condition
        M_j = np.interp(T_j_T_0j, T_T_0[::-1], self.M_1_initial[::-1])

        # calculate mass flow functions at inlet and in jet
        mass_flow_function_1 = utils.mass_flow_function(M_1_grid)
        mass_flow_function_j = utils.mass_flow_function(M_j)

        # calculate nozzle area ratio per streamline assuming isentropic compressor
        sigma_grid = (
            mass_flow_function_1 * np.power(T_03_T_01, 0.5 - utils.gamma / (utils.gamma - 1))
            / mass_flow_function_j
        )

        # calculate inlet streamtube areas (array has length N - 1)
        A_1i = np.pi * np.array([r_1i**2 - r_1i_1**2 for (r_1i, r_1i_1) in zip(rr[1:], rr[:-1])])

        # calculate mid-point average of streamline area ratio values
        sigma_grid_midpoint = 0.5 * (sigma_grid[1:, :] + sigma_grid[:-1, :])

        # calculate exit streamtube areas
        A_ji_grid = A_1i[:, None] * sigma_grid_midpoint
        A_ji_grid_cumulative = np.cumsum(A_ji_grid, axis = 0)

        # calculate exit streamline radii
        rr_j = np.sqrt(A_ji_grid_cumulative / np.pi)
        rr_j = np.vstack([np.zeros((1, rr_j.shape[1])), rr_j])

        # calculate thrust coefficient
        M_j_2_r = M_j**2 * rr_j
        self.C_th_initial = np.zeros(utils.Defaults.solver_grid)
        for index in range(utils.Defaults.solver_grid):
            self.C_th_initial[index] = (
                2 * utils.gamma / (1 - self.hub_tip_ratio**2)
                * (rr_j[-1, index] / rr[-1])**2
                * utils.stagnation_pressure_ratio(self.scenario.M)
                * utils.cumulative_trapezoid(rr_j[:, index], M_j_2_r[:, index])[-1]
                - utils.mass_flow_function(self.M_1_initial[index])
                * utils.velocity_function(self.scenario.M)
            )

        # interpolate to find actual inlet Mach number
        self.M_1 = np.interp(self.C_th_design, self.C_th_initial[1:], self.M_1_initial[1:])

        # end timer
        t2 = timer()
        print(f"Initial guess calculated in {t2 - t1:.4g}s!")

    def evaluate(self):
        """Determine key performance metrics for the engine system."""
        # investigate nozzle performance and extract properties
        self.nozzle.evaluate(self.hub_tip_ratio)
        self.T_0_ratio = self.nozzle.T_0_ratio
        self.p_0_ratio = self.nozzle.p_0_ratio
        self.nozzle_area_ratio = self.nozzle.area_ratio

        # calculate thrust coefficient (including pressure terms)
        self.C_th = (
            self.nozzle.C_th[-1]
            - utils.mass_flow_function(self.M_1)
            * utils.velocity_function(self.M_flight)
            - self.nozzle_area_ratio * self.p_atm
        )

        # calculate jet velocity ratio
        rho_v_2_r = (
            self.nozzle.exit.p * self.nozzle.exit.M**2
            * self.nozzle.exit.rr / self.nozzle.exit.rr[-1]
        )
        self.jet_velocity_ratio = (
            utils.mass_flow_function(self.M_1) * utils.velocity_function(self.scenario.M)
            / (
                2 * utils.gamma * self.nozzle_area_ratio * utils.cumulative_trapezoid(
                    self.nozzle.exit.rr / self.nozzle.exit.rr[-1], rho_v_2_r
                )[-1]
            )
        )

        # calculate compressor (isentropic) efficiency
        h_03s_h_01_r = (
            self.nozzle.exit.p * self.nozzle.exit.M / np.sqrt(self.nozzle.exit.T)
            * (np.power(self.nozzle.exit.p_0, (utils.gamma - 1) / utils.gamma) - 1)
            * self.nozzle.exit.rr
        )
        h_03_h_01_r = (
            self.nozzle.exit.p * self.nozzle.exit.M / np.sqrt(self.nozzle.exit.T)
            * (self.nozzle.exit.T_0 - 1) * self.nozzle.exit.rr
        )
        self.eta_comp = (
            utils.cumulative_trapezoid(self.nozzle.exit.rr, h_03s_h_01_r)[-1]
            / utils.cumulative_trapezoid(self.nozzle.exit.rr, h_03_h_01_r)[-1]
        )

        # calculate nozzle efficiency
        rho_v_3_r = (
            self.nozzle.exit.p * self.nozzle.exit.M**3 * np.sqrt(self.nozzle.exit.T)
            * self.nozzle.exit.rr / self.nozzle.exit.rr[-1]
        )
        self.eta_nozz = (
            (
                2 * utils.gamma * (utils.gamma - 1) * self.nozzle_area_ratio
                * utils.cumulative_trapezoid(
                    self.nozzle.exit.rr / self.nozzle.exit.rr[-1], rho_v_3_r
                )[-1]
                - np.sqrt(utils.gamma - 1) * utils.mass_flow_function(self.M_1)
                * utils.velocity_function(self.scenario.M)**2
            ) / (
                4 * utils.gamma * self.nozzle_area_ratio * utils.cumulative_trapezoid(
                    self.nozzle.exit.rr / self.nozzle.exit.rr[-1],
                    h_03s_h_01_r / self.nozzle.exit.rr[-1]
                )[-1]
            )
        )

        # calculate propulsive efficiency
        self.eta_prop = (
            2 * self.C_th * utils.velocity_function(self.scenario.M)
            / ( 
                2 * self.nozzle_area_ratio * utils.gamma * np.sqrt(utils.gamma - 1)
                * utils.cumulative_trapezoid(
                    self.nozzle.exit.rr / self.nozzle.exit.rr[-1], rho_v_3_r
                )[-1]
                - utils.mass_flow_function(self.M_1) * utils.velocity_function(self.scenario.M)**2
            )
        )

        # calculate overall efficiency
        self.eta_elec = 1
        self.eta_overall = (
            0.5 * np.sqrt(utils.gamma - 1) / utils.gamma * self.C_th
            * utils.velocity_function(self.scenario.M) / (
                self.nozzle_area_ratio * utils.cumulative_trapezoid(
                    self.nozzle.exit.rr / self.nozzle.exit.rr[-1],
                    h_03_h_01_r / self.nozzle.exit.rr[-1]
                )[-1]
            )
        )

        # iterate over all stages
        for stage in self.stages:

            # investigate stage performance
            stage.evaluate()

    def empirical_design(self):
        """Determines the actual geometry of the engine."""
        # store relevant geometry parameters separately for convenience
        aspect_ratio = self.geometry["aspect_ratio"]
        diffusion_factor = self.geometry["diffusion_factor"]
        deviation_constant = self.geometry["deviation_constant"]

        # loop over all blade rows
        for blade_row in self.blade_rows:

            # calculate chord distribution and deviation
            blade_row.calculate_chord(aspect_ratio, diffusion_factor)
            blade_row.calculate_deviation(deviation_constant)

        # store list containing number of blades for each row
        self.geometry["no_of_blades"] = [blade_row.no_of_blades for blade_row in self.blade_rows]

    def calculate_off_design(self):
        """Calculates the off-design performance characterstics of each stage."""
        # loop over each stage
        for stage in self.stages:

            # for different values of flow coefficient between the given bounds
            phi_min = self.off_design["phi_min"]
            phi_max = self.off_design["phi_max"]
            for phi in [phi_min, phi_max]:

                # calculate off-design performance for stage
                stage.calculate_off_design(self.hub_tip_ratio, phi)
        
    def dimensional_values(self):
        """Convert some values back to dimensional values for export to CFD."""
        # get mean of mean-line chords for each blade row
        chords = [np.interp(
            0.5 * (blade_row.exit.rr[0] + blade_row.exit.rr[-1]),
            blade_row.exit.rr, blade_row.exit.axial_chord
        ) for blade_row in self.blade_rows]
        nominal_chord = np.mean(chords)

        # convert to mm
        chord_spacing = 4
        nominal_chord_mm = chord_spacing * nominal_chord * self.diameter / 2

        # space blade rows one nominal chord apart
        #self.xx = np.linspace(-0.1, len(self.blade_rows) + 0.1, utils.Defaults.export_grid)
        xx = nominal_chord * np.repeat(np.arange(len(self.blade_rows)), 2)

        # add tip axial chord to the second x-element corresponding to each blade row
        xx[1::2] += np.array([
            blade_row.exit.axial_chord[-1] for blade_row in self.blade_rows
        ])

        # fit spline through casing and hub radii
        r_casing_spline = make_interp_spline(
            xx,
            #[blade_row.exit.rr[-1] for blade_row in self.blade_rows],
            [
                x.rr[-1] for blade_row in self.blade_rows
                for x in (blade_row.inlet, blade_row.exit)
            ],
            k = 3,
            bc_type = "clamped"
        )
        r_hub_spline = make_interp_spline(
            xx,
            [
                x.rr[0] for blade_row in self.blade_rows
                for x in (blade_row.inlet, blade_row.exit)
            ],
            k = 3,
            bc_type = "clamped"
        )

        xx_mm = np.linspace(-0.1, len(self.blade_rows) + 0.1, utils.Defaults.export_grid)
        r_casing_mm = r_casing_spline(xx_mm)
        r_hub_mm = r_hub_spline(xx_mm)

        print(f"xx_mm: {xx_mm}")
        print(f"r_casing_mm: {r_casing_mm}")
        print(f"r_hub_mm: {r_hub_mm}")

    def export(self):
        """Exports the engine's parameters as a .mat file for CFD."""
        # store variable for calculating blade x-coordinates
        x_ref = 0

        # create empty dictionaries
        self.export_dictionary = {}
        self.blades_dictionary = {}

        # determine dimensionless span distribution to export data across
        span = np.linspace(-0.05, 1.05, utils.Defaults.export_grid)

        # loop over all blade rows
        for index, blade_row in enumerate(self.blade_rows):

            # interpolate over blade row inlet span to get inlet metal angle distribution
            blade_row.inlet.span = (
                (blade_row.inlet.rr - blade_row.inlet.rr[0])
                / (blade_row.inlet.rr[-1] - blade_row.inlet.rr[0])
            )
            inlet_metal_interp = interp1d(
                blade_row.inlet.span,
                utils.rad_to_deg(blade_row.inlet.metal_angle),
                kind = "linear",
                bounds_error = False,
                fill_value = "extrapolate"
            )
            inlet_metal_angle = inlet_metal_interp(span)

            # interpolate over blade row exit span to get exit metal angle distribution
            blade_row.exit.span = (
                (blade_row.exit.rr - blade_row.exit.rr[0])
                / (blade_row.exit.rr[-1] - blade_row.exit.rr[0])
            )
            exit_metal_interp = interp1d(
                blade_row.exit.span,
                utils.rad_to_deg(blade_row.exit.metal_angle),
                kind = "linear",
                bounds_error = False,
                fill_value = "extrapolate"
            )
            exit_metal_angle = exit_metal_interp(span)

            # interpolate over blade row exit span to get chord distribution
            chord_interp = interp1d(
                blade_row.exit.span,
                self.diameter / 2 * blade_row.exit.chord,
                kind = "linear",
                bounds_error = False,
                fill_value = "extrapolate"
            )
            chord = chord_interp(span)

            # calculate blade x-coordinate
            x_ref += 2 * max(chord)

            # store inlet and exit midspan radii
            inlet_radius = self.diameter / 2 * (blade_row.inlet.rr[0] + blade_row.inlet.rr[-1]) / 2
            exit_radius = self.diameter / 2 * (blade_row.exit.rr[0] + blade_row.exit.rr[-1]) / 2

            # store inlet and exit areas
            inlet_area = (
                np.pi * (self.diameter / 2)**2
                * (blade_row.inlet.rr[-1]**2 - self.hub_tip_ratio**2)
            )
            exit_area = (
                np.pi * (self.diameter / 2)**2
                * (blade_row.exit.rr[-1]**2 - self.hub_tip_ratio**2)
            )

            # calculate blade rpm
            if isinstance(blade_row, Rotor):

                rpm = (
                    blade_row.inlet.M_blade[-1] * np.sqrt(blade_row.inlet.T[-1]) * np.sqrt(
                        utils.gamma * utils.R * self.scenario.T_0
                    ) / (0.5 * self.diameter)
                )
                rpm = utils.rad_s_to_rpm(rpm)

            else:

                rpm = 0

            v_x = (
                0.5 * (blade_row.inlet.v_x[0] + blade_row.inlet.v_x[-1])
                * np.sqrt(utils.gamma * utils.R * self.scenario.T_0)
            )

            print(f"self.scenario.T_0: {self.scenario.T_0}")
            print(f"blade_row.inlet.M_blade[-1] * np.sqrt(blade_row.inlet.T[-1]): {blade_row.inlet.M_blade[-1] * np.sqrt(blade_row.inlet.T[-1])}")
            print(f"rpm: {rpm}")

            # create nested dictionary corresponding to blade row data
            self.blades_dictionary[f"blade_{index + 1}"] = {
                "span": span.tolist(),
                "inlet_metal_angle": inlet_metal_angle.tolist(),
                "exit_metal_angle": exit_metal_angle.tolist(),
                "chord": chord.tolist(),
                "x_ref": x_ref,
                "inlet_radius": inlet_radius,
                "exit_radius": exit_radius,
                "inlet_area": inlet_area,
                "exit_area": exit_area,
                "area_ratio": exit_area / inlet_area,
                "rpm": rpm,
                "v_x": v_x,
                "no_of_blades": blade_row.no_of_blades
            }

        # add blades dictionary to export dictionary
        self.export_dictionary["blades"] = self.blades_dictionary

        # add metadata to dictionary
        self.export_dictionary["metadata"] = {
            # export data-time information
            "export_date": datetime.now().strftime("%Y-%m-%d"),
            "export_time": datetime.now().strftime("%H:%M:%S"),
            "export_timestamp": datetime.now().isoformat(),

            # export flight scenario details
            "flight_mach_number": self.M_flight,
            "thrust_coefficient": self.C_th,
            "hub_tip_ratio": self.hub_tip_ratio,

            # export original input information
            "no_of_stages": self.no_of_stages,
            "vortex_exponent": self.vortex_exponent,
            "solver_order": self.solver_order,
            "pressure_loss_coefficient": self.Y_p,
            "flow_coefficient": self.phi,
            "stage_loading_coefficient": self.psi,

            # export geometry information
            "aspect_ratio": self.geometry["aspect_ratio"],
            "diffusion_factor": self.geometry["diffusion_factor"],
            "deviation_constant": self.geometry["deviation_constant"],

            # export solver outputs
            "inlet_mach_number": self.M_1,
            "nozzle_area_ratio": self.nozzle_area_ratio,

            # export solver details
            "solver_grid_points": utils.Defaults.solver_grid,
            "export_grid_points": utils.Defaults.export_grid
        }

        # save dictionary as .mat file
        filename = (
            f"high_speed_solver_export_"
            f"{self.export_dictionary['metadata']['export_timestamp'].replace(':', '-')}.mat"
        )
        savemat(filename, self.export_dictionary)
        
        # save dictionary as JSON file
        json_filename = (
            f"high_speed_solver_export_"
            f"{self.export_dictionary['metadata']['export_timestamp'].replace(':', '-')}.json"
        )
        with open(json_filename, 'w') as f:
            json.dump(self.export_dictionary, f, indent = 2)

        # print feedback to user
        print(
            f"Engine data successfully exported as "
            f"{utils.Colours.GREEN}{filename}{utils.Colours.END} and "
            f"{utils.Colours.GREEN}{json_filename}{utils.Colours.END}!"
        )

# plotting functions ------------------------------------------------------------------------------

    def plot_velocity_triangles(self):
        """Plots the Mach triangles and approximate blade shapes for the compressor."""
        # create plot for displaying velocity triangles
        fig, ax = plt.subplots(figsize = (10, 6))

        scaling = 1

        # coordinates for a brace
        verts = [
            (-0.15, -0.30),
            (-0.05, -0.30),
            (-0.05, -0.125),
            (-0.05, 0.00),
            (0.00, 0.00),
            (-0.05, 0.00),
            (-0.05, 0.125),
            (-0.05, 0.30),
            (-0.15, 0.30)
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3
        ]

        # labels for plotting
        labels = ["HUB", "MID-SPAN", "TIP"]

        # array of spanwise position indices to plot
        indices = [0, int(np.floor(utils.Defaults.solver_grid / 2)), -1]

        # set legend labels
        ax.plot([], [], color = 'C0', label = 'Absolute Mach number')
        ax.plot([], [], color = 'C4', label = 'Relative Mach number')
        ax.plot([], [], color = 'C3', label = 'Blade Mach number')

        # configure plot
        ax.set_aspect('equal')
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])

        # set axis limits
        ax.set_xlim(-0.4, len(self.blade_rows) + 0.7)
        ax.set_ylim(-0.7, len(indices) - 0.2)

        # set title
        ax.text(
            0.5, 1.02,
            f"$ C_\\thorn $ = {self.C_th:.3g}, $ M_\\infty $ = {self.M_flight:.3g}, "
            f"$ \\phi $ = {self.phi:.3g}, $ \\psi $ = {self.psi:.3g}, n = {self.vortex_exponent:.3g}",
            transform = ax.transAxes,
            ha = 'center',
            va = 'bottom',
            fontsize = 16
        )

        # tight layout
        plt.tight_layout()

        # iterate over all inlet and exit streamtubes chosen for plotting
        for row, index in enumerate(indices):
                
            # iterate over all blade rows
            for column, blade_row in enumerate(self.blade_rows):

                # plot blade row at chosen spanwise positions
                blade_row.draw_blades()
                ax.plot(blade_row.xx[row] + column, blade_row.yy[row] + row, color = 'k')

                # add velocity triangles
                x_te = blade_row.xx[row][0]
                y_te = blade_row.yy[row][0]

                # annotate inlet absolute velocity vector
                ax.annotate(
                    "",
                    xy = (column, row),
                    xytext = (
                        column - blade_row.inlet.v_x[index],
                        row - blade_row.inlet.v_theta[index]
                    ),
                    arrowprops = dict(
                        arrowstyle = "->", color = 'C0',
                        shrinkA = 0, shrinkB = 0, lw = 1.5
                    )
                )
                
                # annotate exit absolute velocity vector
                ax.annotate(
                    "",
                    xy = (
                        column + x_te + blade_row.exit.v_x[index],
                        row + y_te + blade_row.exit.v_theta[index]
                    ),
                    xytext = (column + x_te, row + y_te),
                    arrowprops = dict(
                        arrowstyle = "->", color = 'C0',
                        shrinkA = 0, shrinkB = 0, lw = 1.5
                    )
                )

                if hasattr(blade_row.exit, "M_rel"):
                
                    # annotate inlet relative velocity vector
                    ax.annotate(
                        "",
                        xy = (column, row),
                        xytext = (
                            column - blade_row.inlet.v_x[index],
                            row - blade_row.inlet.v_theta_rel[index]
                        ),
                        arrowprops = dict(
                            arrowstyle = "->", color = 'C4',
                            shrinkA = 0, shrinkB = 0, lw = 1.5
                        )
                    )
                    
                    # annotate exit relative velocity vector
                    ax.annotate(
                        "",
                        xy = (
                            column + x_te + blade_row.exit.v_x[index],
                            row + y_te + blade_row.exit.v_theta_rel[index]
                        ),
                        xytext = (column + x_te, row + y_te),
                        arrowprops = dict(
                            arrowstyle = "->", color = 'C4',
                            shrinkA = 0, shrinkB = 0, lw = 1.5
                        )
                    )
                    
                    # annotate inlet blade velocity vector
                    ax.annotate(
                        "",
                        xy = (
                            column - blade_row.inlet.v_x[index],
                            row - blade_row.inlet.v_theta_rel[index]
                        ),
                        xytext = (
                            column - blade_row.inlet.v_x[index],
                            row - blade_row.inlet.v_theta[index]
                        ),
                        arrowprops = dict(
                            arrowstyle = "->", color = 'C3',
                            shrinkA = 0, shrinkB = 0, lw = 1.5
                        )
                    )
                    
                    # annotate exit blade velocity vector
                    ax.annotate(
                        "",
                        xy = (
                            column + x_te + blade_row.exit.v_x[index],
                            row + y_te + blade_row.exit.v_theta[index]
                        ),
                        xytext = (
                            column + x_te + blade_row.exit.v_x[index],
                            row + y_te + blade_row.exit.v_theta_rel[index]
                        ),
                        arrowprops = dict(
                            arrowstyle = "->", color = 'C3',
                            shrinkA = 0, shrinkB = 0, lw = 1.5
                        )
                    )

            # add brace to plot
            dx = 2 * len(self.stages)
            dy = row - 0.1
            brace = patches.PathPatch(
                Path([(x + dx, y + dy) for x, y in verts], codes),
                fill = False, linewidth = 1
            )
            ax.add_patch(brace)
            ax.text(
                dx + 0.15, dy,
                f"{labels[row]}",
                ha = "left", va = "center"
            )

        plt.show()

    def plot_spanwise(self, quantities = utils.Defaults.quantity_list):
        """Plots the spanwise variation of flow angle at each axial position."""
        # loop over input quantites
        for quantity_list in quantities:

            # reshape array of quantity-label pairs
            quantity_labels = np.reshape(quantity_list, (-1, 2))

            # create plot with an axis for each blade row inlet and exit and reshape axes
            fig, axes = plt.subplots(ncols = len(self.blade_rows) + 2, figsize = (14, 6))

            # assign values for capturing appropriate axis limits
            x_min = 1e12
            x_max = -1e12

            # loop over all quuantity-label pairs
            for quantity_label in quantity_labels:
    
                # store as intermediate variables for convenience
                quantity = quantity_label[0]
                label = quantity_label[1]

                # set colour
                colour = next(self.colour_cycle)

                # set legend entry in final axis
                axes[-1].plot([], [], color = colour, label = label)

                # check if blade row has quantity stored
                if hasattr(self.blade_rows[0].inlet, quantity):

                    x = getattr(self.blade_rows[0].inlet, quantity)
                    value = x.value if hasattr(x, "value") else x
                    value = utils.rad_to_deg(value) if "°" in label else value

                    # plot rotor inlet conditions
                    span = (
                        (self.blade_rows[0].inlet.rr - self.blade_rows[0].inlet.rr[0])
                        / (self.blade_rows[0].inlet.rr[-1] - self.blade_rows[0].inlet.rr[0])
                    )
                    axes[0].plot(value, span, alpha = 0.5, linewidth = 3, color = colour)

                    # update x-axis limits
                    x_min = min(x_min, *value)
                    x_max = max(x_max, *value)

                # iterate over all axes:
                for ax, blade_row in zip(axes[1:], self.blade_rows + [self.nozzle]):

                    # check if blade row has quantity stored
                    if hasattr(blade_row.exit, quantity):

                        # store values as intermediate variable
                        x = getattr(blade_row.exit, quantity)
                        value = x.value if hasattr(x, "value") else x
                        value = utils.rad_to_deg(value) if "°" in label else value

                        # plot blade row exit conditions
                        span = (
                            (blade_row.exit.rr - blade_row.exit.rr[0])
                            / (blade_row.exit.rr[-1] - blade_row.exit.rr[0])
                        )
                        ax.plot(value, span, alpha = 0.5, linewidth = 3, color = colour)

                        # update x-axis limits
                        x_min = min(x_min, *value)
                        x_max = max(x_max, *value)

            # loop over all axes in the figure
            for ax in axes:

                # set axis x- and y-limits
                ax.set_xlim(x_min - (x_max - x_min) / 10, x_max + (x_max - x_min) / 10)
                ax.set_ylim(0, 1)

                # set grid and maximum number pf x-ticks
                ax.grid()
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbins = 2))

            # set y-label and tight layout
            axes[0].set_ylabel('Dimensionless span')
            plt.tight_layout()

            # create legend
            plt.subplots_adjust(bottom = 0.16)
            axes[-1].legend(
                loc = 'center', bbox_to_anchor = (0.5, 0.05), bbox_transform = fig.transFigure
            )
        
        # show plots
        plt.show()
    
    def plot_section(self):
        """Plots a section view through the engine highlighting streamline behaviour."""
        # create plot
        fig, ax = plt.subplots(figsize = (14, 6))

        # initialise empty list to hold x- and y-values
        xx = [0, 1]
        yy = []

        # store rotor inlet values
        yy.append(self.blade_rows[0].inlet.rr)

        # loop over all blade rows
        for index, blade_row in enumerate(self.blade_rows):

            xx.append(index + 2)
            yy.append(blade_row.exit.rr)

        # store nozzle exit values
        yy.append(self.nozzle.exit.rr)

        # transpose y-array
        yy = np.transpose(yy)

        # plot each streamline separately
        for streamline in yy:

            # plot streamline
            ax.plot(xx, streamline, linewidth = 1, color = 'k')

        plt.show()

    def plot_matlab(self):
        """Creates a spanwise flow angle plot with matlab results overlaid."""
        # run matlab engine
        eng = matlab.engine.start_matlab()
        eng.eval("set(0, 'DefaultFigureVisible', 'off')", nargout=0)

        # path to matlab_folder
        python_dir = FilePath(__file__).resolve().parent
        parent_dir = python_dir.parent
        matlab_dir = parent_dir / "forSlava"

        # add MATLAB folder to MATLAB search path
        eng.addpath(str(matlab_dir), nargout = 0)

        # create dummy buffers
        out = io.StringIO()
        err = io.StringIO()

        # run script rejecting terminal outputs
        eng.run("DuctedFanDesign_220503", nargout = 0, stdout = out, stderr = err)

        # extract information
        a = eng.workspace['a']
        m_dot = eng.workspace['mdot']
        rho = eng.workspace['ro']

        # convert to numpy arrays
        alpha = np.array(a['alpha'], dtype = float)
        v_x = np.array(a['vx'], dtype = float)
        m_dot = np.array(m_dot, dtype = float)
        rho = np.array(rho, dtype = float)[0][0]

        # shut down engine
        eng.quit()

        # create array of x-values on matlab grid and solver grid
        xx_matlab = np.linspace(0, 1, alpha.shape[0])
        xx = np.linspace(0, 1, utils.Defaults.solver_grid)

        # get array of radii from array of spans
        rr = self.hub_tip_ratio + (1 - self.hub_tip_ratio) * xx

        # store matlab flow angles
        self.blade_rows[0].inlet.matlab_rel_angle = np.interp(xx, xx_matlab, alpha[:, 0, 1])
        self.blade_rows[0].exit.matlab_rel_angle = np.interp(xx, xx_matlab, alpha[:, 1, 1])
        self.blade_rows[0].exit.matlab_angle = np.interp(xx, xx_matlab, alpha[:, 1, 2])

        # store matlab axial velocity
        self.blade_rows[0].inlet.matlab_v_x = np.interp(xx, xx_matlab, v_x[:, 0, 0])
        self.blade_rows[0].exit.matlab_v_x = np.interp(xx, xx_matlab, v_x[:, 1, 0])
        self.blade_rows[1].exit.matlab_v_x = np.interp(xx, xx_matlab, v_x[:, 2, 0])

        # loop over all blade rows
        for blade_row in self.blade_rows:

            # calculate cumulative (dimensional) mass flow rate
            dm_dr = 2 * rho * blade_row.exit.matlab_v_x * rr * self.scenario.A / (1 - self.hub_tip_ratio**2)
            blade_row.exit.m_dot_matlab = utils.cumulative_trapezoid(dm_dr, rr)
            blade_row.exit.m_dot_kg_s = (
                blade_row.exit.m_dot * self.scenario.A * self.scenario.p_0 / np.sqrt(utils.c_p * self.scenario.T_0)
            )

            # calculate dimensional density variation
            blade_row.exit.rho_kg_m_3 = (
                blade_row.exit.p / (utils.R * blade_row.exit.T) * self.scenario.p_0 / self.scenario.T_0
            )
            blade_row.exit.rho_matlab = rho * np.ones(utils.Defaults.solver_grid)

            # calculate solver flow velocity in m_s
            blade_row.exit.v_x_m_s = blade_row.exit.v_x * np.sqrt(utils.gamma * utils.R * self.scenario.T_0)

        # calculate cumulative (dimensional) mass flow rate at inlet
        dm_dr = 2 * rho * self.blade_rows[0].inlet.matlab_v_x * rr * self.scenario.A / (1 - self.hub_tip_ratio**2)
        self.blade_rows[0].inlet.m_dot_matlab = utils.cumulative_trapezoid(dm_dr, rr)
        self.blade_rows[0].inlet.m_dot_kg_s = (
            self.blade_rows[0].inlet.m_dot * self.scenario.A * self.scenario.p_0 / np.sqrt(utils.c_p * self.scenario.T_0)
        )

        # calculate dimensional density variation at inlet
        self.blade_rows[0].inlet.rho_kg_m_3 = (
            self.blade_rows[0].inlet.p / (utils.R * self.blade_rows[0].inlet.T) * self.scenario.p_0 / self.scenario.T_0
        )

        # calculate solver flow velocity in m/s at inlet
        self.blade_rows[0].inlet.v_x_m_s = (
            self.blade_rows[0].inlet.v_x * np.sqrt(utils.gamma * utils.R * self.scenario.T_0)
        )

        # define quantities to plot
        quantities = [
            [
                'matlab_angle', 'MatLab absolute flow angle',
                'matlab_rel_angle', 'MatLab relative flow angle',
                'alpha', 'Absolute flow angle (°)',
                'beta', 'Relative flow angle (°)'
            ],
            [
                'matlab_v_x', 'MatLab axial flow velocity (m/s)',
                'v_x_m_s', 'Axial flow velocity (m/s)'
            ],
            [
                'm_dot_matlab', 'MatLab cumulative mass flow rate (kg/s)',
                'm_dot_kg_s', 'Cumulative mass flow rate (kg/s)'
            ],
            [
                'rho_matlab', 'MatLab density (kg/m^3)',
                'rho_kg_m_3', 'Density (kg/m^3)'
            ]
        ]

        # plot spanwise variation
        self.plot_spanwise(quantities)

    def plot_convergence(self):
        """Plots the thrust calculation convergence history for the engine."""
        # create plot
        fig, ax = plt.subplots(figsize = (10, 6))

        # plot initial guess curve
        ax.plot(self.M_1_initial, self.C_th_initial, label = "Initial", color = 'C0')

        # plot actual guesses as individual datapoints
        ax.plot(
            self.M_1_guesses, self.C_th_guesses, linestyle = '', marker = '.', markersize = 8,
            label = "Guesses", color = 'C3'
        )

        # plot target thrust
        ax.axhline(self.C_th_design, label = "Design", color = 'C4')

        # set axis limits
        x_min = np.min(self.M_1_guesses)
        x_max = np.max(self.M_1_guesses)
        y_min = np.min(self.C_th_guesses)
        y_max = np.max(self.C_th_guesses)
        margin = 0.5
        ax.set_xlim(x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min))
        ax.set_ylim(y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min))

        # configure and show plot
        ax.set_xlabel("Inlet Mach number")
        ax.set_ylabel("Thrust Coefficient")
        ax.grid()
        ax.legend()
        plt.show()

    def plot_off_design(self):
        """"""
        pass

# obsolete functions
    
    def old_design(self):
        """Determines appropriate values for blade metal angles for the given requirements."""
        # iterate over all stages
        print(f"\nself.M_1: {self.M_1}")
        for index, stage in enumerate(self.stages):

            # handle all other stages
            if index != 0:

                # set stage inlet to previous stage exit conditions
                stage.rotor.inlet = copy.deepcopy(self.stages[index - 1].blade_rows[1].exit)

            # define rotor blade geometry
            t1 = timer()
            stage.rotor.design(stage.phi, stage.psi, self.vortex_exponent)
            t2 = timer()
            utils.debug(f"Rotor design duration: {utils.Colours.GREEN}{t2 - t1:.3g}s{utils.Colours.END}")

            # stator inlet conditions are rotor exit conditions
            stage.stator.inlet = copy.deepcopy(stage.rotor.exit)

            # define stator blade geometry
            t1 = timer()
            stage.stator.design()
            t2 = timer()
            utils.debug(f"Stator design duration: {utils.Colours.GREEN}{t2 - t1:.3g}s{utils.Colours.END}")

        # nozzle inlet conditions are final stator exit conditions
        self.nozzle.inlet = copy.deepcopy(self.blade_rows[-1].exit)

        # design nozzle to match atmospheric pressure in jet periphery
        self.p_atm = utils.stagnation_pressure_ratio(self.M_flight)
        t1 = timer()
        self.nozzle.design(self.p_atm)
        t2 = timer()
        utils.debug(f"Nozzle design duration: {utils.Colours.GREEN}{t2 - t1:.3g}s{utils.Colours.END}")

        # evaluate engine performance
        t1 = timer()
        self.evaluate()
        t2 = timer()
        utils.debug(f"Engine evaluation duration: {utils.Colours.GREEN}{t2 - t1:.3g}s{utils.Colours.END}")

    def old_velocity_triangles(self):
        """Function to plot the velocity triangles and pressure and temperature distributions."""
        # determine factor to scale Mach triangles by
        scaling = 1 / (4 * self.M_1)

        # coordinates for a brace
        verts = [
            (-0.15, -0.30),
            (-0.05, -0.30),
            (-0.05, -0.125),
            (-0.05, 0.00),
            (0.00, 0.00),
            (-0.05, 0.00),
            (-0.05, 0.125),
            (-0.05, 0.30),
            (-0.15, 0.30)
        ]

        codes = [
            Path.MOVETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3
        ]

        # labels for plotting
        labels = ["MID-SPAN"]

        # initialise array of spanwise position indices to plot
        spanwise_positions = [0]

        # if more than 2 streamtubes exist
        if len(self.blade_rows[0].inlet) > 2:

            # plot central streamtube
            spanwise_positions.append(int(np.floor(len(self.blade_rows[0].inlet) / 2)))
            labels = ["HUB"] + labels

        # if more than 1 streamtubes exist
        if len(self.blade_rows[0].inlet) > 1:

            # plot outer-most streamtube
            spanwise_positions.append(-1)
            labels.append("TIP")

        # create plot for displaying velocity triangles
        fig, ax = plt.subplots(figsize = (11, 6))

        # set legend labels
        ax.plot([], [], color = 'C0', label = 'Absolute Mach number')
        ax.plot([], [], color = 'C4', label = 'Relative Mach number')
        ax.plot([], [], color = 'C3', label = 'Blade Mach number')

        # configure plot
        ax.set_aspect('equal')
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])

        # set axis limits
        ax.set_xlim(-0.4, len(self.blade_rows) + 0.7)
        ax.set_ylim(-0.7, len(spanwise_positions) - 0.2)

        # set title
        ax.text(
            0.5, 1.02,
            f"$ C_\\thorn $ = {self.C_th:.3g}, $ M_\\infty $ = {self.M_flight:.3g}, "
            f"$ \\phi $ = {self.stages[0].phi:.3g}, $ \\psi $ = {self.stages[0].psi:.3g}, n = {self.vortex_exponent:.3g}",
            transform = ax.transAxes,
            ha = 'center',
            va = 'bottom',
            fontsize = 12
        )

        # tight layout
        plt.tight_layout()

        # save figure
        directory = "figures"
        filename = f"{inspect.currentframe().f_code.co_name}"
        path = os.path.join(directory, filename)
        plt.savefig(path, dpi = 300)

        # iterate over all inlet and exit streamtubes chosen for plotting
        for j, k in enumerate(spanwise_positions):
                
            # iterate over all blade rows
            for index, blade_row in enumerate(self.blade_rows):

                # plot blade row at chosen spanwise positions
                blade_row.plot_blade_row(ax, index, j, k, scaling)

                # add brace to plot
                dx = 2 * len(self.stages)
                dy = j - 0.1
                brace = patches.PathPatch(
                    Path([(x + dx, y + dy) for x, y in verts], codes),
                    fill = False, linewidth = 1
                )
                ax.add_patch(brace)
                ax.text(
                    dx + 0.15, dy,
                    f"{labels[j]}",
                    ha = "left", va = "center"
                )

            # save figure
            directory = "figures"
            filename = f"{inspect.currentframe().f_code.co_name}_{j}"
            path = os.path.join(directory, filename)
            plt.savefig(path, dpi = 300)

    def old_contours(self):
        """Creates a plot of a section view of the engine with contours of a specified quantity."""
        # plotting parameters
        alpha = 0.5
        rotor_tip_span = 0.98

        # create a plot
        fig, ax = plt.subplots(figsize = (11, 6))

        # add axis labels
        ax.set_xlabel('x-axis')
        ax.set_ylabel('Dimensionless radius')

        # remove x-ticks
        ax.set_xticks([])

        # set axis limits and aspect ratio
        margin = 0.1
        ax.set_xlim(-margin, len(self.blade_rows) + 0.5 + margin)
        ax.set_ylim(-margin, 1 + margin)
        ax.set_aspect('equal')

        # set title
        ax.text(
            0.5, 1.02,
            f"$ C_\\thorn $ = {self.C_th:.3g}, $ M_\\infty $ = {self.M_flight:.3g}, "
            f"$ \\phi $ = {self.stages[0].phi:.3g}, $ \\psi $ = {self.stages[0].psi:.3g}, n = {self.vortex_exponent:.3g}",
            transform = ax.transAxes,
            ha = 'center',
            va = 'bottom',
            fontsize = 12
        )

        # tight layout
        plt.tight_layout()

        # save figure
        directory = "figures"
        filename = f"{inspect.currentframe().f_code.co_name}"
        path = os.path.join(directory, filename)
        plt.savefig(path, dpi = 300)

        # initialise fine array of x-values
        xx = np.linspace(0, len(self.blade_rows) - 0.5, 200)
        yy = np.linspace(xx[-1], xx[-1] + 1, 50)

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

            # get list of rotor hub and tip x-coordinates
            x_hub = np.linspace(
                rotor.x - 0.25 * min(rotor.exit[0].c, 1),
                rotor.x + 0.25 * min(rotor.exit[0].c, 1), 100
            )
            x_tip = np.linspace(
                rotor.x - 0.25 * min(rotor.exit[-1].c, 1),
                rotor.x + 0.25 * min(rotor.exit[-1].c, 1), 100
            )

            # create array of rotor vertex positions
            vertices = np.column_stack([
                np.concatenate([x_tip, x_hub[::-1]]),
                np.concatenate([rotor_tip_span * spline(x_tip), np.full_like(x_hub, rotor.r_hub)])
            ])

            # create rotor polygon and add
            polygon = Polygon(
                vertices, closed = True,
                facecolor = (0.8392, 0.1529, 0.1569, alpha),
                edgecolor = (0.8392, 0.1529, 0.1569),
                linewidth = 2
            )
            ax.add_patch(polygon)

            # draw x-shape over rotor
            ax.plot([x_hub[0], x_tip[-1]], [rotor.r_hub, rotor_tip_span * spline(x_tip[-1])], color = 'C3')
            ax.plot([x_tip[0], x_hub[-1]], [rotor_tip_span * spline(x_tip[0]), rotor.r_hub], color = 'C3')

            # get list of stator hub and tip x-coordinates
            x_hub = np.linspace(
                stator.x - 0.25 * min(stator.exit[0].c, 1),
                stator.x + 0.25 * min(stator.exit[0].c, 1), 100
            )
            x_tip = np.linspace(
                stator.x - 0.25 * min(stator.exit[-1].c, 1),
                stator.x + 0.25 * min(stator.exit[-1].c, 1), 100
            )

            # create array of stator vertex positions
            vertices = np.column_stack([
                np.concatenate([x_tip, x_hub[::-1]]),
                np.concatenate([spline(x_tip), np.full_like(x_hub, rotor.r_hub)])
            ])

            # create stator polygon and add
            polygon = Polygon(
                vertices, closed = True,
                facecolor = (0.1216, 0.4667, 0.7059, alpha),
                edgecolor = (0.1216, 0.4667, 0.7059),
                linewidth = 2
            )
            ax.add_patch(polygon)

            # draw x-shape over stator
            ax.plot([x_hub[0], x_tip[-1]], [stator.r_hub, spline(x_tip[-1])], color = 'C0')
            ax.plot([x_tip[0], x_hub[-1]], [spline(x_tip[0]), stator.r_hub], color = 'C0')

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
        for index in range(len(self.blade_rows[0].inlet)):

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

            # save figure with lower bound of hub streamtube plotted only
            if index == 0:

                # save figure
                directory = "figures"
                filename = f"{inspect.currentframe().f_code.co_name}_0"
                path = os.path.join(directory, filename)
                plt.savefig(path, dpi = 300)

                # draw centreline
                ax.axhline(y = 0, linestyle = '--', color = 'k')
                ax.text(
                    x = len(self.stages) + 0.25, y = 0.03, s = "Centreline",
                    ha = 'center', va = 'center'
                )

                # save figure
                directory = "figures"
                filename = f"{inspect.currentframe().f_code.co_name}_1"
                path = os.path.join(directory, filename)
                plt.savefig(path, dpi = 300)

                # draw inlet plane
                ax.axvline(x = 0, linestyle = '--', color = 'C1')
                ax.text(
                    x = 0.03, y = self.blade_rows[0].r_hub / 4,
                    s = "Compressor inlet plane", color = 'C1',
                    ha = 'left', va = 'center'
                )

                # save figure
                directory = "figures"
                filename = f"{inspect.currentframe().f_code.co_name}_2"
                path = os.path.join(directory, filename)
                plt.savefig(path, dpi = 300)

                # draw exit plane
                ax.axvline(x = len(self.blade_rows) + 0.5, linestyle = '--', color = 'C1')
                ax.text(
                    x = len(self.blade_rows) + 0.5 - 0.03, y = 1,
                    s = "Nozzle exit plane", color = 'C1',
                    ha = 'right', va = 'center'
                )

                # save figure
                directory = "figures"
                filename = f"{inspect.currentframe().f_code.co_name}_3"
                path = os.path.join(directory, filename)
                plt.savefig(path, dpi = 300)

        # save figure
        directory = "figures"
        filename = f"{inspect.currentframe().f_code.co_name}_4"
        path = os.path.join(directory, filename)
        plt.savefig(path, dpi = 300)

    def old_spanwise_variations(self, quantity_list = utils.Defaults.quantity_list):
        """Creates a plot of the spanwise variations of a specified quantity for each blade row."""
        # loop over all list entries
        for quantities in quantity_list:

            # separate input list into pairs of values
            q_label_bools = [quantities[i:i + 3] for i in range(0, len(quantities), 3)]

            # helper function to retrieve values from appropriate place and with correct units
            def get_attribute(in_out, q, label, bool):
                """Retrieves an attribute, handling cases where no attribute exists."""
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
                        span = (rr - rr[0]) / (rr[-1] - rr[0])

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
                            ax[index].plot(spline(rr), span, alpha = 0.5, linewidth = 3, color = colour)

                            # plot inlet conditions
                            ax[index].plot(
                                [get_attribute(in_out, q, label, bool) for in_out in inlet_outlet],
                                (np.array([in_out.r for in_out in inlet_outlet]) - rr[0])
                                / (rr[-1] - rr[0]),
                                linestyle = '', marker = '.', markersize = 8, color = colour
                            )

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
            axes[0].set_ylabel('Dimensionless span')
            plt.tight_layout()

            # set x-label
            plt.subplots_adjust(bottom = 0.16)

            # create legend
            ax.legend(loc='center', bbox_to_anchor=(0.5, 0.05), bbox_transform=fig.transFigure)

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
