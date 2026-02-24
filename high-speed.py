# import modules

import sys
from time import perf_counter as timer

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from engine import Engine
from flow_state import Flow_state
from flight_scenario import Flight_scenario
from output import Output
import utils

# main function

def main():
    """Function to run on script execution."""
    # visualise colour options for convenience
    """colours = utils.Colours()
    print(colours)"""

    # create a flight scenario with default parameters
    flight_scenario = Flight_scenario(
        "default",
        utils.Defaults.altitude,
        utils.Defaults.flight_speed,
        utils.Defaults.diameter,
        utils.Defaults.hub_tip_ratio,
        utils.Defaults.thrust
    )

    # create an engine with default parameters
    engine = Engine(
        flight_scenario,
        utils.Defaults.no_of_stages,
        utils.Defaults.vortex_exponent,
        utils.Defaults.solver_order,
        utils.Defaults.Y_p,
        utils.Defaults.phi,
        utils.Defaults.psi
    )
    print(engine)

    # add geometry
    engine.geometry = {
        "aspect_ratio": utils.Defaults.aspect_ratio,
        "diffusion_factor": utils.Defaults.diffusion_factor,
        "deviation_constant": utils.Defaults.deviation_constant
    }

    # calculate blade angles and export engine
    engine.empirical_design()
    engine.export()

    return

    if utils.Defaults.no_of_stages == None:

        # determine from user how many stages to construct the engine with
        print(f"{utils.Colours.RED}Please state the desired number of stages:{utils.Colours.END}")
        while True:

            user_input = input()
            try:

                no_of_stages = int(user_input)
                break

            except ValueError:

                print(f"{utils.Colours.RED}Error: Please provide a positive integer.{utils.Colours.END}")

    else:

        no_of_stages = utils.Defaults.no_of_stages

    # user feedback
    print(f"{utils.Colours.GREEN}{no_of_stages}{utils.Colours.END} stage(s) selected!")
    print(
        f"Analysing {utils.Colours.GREEN}{utils.Defaults.no_of_annuli}{utils.Colours.END} annular "
        f"streamtube(s)!"
    )

    # store list of conditions to design candidate engines for
    flight_scenarios = [
        Flight_scenario(
            label = "Cruise",
            altitude = 0,
            velocity = 60,
            diameter = utils.Defaults.engine_diameter,
            hub_tip_ratio = utils.Defaults.hub_tip_ratio,
            thrust = 30
        ),
        Flight_scenario(
            label = "Static",
            altitude =  0,
            velocity = 0,
            diameter = utils.Defaults.engine_diameter,
            hub_tip_ratio = utils.Defaults.hub_tip_ratio,
            thrust = 50
        ),
        Flight_scenario(
            label = "Take-off",
            altitude = 0,
            velocity = 20,
            diameter = utils.Defaults.engine_diameter,
            hub_tip_ratio = utils.Defaults.hub_tip_ratio,
            thrust = 30
        )
    ]

    # iterate over every flight scenario
    for scenario in flight_scenarios:

        print(f"Designing for scenario:{scenario}")

        scenario.x = []
        scenario.y = []

        scenario.engine = Engine(scenario)

        quantities = [
            [
                'M', 'Mach number', True,
                'M_rel', 'Relative Mach number', True
            ],
            #['T', 'Static temperature', True, 'T_0', 'Stagnation temperature', True],
            [
                'alpha', 'Flow angle (°)', True,
                'beta', 'Relative flow angle (°)', True,
                'metal_angle', 'Metal angle (°)', False
            ],
            #['rho', 'Density', True],
            #['m', 'Mass flow', False],
            ['M_x', 'Axial Mach number', False],
            [
                'pitch_to_chord', 'Pitch-to-chord ratio', False,
                's', 'Pitch', False,
                'c', 'Chord', False
            ],
            [
                'phi', 'Flow coefficient', False,
                'psi', 'Stage loading coefficient', False,
                'reaction', 'Reaction', False
            ],
            [
                'DF', 'Diffusion factor', False
            ]
        ]

        scenario.engine.plot_spanwise_variations(quantities)

        scenario.engine.plot_velocity_triangles()
        scenario.engine.plot_contours()
        plt.show()

    return

    # plot key parameters of the candidate engines
    fig, ax = plt.subplots()
    for scenario in flight_scenarios:
        
        areas = [engine.nozzle.area_ratio for engine in scenario.engines]
        ax.plot(
            areas, [engine.jet_velocity_ratio for engine in scenario.engines],
            color = scenario.colour, linestyle = '', marker = '.'
        )
        ax.plot(
            areas, [engine.C_th for engine in scenario.engines],
            color = scenario.colour, linestyle = '', marker = 'x'
        )
        #ax.plot(
        #    areas, [engine.eta_p for engine in scenario.engines],
        #    color = scenario.colour, linestyle = '', marker = 'v'
        #)
        ax.plot(
            areas, [engine.nozzle.exit.M for engine in scenario.engines],
            color = scenario.colour, linestyle = '', marker = 'v'
        )
        ax.plot([],[], color = scenario.colour, label = scenario.label)

    # add legend labels manually
    ax.plot([], [], linestyle = '', marker = '.', color  = 'k', label = "Jet velocity ratio")
    ax.plot([], [], linestyle = '', marker = 'x', color  = 'k', label = "Thrust coefficient")
    ax.plot([], [], linestyle = '', marker = 'v', color  = 'k', label = "Inlet Mach number")

    # plot configuration
    ax.set_xlabel("Area ratio")
    ax.grid()
    ax.legend()

    # leave room at bottom of plot for slider
    plt.subplots_adjust(bottom=0.25)

    # place slider below the plot and aligned with the x-axis
    axpos = ax.get_position()
    slider_height = 0.03
    slider_bottom = axpos.y0 - 0.2

    # add slider to plot
    slider_ax = fig.add_axes([
        axpos.x0,
        slider_bottom,
        axpos.width,
        slider_height
    ])

    # match the slider range to the x-axis limits
    x_min, x_max = ax.get_xlim()
    slider = Slider(slider_ax, "Area ratio:", x_min, x_max, valinit=x_min)

    # slider update function
    def update(val):

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

    # store the user input jet velocity ratio from slider
    area_ratio = slider.val

    # determine from user which of the candidate engines to analyse further
    """print(f"{utils.Colours.RED}Please state the desired jet velocity ratio:{utils.Colours.END}")
    while True:

        user_input = input()
        try:

            design_epsilon = float(user_input)
            break

        except ValueError:

            print(f"{utils.Colours.RED}Error: Please provide a valid jet velocity ratio.{utils.Colours.END}")"""

    # single out specified engine from list of candidate engines
    print(
        f"{utils.Colours.GREEN}Nozzle area ratio of {area_ratio:.4g} selected!{utils.Colours.END}"
    )
    jet_velocity_ratios = [engine.nozzle.area_ratio for engine in flight_scenarios[-1].engines]
    design_index = min(enumerate(jet_velocity_ratios), key=lambda x: abs(x[1] - area_ratio))[0]
    engine = flight_scenarios[-1].engines[design_index]

    # display default engine information to user
    print(engine)
    print(repr(engine))

    # determine from user whether or not to use the default blade row configurations
    print(
        f"{utils.Colours.RED}Would you like to accept the default blade row configurations? [y / n]"
        f"{utils.Colours.END}"
    )
    while True:

        user_input = input()
        if user_input == "y" or user_input == "n":
            
            break
        
        else:

            print(f"{utils.Colours.RED}Error: Please respond with [y / n].{utils.Colours.END}")

    # ask user which blade row they would like to make changes to
    if user_input == "n":

        while True:

            print(
                f"{utils.Colours.RED}Which blade row would you like to edit? "
                f"[1 - {len(engine.blade_rows)}]{utils.Colours.END}"
            )
            while True:

                user_input = input()
                try:

                    index = int(user_input)
                    break

                except ValueError:

                    print(
                        f"{utils.Colours.RED}Error: Please provide a positive integer. "
                        f"[1 - {len(engine.blade_rows)}]{utils.Colours.END}"
                    )

            # display blade row and call function to modify its properties
            print(f"\n[{index}] {engine.blade_rows[index]}")
            engine.blade_rows[index].modify_blade_row()

            print(repr(engine))
            print(f"{utils.Colours.RED}Would you like to edit another blade row? [y / n]{utils.Colours.END}")
            while True:

                user_input = input()
                if user_input == "y" or user_input == "n":
                    
                    break

                else:

                    print(f"{utils.Colours.RED}Error: Please respond with [y / n].{utils.Colours.END}")

            # exit loop if user has no further blade rows they would like to edit
            if user_input == "n":

                break

    # analyse engine
    print(f"{utils.Colours.GREEN}Analysing engine...{utils.Colours.END}")
    engine.collect_flow_states()
    engine.plot_velocity_triangles()
    engine.analyse()
    engine.collect_flow_states()
    engine.plot_velocity_triangles()
    plt.show()

    # store nominal flow and stage loading coefficients for each stage
    design_phi = [stage.phi for stage in engine.stages]
    design_psi = [stage.psi for stage in engine.stages]
    design_reaction = [stage.reaction for stage in engine.stages]

    # prepare to sweep over velocities and preallocate storage
    vv = np.linspace(utils.Defaults.M_min, utils.Defaults.M_max, utils.Defaults.N)
    n_stages = len(engine.stages)
    flow_coefficients = np.zeros((len(vv), n_stages))
    stage_loading_coefficients = np.zeros((len(vv), n_stages))
    reactions = np.zeros((len(vv), n_stages))
    jet_velocity_ratios = np.zeros((len(vv), 1))
    thrust_coefficients = np.zeros((len(vv), 1))

    # sweep over velocities and store flow and stage loading coefficients
    for i, M in enumerate(vv):

        engine.blade_rows[0].inlet = Flow_state(
            M,
            utils.Defaults.inlet_swirl, 1, 1
        )
        try:

            engine.analyse()

        except Exception as error:

            print(
                f"{utils.Colours.RED}Engine analysis failed at M = {M}: {error}"
                f"{utils.Colours.END}"
            )
            continue

        for j, stage in enumerate(engine.stages):

            flow_coefficients[i, j] = stage.phi
            stage_loading_coefficients[i, j] = stage.psi
            reactions[i, j] = stage.reaction

        jet_velocity_ratios[i] = engine.jet_velocity_ratio

    # plot compressor characteristic
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize = (12, 7)
    )

    #fig, ax = plt.subplots(figsize=(8, 4))
    for index in range(n_stages):

        ax_left.plot(
            flow_coefficients[:, index],
            stage_loading_coefficients[:, index],
            label = f"Stage {index + 1}",
            color = engine.stages[index].colour
        )

        ax_right.plot(
            flow_coefficients[:, index],
            reactions[:, index],
            color = engine.stages[index].colour
        )
        
    for index in range(n_stages):

        ax_left.plot(
            design_phi[index],
            design_psi[index],
            color = engine.stages[index].colour,
            linestyle = '',
            marker = '.', markersize = 8
        )

        ax_right.plot(
            design_phi[index],
            design_reaction[index],
            color = engine.stages[index].colour,
            linestyle = '',
            marker = '.', markersize = 8
        )

    ax_left.set_xlabel("Flow coefficient (ϕ)")
    ax_left.set_ylabel("Stage loading coefficient (ψ)")
    ax_left.set_title("Compressor stage characteristics")
    ax_left.grid()
    ax_left.legend()
    
    ax_right.set_xlabel("Flow coefficient (ϕ)")
    ax_right.set_ylabel("Reaction (Δ)")
    ax_right.grid()

    plt.show()

# run main() on running the script

if __name__ == "__main__":
    """Run main() on running the script."""
    # check if enough arguments are provided to declare number of stages immediately
    if len(sys.argv) > 1:

        try:

            # store number of stages and attempt to convert to integer
            utils.Defaults.no_of_stages = int(sys.argv[1])

        except Exception as error:

            # explain error if user provides a non-integer input
            print(
                f"{utils.Colours.RED}Please provide an integer argument for the number of stages!"
                f"{utils.Colours.END}\n{error}"
            )

    if len(sys.argv) > 2:

        try:

            # store number of annuli and attempt to convert to integer
            utils.Defaults.no_of_annuli = int(sys.argv[2])

        except Exception as error:

            # explain error if user provides a non-integer input
            print(
                f"{utils.Colours.RED}Please provide an integer argument for the number of annuli!"
                f"{utils.Colours.END}\n{error}"
            )

    if len(sys.argv) > 3:

        if sys.argv[3] == "v":

            # enter debug mode and switch off the loading bar
            utils.Defaults.debug = True
            utils.Defaults.loading_bar = False

        else:

            # explain error if user provides a non-integer input
            print(
                f"{utils.Colours.RED}Please provide the argument 'v' to enter debug mode!"
                f"{utils.Colours.END}"
            )

    # call main function
    main()
