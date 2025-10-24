# import modules

import matplotlib.pyplot as plt
import numpy as np

from engine import Engine
from flow_state import Flow_state
import utils

# main function

def main():
    """Function to run on script execution."""
    # visualise colour options for convenience
    """colours = utils.Colours()
    print(colours)"""

    # determine from user how many stages to construct the engine with
    print(f"{utils.Colours.RED}Please state the desired number of stages:{utils.Colours.END}")
    while True:

        user_input = input()
        try:

            no_of_stages = int(user_input)
            break

        except ValueError:

            print(f"{utils.Colours.RED}Error: Please provide a positive integer.{utils.Colours.END}")

    # user feedback
    print(f"{utils.Colours.GREEN}{no_of_stages} stages selected!{utils.Colours.END}")
    print(f"{utils.Colours.CYAN}Constructing engines for analysis...{utils.Colours.END}")

    # store candidate engines in a list
    candidate_engines = []
    for M in np.linspace(utils.Defaults.M_min, utils.Defaults.M_max, utils.Defaults.N):

        candidate_engines.append(Engine(no_of_stages, M))

    # from list of candidate engines, store key parameters
    jet_velocity_ratios = [engine.jet_velocity_ratio for engine in candidate_engines]
    inlet_Mach_numbers = [engine.M_design for engine in candidate_engines]
    thrust_coefficients = [engine.C_T for engine in candidate_engines]
    efficiencies = [engine.eta_p for engine in candidate_engines]
    nozzle_pressure_ratios = [engine.nozzle_p_r for engine in candidate_engines]

    # plot key parameters of the candidate engines
    fig, ax = plt.subplots()
    ax.plot(jet_velocity_ratios, thrust_coefficients, label = "Thrust coefficient")
    ax.plot(jet_velocity_ratios, efficiencies, label = "Polytropic efficiency")
    ax.plot(jet_velocity_ratios, nozzle_pressure_ratios, label = "Nozzle pressure ratio")
    ax.plot(jet_velocity_ratios, inlet_Mach_numbers, label = "Inlet Mach number")
    ax.grid()
    ax.legend()
    plt.show()

    # determine from user which of the candidate engines to analyse further
    print(f"{utils.Colours.RED}Please state the desired jet velocity ratio:{utils.Colours.END}")
    while True:

        user_input = input()
        try:

            design_epsilon = float(user_input)
            break

        except ValueError:

            print(f"{utils.Colours.RED}Error: Please provide a valid jet velocity ratio.{utils.Colours.END}")

    # single out specified engine from list of candidate engines
    print(f"{utils.Colours.GREEN}Jet velocity ratio of {design_epsilon} selected!{utils.Colours.END}")
    design_index = min(enumerate(jet_velocity_ratios), key=lambda x: abs(x[1] - design_epsilon))[0]
    engine = candidate_engines[design_index]

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

    print(f"{utils.Colours.GREEN}Blade configurations stored!{utils.Colours.END}")

    # analyse engine
    print(f"{utils.Colours.GREEN}Analysing engine...{utils.Colours.END}")
    engine.analyse()
    engine.collect_flow_states()
    engine.visualise_velocity_triangles()

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
        engine.analyse()

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
    ax_right.legend()

    plt.show()

# run main() on running the script

if __name__ == "__main__":
    """Run main() on running the script."""
    main()