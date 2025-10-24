# import modules

import matplotlib.pyplot as plt
import numpy as np

from engine_2025_10_11 import Engine
from blade_row_2025_10_11 import Blade_row
from flow_state_2025_10_11 import Flow_state
import utils_2025_10_11 as utils

# main function

def main():
    """Function to run on script execution."""
    # visualise colour options for convenience
    """colours = utils.Colours()
    print(colours)"""
    print(f"{utils.Colours.CYAN}Constructing engine for analysis...{utils.Colours.END}")

    # determine from user how many stages to construct the engine with
    print(f"{utils.Colours.RED}Please state the desired number of stages:{utils.Colours.END}")
    while True:

        user_input = input()
        try:

            no_of_stages = int(user_input)
            break

        except ValueError:

            print(f"{utils.Colours.RED}Error: Please provide a positive integer.{utils.Colours.END}")

    print(f"{utils.Colours.GREEN}{no_of_stages} stages selected!{utils.Colours.END}")

    # construct engine class for the appropriate number of stages
    engine = Engine(no_of_stages)

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

            # get new blade speed from user
            """print(f"{utils.Colours.RED}Please state the new blade speed (m/s):{utils.Colours.END}")
            while True:

                user_input = input()
                if user_input == "":
                    
                    blade_speed = engine.blade_rows[index - 1].blade_speed
                    break

                try:
                
                    blade_speed = float(user_input)
                    break
                
                except ValueError:
                
                    print(f"{utils.Colours.RED}Error: Please provide a valid number.{utils.Colours.END}")

            print(f"{utils.Colours.GREEN}Blade speed of {blade_speed:.3g} m/s selected!{utils.Colours.END}")

            # get new metal angle from user
            print(f"{utils.Colours.RED}Please state the new metal angle (°):{utils.Colours.END}")
            while True:

                user_input = input()
                if user_input == "":

                    metal_angle = engine.blade_rows[index - 1].metal_angle
                    break

                try:

                    metal_angle = utils.deg_to_rad(float(user_input))
                    break

                except ValueError:

                    print(f"{utils.Colours.RED}Error: Please provide a valid number.{utils.Colours.END}")

            print(f"{utils.Colours.GREEN}Metal angle of {utils.rad_to_deg(metal_angle):.3g} ° selected!{utils.Colours.END}")

            # get new tip radius from user
            print(f"{utils.Colours.RED}Please state the new tip radius (m):{utils.Colours.END}")
            while True:

                user_input = input()
                if user_input == "":
  
                    tip_radius = engine.blade_rows[index - 1].tip_radius
                    break

                try:

                    tip_radius = float(user_input)
                    break

                except ValueError:

                    print(f"{utils.Colours.RED}Error: Please provide a valid number.")

            print(f"{utils.Colours.GREEN}Tip radius of {tip_radius:.3g} m selected!{utils.Colours.END}")

            # get new hub radius from user
            print(f"{utils.Colours.RED}Please state the new hub radius (m):{utils.Colours.END}")
            while True:

                user_input = input()
                if user_input == "":

                    hub_radius = engine.blade_rows[index - 1].hub_radius
                    break

                try:

                    hub_radius = float(user_input)
                    break

                except ValueError:

                    print(f"{utils.Colours.RED}Error: Please provide a valid number.{utils.Colours.END}")

            print(f"{utils.Colours.GREEN}Hub radius of {hub_radius:.3g} m selected!{utils.Colours.END}")

            # get new stagnation pressure loss coefficient from user
            print(
                f"{utils.Colours.RED}Please state the stagnation pressure loss coefficient:{utils.Colours.END}"
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

                    print(f"{utils.Colours.RED}Error: Please provide a valid number.{utils.Colours.END}")
            
            print(
                f"{utils.Colours.GREEN}Stagnation pressure loss coefficient "
                f"of {Y_p:.3g} selected!{utils.Colours.END}"
            )

            # update blade row and print new engine summary
            engine.blade_rows[index - 1] = Blade_row(
                blade_speed,
                metal_angle,
                tip_radius,
                hub_radius,
                Y_p
            )"""
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

    # prepare to sweep over velocities and preallocate storage
    vv = np.linspace(utils.Defaults.inlet_flow_coefficient / 2, 3 * utils.Defaults.inlet_flow_coefficient / 2, 5)
    n_stages = len(engine.stages)
    flow_coefficients = np.zeros((len(vv), n_stages))
    stage_loading_coefficients = np.zeros((len(vv), n_stages))
    jet_velocity_ratios = np.zeros((len(vv), 1))
    thrust_coefficients = np.zeros((len(vv), 1))

    # sweep over velocities and store flow and stage loading coefficients
    for i, phi in enumerate(vv):

        engine.intake.inlet = Flow_state(
            utils.Defaults.inlet_Mach_number,
            phi, utils.Defaults.inlet_swirl, 1, 1
        )
        engine.analyse()

        for j, stage in enumerate(engine.stages):

            flow_coefficients[i, j] = stage.phi
            stage_loading_coefficients[i, j] = stage.psi

        jet_velocity_ratios[i] = engine.jet_velocity_ratio

    # plot compressor characteristic
    fig, (ax_upper, ax_lower) = plt.subplots(
        1, 2, figsize = (12, 7)
    )

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
            design_phi[index],
            design_psi[index],
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