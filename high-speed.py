# import modules

import sys
from time import perf_counter as timer

import matplotlib.pyplot as plt
import numpy as np

from engine import Engine
from flight_scenario import Flight_scenario
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
        utils.Defaults.phi,
        utils.Defaults.psi,
        utils.Defaults.vortex_exponent,
        utils.Defaults.Y_p,
        utils.Defaults.area_ratio
    )

    # design engine
    engine.design()
    print(engine)

    # add geometry
    engine.geometry = {
        "aspect_ratio": utils.Defaults.aspect_ratio,
        "diffusion_factor": utils.Defaults.diffusion_factor,
        "design_parameter": utils.Defaults.design_parameter
    }

    # calculate blade angles and export engine
    engine.empirical_design()
    engine.export()

# on running the script
if __name__ == "__main__":

    # start timer
    t1 = timer()

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

        if sys.argv[2] == "v":

            # enter debug mode and switch off the loading bar
            utils.Defaults.debug = True

        else:

            # explain error if user provides a the wrong input
            print(
                f"{utils.Colours.RED}Please provide the argument 'v' to enter debug mode!"
                f"{utils.Colours.END}"
            )

    # main function
    main()

    # end timer
    t2 = timer()
    print(f"High-speed.py script completed in {utils.Colours.GREEN}{t2 - t1:.4g}{utils.Colours.END}")
