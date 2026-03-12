# import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from time import perf_counter as timer

# import high speed solver
import utils
from engine import Engine
from flight_scenario import Flight_scenario

# Inputs class
class Inputs:

    # diameter
    diameter = 0.14
    hub_tip_ratio = 25 / 70

    # constants to match Hirono case
    no_of_stages = 1
    vortex_exponent = 0.5
    Y_p = 0
    area_ratio = 1

    # Hirono parameters to sweep
    altitudes = [0, 0, 3000]
    flight_speeds = [20, 0, 40]
    thrusts = [30, 50, 20]
    phis = [0.6, 0.59, 0.63]
    psis = [0.16, 0.18, 0.12]

    # labels
    labels = ["take_off", "static", "cruise"]

# create_engine function
def create_engine(
        altitude = utils.Defaults.altitude,
        flight_speed = utils.Defaults.flight_speed,
        diameter = utils.Defaults.diameter,
        hub_tip_ratio = utils.Defaults.hub_tip_ratio,
        thrust = utils.Defaults.thrust,
        no_of_stages = utils.Defaults.no_of_stages,
        phi = utils.Defaults.phi,
        psi = utils.Defaults.psi,
        vortex_exponent = utils.Defaults.vortex_exponent,
        Y_p = utils.Defaults.Y_p,
        area_ratio = utils.Defaults.area_ratio
    ):
    """Creates an engine with custom parameters."""
    # start timer
    t1 = timer()
    
    # create flight scenario
    flight_scenario = Flight_scenario(
        "", altitude, flight_speed, diameter, hub_tip_ratio, thrust
    )
    
    # create engine
    engine = Engine(
        flight_scenario, no_of_stages, phi, psi, vortex_exponent, Y_p, area_ratio
    )

    # end timer
    t2 = timer()
    print(f"Test engine created in {utils.Colours.GREEN}{t2 - t1:.4g}{utils.Colours.END} s.")

    return engine

def main():

    for index in [0, 1, 2]:

        Inputs.altitude = Inputs.altitudes[index]
        Inputs.flight_speed = Inputs.flight_speeds[index]
        Inputs.thrust = Inputs.thrusts[index]
        Inputs.phi = Inputs.phis[index]
        Inputs.psi = Inputs.psis[index]

        label = Inputs.labels[index]

        export_engine(label)

def export_engine(label):

    # create engine
    Inputs.engine = create_engine(
        altitude = Inputs.altitude,
        flight_speed = Inputs.flight_speed,
        diameter = Inputs.diameter,
        hub_tip_ratio = Inputs.hub_tip_ratio,
        thrust = Inputs.thrust,
        no_of_stages = Inputs.no_of_stages,
        phi = Inputs.phi,
        psi = Inputs.psi,
        vortex_exponent = Inputs.vortex_exponent,
        Y_p = Inputs.Y_p,
        area_ratio = Inputs.area_ratio
    )

    # add geometry
    Inputs.engine.geometry = {
        "aspect_ratio": utils.Defaults.aspect_ratio,
        "diffusion_factor": utils.Defaults.diffusion_factor,
        "design_parameter": utils.Defaults.design_parameter
    }

    # calculate blade angles and export engine
    Inputs.engine.empirical_design()
    Inputs.engine.export(f"Hirono_{label}")

if __name__ == "__main__":

    main()